import argparse

import numpy as np
import pandas as pd
import hdbscan
from sklearn.cluster import DBSCAN

import lidar_utils

# ─────────────────────────────────────────────
# CONFIGURATION DES CLASSES
# ─────────────────────────────────────────────

CLASS_MAP = {
    (38, 23, 180):  {"id": 0, "label": "Antenna"},
    (177, 132, 47): {"id": 1, "label": "Cable"},
    (129, 81, 97):  {"id": 2, "label": "Electric pole"},
    (66, 132, 9):   {"id": 3, "label": "Wind turbine"}
}

BACKGROUND_ID    = 4
BACKGROUND_LABEL = "background"

# Couleurs background identifiées par visualisation
BACKGROUND_COLORS = {
    (10, 78, 149):  "vegetation",  # Arbres → confusion principale avec Antenna
    (146, 61, 0):   "terrain",     # Sol / rochers
}

MAX_BACKGROUND_CLUSTERS_PER_COLOR = 10  # Augmenté (était 5)
MAX_POINTS_DBSCAN = 5000

# Seuil de fusion adaptatif par classe (en mètres)
XY_MERGE_THRESHOLD = {
    0: 10.0,   # Antenna       → fragments verticaux proches
    1: 30.0,   # Cable         → un câble s'étend sur des dizaines de mètres
    2:  8.0,   # Electric pole
    3: 25.0,   # Wind turbine  → pales très larges
}

# ─────────────────────────────────────────────
# RÈGLES GÉOMÉTRIQUES PAR CLASSE
# Basées sur les stats réelles du dataset :
#   Antenna      : height=33m, width=7m   → vertical compact
#   Cable        : height=0.79m, length=7m → quasi horizontal
#   Electric pole: height=29m, width=5m   → vertical fin
#   Wind turbine : height=59m, length=38m → grand dans toutes les dimensions
# ─────────────────────────────────────────────

CLASS_RULES = {
    0: {  # Antenna
        "label":                     "Antenna",
        "min_cluster_size":           10,
        "min_samples":                 5,
        "cluster_selection_epsilon":   3.0,
        "min_height":                  5.0,
        "max_height":                170.0,
        "max_footprint":             200.0,
        "min_elongation":              1.5,   # height/max(w,l) > 1.5 → vertical
    },
    1: {  # Cable
        "label":                     "Cable",
        "min_cluster_size":            3,
        "min_samples":                 2,
        "cluster_selection_epsilon":  20.0,   # Grand epsilon → un câble entier par cluster
        "max_height":                 20.0,
        "min_length":                  2.0,   # Un câble fait au moins 2m
        "max_flatness":                0.3,   # Très asymétrique (long et fin)
        "min_fragment":                0.01,
        "z_scale":                     0.1,
    },
    2: {  # Electric pole
        "label":                     "Electric pole",
        "min_cluster_size":            5,
        "min_samples":                 3,
        "cluster_selection_epsilon":   3.0,
        "min_height":                  3.0,
        "max_height":                 80.0,
        "min_elongation":              1.0,
        "max_footprint":             300.0,
    },
    3: {  # Wind turbine
        "label":                     "Wind turbine",
        "min_cluster_size":           15,
        "min_samples":                 8,
        "cluster_selection_epsilon":   6.0,
        "min_height":                 15.0,
        "max_height":                200.0,
        "min_footprint":              10.0,
    },
}


# ─────────────────────────────────────────────
# CALCUL BOUNDING BOX ORIENTÉE
# ─────────────────────────────────────────────

def calculate_oriented_bbox(pts):
    """Calcule la boîte englobante orientée (centre, dimensions, yaw) via PCA."""
    pts_xy = pts[:, :2]
    centroid_xy = pts_xy.mean(axis=0)
    pts_centered = pts_xy - centroid_xy

    cov = np.cov(pts_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    main_axis = eigenvectors[:, np.argmax(eigenvalues)]
    yaw = np.arctan2(main_axis[1], main_axis[0])

    cos_y, sin_y = np.cos(-yaw), np.sin(-yaw)
    rotation_matrix = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
    pts_rotated_xy = pts_centered @ rotation_matrix.T

    min_xy, max_xy = pts_rotated_xy.min(axis=0), pts_rotated_xy.max(axis=0)

    local_center_xy = (min_xy + max_xy) / 2
    inv_rot = np.array([[cos_y, sin_y], [-sin_y, cos_y]])
    global_center_xy = local_center_xy @ inv_rot.T + centroid_xy

    return {
        "cx": global_center_xy[0],
        "cy": global_center_xy[1],
        "cz": (pts[:, 2].max() + pts[:, 2].min()) / 2,
        "w":  max_xy[1] - min_xy[1],
        "l":  max_xy[0] - min_xy[0],
        "h":  pts[:, 2].max() - pts[:, 2].min(),
        "yaw": yaw
    }


def calculate_cable_bbox(pts):
    """
    Bbox spécialisée pour les câbles :
    - PCA sur XY uniquement → axe principal = direction du câble
    - L = longueur le long du câble, W = épaisseur perpendiculaire
    - H = vrai Z max - Z min
    """
    pts_xy = pts[:, :2]
    centroid = pts_xy.mean(axis=0)
    centered = pts_xy - centroid

    cov = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Axe principal = direction du câble (plus grande variance)
    main_axis = eigenvectors[:, np.argmax(eigenvalues)]
    perp_axis = eigenvectors[:, np.argmin(eigenvalues)]
    yaw = np.arctan2(main_axis[1], main_axis[0])

    # Projeter sur les deux axes
    along = centered @ main_axis
    perp  = centered @ perp_axis

    bbox_l = along.max() - along.min()
    bbox_w = perp.max()  - perp.min()

    # Centre global
    center_along = (along.max() + along.min()) / 2
    center_perp  = (perp.max()  + perp.min())  / 2
    cx = centroid[0] + center_along * main_axis[0] + center_perp * perp_axis[0]
    cy = centroid[1] + center_along * main_axis[1] + center_perp * perp_axis[1]

    return {
        "cx": cx, "cy": cy,
        "cz": (pts[:, 2].max() + pts[:, 2].min()) / 2,
        "w":  bbox_w,
        "l":  bbox_l,
        "h":  pts[:, 2].max() - pts[:, 2].min(),
        "yaw": yaw,
    }


# ─────────────────────────────────────────────
# RÈGLES GÉOMÉTRIQUES
# ─────────────────────────────────────────────

def passes_geometric_rules(bbox, rules, verbose=False):
    """
    Vérifie que la bbox respecte les règles géométriques de sa classe.
    Retourne (True, "") si valide, (False, raison) sinon.
    """
    w, l, h   = bbox["w"], bbox["l"], bbox["h"]
    max_wl    = max(max(w, l), 1e-6)
    min_wl    = min(w, l)
    footprint = w * l
    elongation = h / max_wl
    flatness   = min_wl / max_wl

    min_frag = rules.get("min_fragment", 0.5)

    checks = [
        (h  < rules.get("min_height",    0),      f"height {h:.1f} < min {rules.get('min_height',0)}"),
        (h  > rules.get("max_height", 9999),      f"height {h:.1f} > max {rules.get('max_height',9999)}"),
        (w  < rules.get("min_width",     0),      f"width {w:.1f} < min {rules.get('min_width',0)}"),
        (l  < rules.get("min_length",    0),      f"length {l:.1f} < min {rules.get('min_length',0)}"),
        (footprint > rules.get("max_footprint", 9999), f"footprint {footprint:.1f} > max {rules.get('max_footprint',9999)}"),
        (footprint < rules.get("min_footprint",    0), f"footprint {footprint:.1f} < min {rules.get('min_footprint',0)}"),
        (elongation < rules.get("min_elongation",  0), f"elongation {elongation:.2f} < min {rules.get('min_elongation',0)}"),
        ("max_flatness" in rules and flatness > rules["max_flatness"], f"flatness {flatness:.2f} > max {rules.get('max_flatness',1)}"),
        (min(w, l) < min_frag,                    f"fragment min(w,l)={min(w,l):.2f} < {min_frag}"),
        (bbox["cz"] < -50.0,                      f"artefact cz={bbox['cz']:.1f} < -50"),
    ]

    for failed, reason in checks:
        if failed:
            if verbose:
                print(f"        ✗ Rejeté : {reason}")
            return False

    return True


# ─────────────────────────────────────────────
# FUSION DES CLUSTERS PROCHES (même classe)
# ─────────────────────────────────────────────

def merge_close_clusters(clusters):
    """
    Fusionne les clusters de même classe dont les centres XY sont proches.
    Utilise un seuil adaptatif par classe (XY_MERGE_THRESHOLD).
    Après fusion, re-applique les règles géométriques sur la bbox résultante.
    """
    if len(clusters) <= 1:
        return clusters

    merged = []
    used = [False] * len(clusters)

    for i, c1 in enumerate(clusters):
        if used[i]:
            continue

        group_pts = [c1["pts"]]
        used[i] = True
        class_id = c1["class_id"]
        threshold = XY_MERGE_THRESHOLD.get(class_id, 10.0)

        for j, c2 in enumerate(clusters):
            if used[j] or c2["class_id"] != class_id:
                continue

            dx = c1["bbox"]["cx"] - c2["bbox"]["cx"]
            dy = c1["bbox"]["cy"] - c2["bbox"]["cy"]
            if np.sqrt(dx**2 + dy**2) < threshold:
                group_pts.append(c2["pts"])
                used[j] = True

        all_pts = np.vstack(group_pts)
        bbox = calculate_oriented_bbox(all_pts)

        # Re-appliquer les règles sur la bbox fusionnée
        rules = CLASS_RULES[class_id]
        if not passes_geometric_rules(bbox, rules):
            continue

        merged.append({
            "pts":         all_pts,
            "bbox":        bbox,
            "class_id":    class_id,
            "class_label": c1["class_label"],
        })

    return merged


# ─────────────────────────────────────────────
# CLUSTERING PAR CLASSE (HDBSCAN)
# ─────────────────────────────────────────────

def cluster_class(pts_class, class_id, verbose=False):
    """
    Applique HDBSCAN avec les règles géométriques de la classe.
    Pour les câbles, clustering en 2D (XY) puis reconstruction Z réelle.
    """
    rules   = CLASS_RULES[class_id]
    results = []
    rejected_counts = {}

    if len(pts_class) < rules["min_cluster_size"]:
        return results

    # Sous-échantillonnage
    if len(pts_class) > MAX_POINTS_DBSCAN:
        step = len(pts_class) // MAX_POINTS_DBSCAN
        pts_class = pts_class[::step]

    # Câbles : réduire l'influence de Z (clustering horizontal)
    # au lieu d'ignorer Z complètement (évite de mélanger câbles à hauteurs différentes)
    z_scale = rules.get("z_scale", 1.0)
    if z_scale != 1.0:
        pts_for_clustering = pts_class.copy()
        pts_for_clustering[:, 2] = pts_for_clustering[:, 2] * z_scale
    else:
        pts_for_clustering = pts_class

    try:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=rules["min_cluster_size"],
            min_samples=rules["min_samples"],
            cluster_selection_epsilon=rules["cluster_selection_epsilon"],
            core_dist_n_jobs=-1,
        )
        labels = clusterer.fit_predict(pts_for_clustering)
    except Exception:
        try:
            labels = DBSCAN(
                eps=rules["cluster_selection_epsilon"],
                min_samples=rules["min_samples"],
                n_jobs=-1
            ).fit_predict(pts_class)
        except MemoryError:
            return results

    unique_labels = [l for l in set(labels) if l != -1]

    for label in unique_labels:
        mask   = labels == label
        # Toujours utiliser les pts 3D originaux pour la bbox
        pts_obj = pts_class[mask]
        if len(pts_obj) < 3:
            continue

        # Bbox spécialisée câble (PCA sur XY uniquement → axe = direction du câble)
        if class_id == 1:
            bbox = calculate_cable_bbox(pts_obj)
        else:
            bbox = calculate_oriented_bbox(pts_obj)

        if not passes_geometric_rules(bbox, rules, verbose=verbose):
            continue

        results.append({
            "pts":         pts_obj,
            "bbox":        bbox,
            "class_id":    class_id,
            "class_label": rules["label"],
        })

    if verbose and rejected_counts:
        for reason, count in rejected_counts.items():
            print(f"      Rejeté ({count}x) : {reason}")

    return results


# ─────────────────────────────────────────────
# DÉTECTION CÂBLES PAR LINÉARITÉ
# ─────────────────────────────────────────────

def detect_cables_by_linearity(pts_cable, pole_cx, pole_cy, angle_gap_threshold=10.0, min_pts=5, verbose=False):
    """
    Sépare les câbles par leur angle depuis le pylône.
    Chaque câble part dans une direction unique depuis le pylône.
    1. Calculer l'angle de chaque point par rapport au pylône
    2. Trier par angle → détecter les ruptures
    3. Une bbox par groupe angulaire = un câble
    """
    if len(pts_cable) < min_pts:
        return []

    # Calculer l'angle de chaque point depuis le pylône
    angles = np.degrees(np.arctan2(
        pts_cable[:, 1] - pole_cy,
        pts_cable[:, 0] - pole_cx
    ))

    # Trier par angle
    sort_a   = np.argsort(angles)
    pts_a    = pts_cable[sort_a]
    ang_s    = angles[sort_a]

    # Détecter les ruptures angulaires → câbles distincts
    ang_gaps = np.diff(ang_s)
    splits   = np.where(ang_gaps > angle_gap_threshold)[0] + 1
    groups   = np.split(pts_a, splits)

    if verbose:
        print(f"   Câbles : {len(groups)} groupes angulaires (gap={angle_gap_threshold}°)")

    results = []
    for g_idx, group in enumerate(groups):
        if len(group) < min_pts:
            continue

        bbox = calculate_cable_bbox(group)

        if verbose:
            ang_min = ang_s[sort_a][0] if len(group) == len(pts_cable) else ang_s[splits[g_idx-1]] if g_idx > 0 else ang_s[0]
            print(f"     Câble {g_idx} : {len(group)} pts, L={bbox['l']:.1f}m, W={bbox['w']:.2f}m")

        if bbox["l"] < 1.0:
            continue

        results.append({
            "pts":         group,
            "bbox":        bbox,
            "class_id":    1,
            "class_label": "Cable",
        })

    return results


# ─────────────────────────────────────────────
# CLUSTERING DES OBSTACLES (avec couleurs)
# ─────────────────────────────────────────────

CABLE_Z_MARGIN             = 8.0   # Marge Z autour du sommet des pylônes
CABLE_POLE_EXCLUSION_RADIUS = 12.0  # Rayon XY d'exclusion autour du corps du pylône
CABLE_GAP_THRESHOLD         = 5.0   # Gap en mètres entre deux câbles distincts

def cluster_obstacles(df_frame, verbose=False):
    """
    Cluster les points labelisés par classe avec HDBSCAN + règles géométriques.

    Ordre de traitement :
    1. Pylônes (Electric pole) → détectés en premier
    2. Câbles → filtrés par bande Z autour du sommet des pylônes détectés
    3. Antennes + Éoliennes → traitement standard

    Retourne une liste de dicts {pts, bbox, class_id, class_label}.
    """
    results = []

    # ── 1. Electric pole en premier ──────────────────
    pole_rgb = (129, 81, 97)
    pole_pts = df_frame.loc[
        (df_frame['r'] == pole_rgb[0]) &
        (df_frame['g'] == pole_rgb[1]) &
        (df_frame['b'] == pole_rgb[2]),
        ['x', 'y', 'z']
    ].to_numpy()

    pole_clusters = []
    if len(pole_pts) >= 3:
        pole_clusters = cluster_class(pole_pts, 2, verbose=verbose)
        results.extend(pole_clusters)

    # ── 2. Câbles guidés par les pylônes ─────────────
    cable_rgb = (177, 132, 47)
    cable_pts = df_frame.loc[
        (df_frame['r'] == cable_rgb[0]) &
        (df_frame['g'] == cable_rgb[1]) &
        (df_frame['b'] == cable_rgb[2]),
        ['x', 'y', 'z']
    ].to_numpy()

    if len(cable_pts) >= 3:
        if pole_clusters:
            # Calculer les bandes Z à partir du sommet des pylônes
            pole_z_bands = []
            pole_centers_xy = []
            for pole in pole_clusters:
                z_top = pole["bbox"]["cz"] + pole["bbox"]["h"] / 2
                pole_z_bands.append((z_top - CABLE_Z_MARGIN, z_top + CABLE_Z_MARGIN))
                pole_centers_xy.append((pole["bbox"]["cx"], pole["bbox"]["cy"]))

            # Filtre 1 : bande Z autour du sommet des pylônes
            z_mask = np.zeros(len(cable_pts), dtype=bool)
            for z_min, z_max in pole_z_bands:
                z_mask |= (cable_pts[:, 2] >= z_min) & (cable_pts[:, 2] <= z_max)

            # Filtre 2 : exclure les points trop proches du corps du pylône (en XY)
            pole_mask = np.ones(len(cable_pts), dtype=bool)
            for px, py in pole_centers_xy:
                dist_xy = np.sqrt((cable_pts[:, 0] - px)**2 + (cable_pts[:, 1] - py)**2)
                pole_mask &= dist_xy > CABLE_POLE_EXCLUSION_RADIUS

            cable_pts_filtered = cable_pts[z_mask & pole_mask]
            if verbose:
                print(f"   Câbles : {len(cable_pts)} → {len(cable_pts[z_mask])} (filtre Z) → {len(cable_pts_filtered)} pts (excl. pylône)")
        else:
            cable_pts_filtered = cable_pts
            if verbose:
                print(f"   Câbles : aucun pylône détecté → utilisation de tous les {len(cable_pts)} pts")

        # Centre XY du pylône le plus proche (pour séparation angulaire)
        if pole_clusters:
            ref_pole = pole_clusters[0]
            pole_cx  = ref_pole["bbox"]["cx"]
            pole_cy  = ref_pole["bbox"]["cy"]
        else:
            pole_cx, pole_cy = 0.0, 0.0

        cable_clusters = detect_cables_by_linearity(
            cable_pts_filtered,
            pole_cx=pole_cx,
            pole_cy=pole_cy,
            angle_gap_threshold=10.0,
            min_pts=5,
            verbose=verbose
        )
        results.extend(cable_clusters)

    # ── 3. Antennes et éoliennes ──────────────────────
    for rgb, info in CLASS_MAP.items():
        if info["id"] in [1, 2]:  # Déjà traités
            continue

        mask = (
            (df_frame['r'] == rgb[0]) &
            (df_frame['g'] == rgb[1]) &
            (df_frame['b'] == rgb[2])
        )
        pts_class = df_frame.loc[mask, ['x', 'y', 'z']].to_numpy()

        if len(pts_class) < 3:
            continue

        clusters = cluster_class(pts_class, info["id"], verbose=verbose)
        results.extend(clusters)

    # Fusion des clusters proches (même objet découpé) + re-validation géométrique
    # Les câbles sont exclus car déjà bien séparés par angle
    non_cable  = [c for c in results if c["class_id"] != 1]
    cables     = [c for c in results if c["class_id"] == 1]
    non_cable  = merge_close_clusters(non_cable)
    results    = non_cable + cables
    return results


# ─────────────────────────────────────────────
# EXTRACTION DU BACKGROUND (par couleur ciblée)
# ─────────────────────────────────────────────

def extract_background_clusters(df_frame, n_clusters=MAX_BACKGROUND_CLUSTERS_PER_COLOR):
    """
    Extrait des clusters de background ciblés par couleur :
    - Végétation (10, 78, 149) → vrais arbres (filtrés z > 2m)
    - Terrain    (146, 61, 0)  → sol / rochers (filtrés z > 0.5m)
    """
    # Hauteur minimale par type de background
    MIN_Z = {
        "vegetation": 2.0,   # Arbres : au moins 2m de haut
        "terrain":    0.5,   # Rochers : au moins 0.5m
    }

    results = []

    for rgb, bg_type in BACKGROUND_COLORS.items():
        mask = (
            (df_frame['r'] == rgb[0]) &
            (df_frame['g'] == rgb[1]) &
            (df_frame['b'] == rgb[2])
        )
        bg_pts = df_frame.loc[mask, ['x', 'y', 'z']].to_numpy()

        if len(bg_pts) < 50:
            continue

        # Filtrer par hauteur minimale → exclure le sol plat
        z_min = MIN_Z.get(bg_type, 0.0)
        bg_pts = bg_pts[bg_pts[:, 2] > z_min]

        if len(bg_pts) < 50:
            continue

        # Sous-échantillonner
        if len(bg_pts) > MAX_POINTS_DBSCAN * 3:
            idx = np.random.choice(len(bg_pts), MAX_POINTS_DBSCAN * 3, replace=False)
            bg_pts = bg_pts[idx]

        try:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=20,
                min_samples=10,
                cluster_selection_epsilon=3.0,
                core_dist_n_jobs=-1,
            )
            labels = clusterer.fit_predict(bg_pts)
        except Exception:
            continue

        unique_labels = [l for l in set(labels) if l != -1]
        if not unique_labels:
            continue

        np.random.shuffle(unique_labels)
        selected = unique_labels[:n_clusters]

        for label in selected:
            pts_obj = bg_pts[labels == label]
            if len(pts_obj) < 10:
                continue

            bbox = calculate_oriented_bbox(pts_obj)
            results.append({
                "pts":         pts_obj,
                "bbox":        bbox,
                "class_id":    BACKGROUND_ID,
                "class_label": BACKGROUND_LABEL,
                "bg_type":     bg_type,
            })

        print(f"      {bg_type}: {len(selected)} clusters extraits")

    return results


# ─────────────────────────────────────────────
# MAIN (pour tester sur une frame)
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file",       required=True)
    parser.add_argument("--pose-index", type=int, default=0)
    parser.add_argument("--verbose",    action="store_true", help="Afficher les clusters rejetés")
    parser.add_argument("--export-csv", default=None, help="Exporter les bboxes dans un CSV pour visualisation")
    args = parser.parse_args()

    try:
        df = lidar_utils.load_h5_data(args.file)
        pose_counts = lidar_utils.get_unique_poses(df)

        if args.pose_index >= len(pose_counts):
            print(f"Erreur: Index {args.pose_index} invalide (Max: {len(pose_counts)-1})")
            return

        selected_pose = pose_counts.iloc[args.pose_index]
        df_frame = lidar_utils.filter_by_pose(df, selected_pose)
        df_frame = df_frame[df_frame["distance_cm"] > 0].copy()

        xyz = lidar_utils.spherical_to_local_cartesian(df_frame)
        df_frame[['x', 'y', 'z']] = xyz

        print(f"\n--- OBSTACLES (POSE #{args.pose_index}) ---")
        obstacle_clusters = cluster_obstacles(df_frame, verbose=args.verbose)

        results = []
        csv_rows = []
        for c in obstacle_clusters:
            bbox = c["bbox"]
            results.append({
                "Classe":   c["class_label"],
                "Centre_X": f"{bbox['cx']:.2f}",
                "Centre_Y": f"{bbox['cy']:.2f}",
                "Centre_Z": f"{bbox['cz']:.2f}",
                "W":        f"{bbox['w']:.2f}",
                "L":        f"{bbox['l']:.2f}",
                "H":        f"{bbox['h']:.2f}",
                "Yaw":      f"{bbox['yaw']:.3f}",
            })
            # Format compatible visualize_predictions.py
            csv_rows.append({
                "ego_x":         selected_pose["ego_x"],
                "ego_y":         selected_pose["ego_y"],
                "ego_z":         selected_pose["ego_z"],
                "ego_yaw":       selected_pose["ego_yaw"],
                "bbox_center_x": bbox["cx"],
                "bbox_center_y": bbox["cy"],
                "bbox_center_z": bbox["cz"],
                "bbox_width":    bbox["w"],
                "bbox_length":   bbox["l"],
                "bbox_height":   bbox["h"],
                "bbox_yaw":      bbox["yaw"],
                "class_ID":      c["class_id"],
                "class_label":   c["class_label"],
            })

        if results:
            print(pd.DataFrame(results).to_string(index=False))
        else:
            print("Aucun obstacle trouvé.")

        # Export CSV pour visualisation
        if args.export_csv and csv_rows:
            pd.DataFrame(csv_rows).to_csv(args.export_csv, index=False)
            print(f"\n✅ CSV exporté → {args.export_csv}")

        print("\n--- BACKGROUND ---")
        bg_clusters = extract_background_clusters(df_frame)
        print(f"Total : {len(bg_clusters)} clusters background extraits")

    except Exception as e:
        print(f"Erreur : {e}")
        raise


if __name__ == "__main__":
    main()
