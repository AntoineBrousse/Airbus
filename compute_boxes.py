import argparse

import numpy as np
import pandas as pd
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

# Classe background
BACKGROUND_ID    = 4
BACKGROUND_LABEL = "background"

# Paramètres DBSCAN par classe
DBSCAN_PARAMS = {
    0: {"eps": 5.0, "min_samples": 10},  # Antenna
    1: {"eps": 1.2, "min_samples": 3},   # Cable
    2: {"eps": 4.0, "min_samples": 5},   # Electric pole
    3: {"eps": 8.0, "min_samples": 10},  # Wind turbine
}

# Paramètres DBSCAN pour le background
DBSCAN_BACKGROUND = {"eps": 3.0, "min_samples": 20}

# Nb max de clusters background à extraire par frame
MAX_BACKGROUND_CLUSTERS = 5

# Sous-échantillonnage max avant DBSCAN
MAX_POINTS_DBSCAN = 5000


# ─────────────────────────────────────────────
# CALCUL BOUNDING BOX ORIENTÉE
# ─────────────────────────────────────────────

def calculate_oriented_bbox(pts):
    """Calcule la boîte englobante orientée (centre, dimensions, yaw)."""
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


# ─────────────────────────────────────────────
# CLUSTERING DES OBSTACLES (avec couleurs)
# ─────────────────────────────────────────────

def cluster_obstacles(df_frame):
    """
    Cluster les points labelisés par classe.
    Retourne une liste de dicts {pts, class_id, class_label}.
    Utilise les eps par classe au lieu d'un eps global.
    """
    results = []

    for rgb, info in CLASS_MAP.items():
        mask = (
            (df_frame['r'] == rgb[0]) &
            (df_frame['g'] == rgb[1]) &
            (df_frame['b'] == rgb[2])
        )
        pts_class = df_frame.loc[mask, ['x', 'y', 'z']].to_numpy()

        if len(pts_class) < 10:
            continue

        # Sous-échantillonnage si trop de points
        if len(pts_class) > MAX_POINTS_DBSCAN:
            step = len(pts_class) // MAX_POINTS_DBSCAN
            pts_class = pts_class[::step]

        params = DBSCAN_PARAMS[info["id"]]

        try:
            clusters = DBSCAN(
                eps=params["eps"],
                min_samples=params["min_samples"],
                n_jobs=-1
            ).fit_predict(pts_class)
        except MemoryError:
            print(f"      MemoryError pour {info['label']} → skip")
            continue

        for cluster_id in set(clusters):
            if cluster_id == -1:
                continue
            pts_obj = pts_class[clusters == cluster_id]
            if len(pts_obj) < 5:
                continue
            # Calculer bbox
            bbox = calculate_oriented_bbox(pts_obj)

            # Filtre fragments : plus petite dimension < 0.5m
            if min(bbox["w"], bbox["l"]) < 0.5:
                continue

            # Filtre artefacts sous le sol
            if bbox["cz"] < -50.0:
                continue

            results.append({
                "pts":         pts_obj,
                "bbox":        bbox,
                "class_id":    info["id"],
                "class_label": info["label"],
            })

    return results


# ─────────────────────────────────────────────
# EXTRACTION DU BACKGROUND
# ─────────────────────────────────────────────

def extract_background_clusters(df_frame, n_clusters=MAX_BACKGROUND_CLUSTERS):
    """
    Extrait des clusters de points NON labelisés (sol, arbres, rochers)
    pour entraîner le classifieur à rejeter le background.

    Stratégie :
    - Prendre les points dont RGB ne correspond à aucune classe obstacle
    - Appliquer DBSCAN dessus
    - Garder N clusters aléatoires comme exemples négatifs
    """
    # Masque : points qui NE sont PAS des obstacles
    obstacle_mask = np.zeros(len(df_frame), dtype=bool)
    for rgb in CLASS_MAP.keys():
        obstacle_mask |= (
            (df_frame['r'] == rgb[0]) &
            (df_frame['g'] == rgb[1]) &
            (df_frame['b'] == rgb[2])
        ).values

    bg_pts = df_frame.loc[~obstacle_mask, ['x', 'y', 'z']].to_numpy()

    if len(bg_pts) < 100:
        return []

    # Sous-échantillonner fortement le background
    # (il y a beaucoup plus de points de sol que d'obstacles)
    max_bg_pts = MAX_POINTS_DBSCAN * 3
    if len(bg_pts) > max_bg_pts:
        idx = np.random.choice(len(bg_pts), max_bg_pts, replace=False)
        bg_pts = bg_pts[idx]

    # Filtrer les points trop proches du sol (z très bas)
    # On veut des exemples de végétation/rochers, pas juste du sol plat
    bg_pts = bg_pts[bg_pts[:, 2] > -10.0]

    if len(bg_pts) < 50:
        return []

    try:
        clusters = DBSCAN(
            eps=DBSCAN_BACKGROUND["eps"],
            min_samples=DBSCAN_BACKGROUND["min_samples"],
            n_jobs=-1
        ).fit_predict(bg_pts)
    except MemoryError:
        return []

    unique_labels = [l for l in set(clusters) if l != -1]
    if not unique_labels:
        return []

    # Sélectionner N clusters aléatoires
    np.random.shuffle(unique_labels)
    selected = unique_labels[:n_clusters]

    results = []
    for label in selected:
        pts_obj = bg_pts[clusters == label]
        if len(pts_obj) < 10:
            continue
        bbox = calculate_oriented_bbox(pts_obj)

        results.append({
            "pts":         pts_obj,
            "bbox":        bbox,
            "class_id":    BACKGROUND_ID,
            "class_label": BACKGROUND_LABEL,
        })

    return results


# ─────────────────────────────────────────────
# MAIN (pour tester sur une frame)
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--pose-index", type=int, default=0)
    args = parser.parse_args()

    try:
        df = lidar_utils.load_h5_data(args.file)
        pose_counts = lidar_utils.get_unique_poses(df)

        if args.pose_index >= len(pose_counts):
            print(f"Erreur: Index {args.pose_index} invalide (Max: {len(pose_counts)-1})")
            return

        selected_pose = pose_counts.iloc[args.pose_index]
        df_frame = lidar_utils.filter_by_pose(df, selected_pose)

        # Filtrer points invalides
        df_frame = df_frame[df_frame["distance_cm"] > 0].copy()

        xyz = lidar_utils.spherical_to_local_cartesian(df_frame)
        df_frame[['x', 'y', 'z']] = xyz

        print(f"\n--- OBSTACLES (POSE #{args.pose_index}) ---")
        obstacle_clusters = cluster_obstacles(df_frame)

        results = []
        for c in obstacle_clusters:
            bbox = c["bbox"]
            results.append({
                "Classe":   c["class_label"],
                "Centre_X": f"{bbox['cx']:.2f}",
                "Centre_Y": f"{bbox['cy']:.2f}",
                "Centre_Z": f"{bbox['cz']:.2f}",
                "W": f"{bbox['w']:.2f}",
                "L": f"{bbox['l']:.2f}",
                "H": f"{bbox['h']:.2f}",
                "Yaw":      f"{bbox['yaw']:.3f}"
            })

        if results:
            print(pd.DataFrame(results).to_string(index=False))
        else:
            print("Aucun obstacle trouvé.")

        print("\n--- BACKGROUND ---")
        bg_clusters = extract_background_clusters(df_frame)
        print(f"{len(bg_clusters)} clusters background extraits")

    except Exception as e:
        print(f"Erreur : {e}")
        raise


if __name__ == "__main__":
    main()
