"""
inference.py - Génération des CSV de soumission pour le hackathon Airbus

Pipeline :
  1. HDBSCAN global sur les points non-sol → clusters candidats
  2. Règles géométriques → filtrer les clusters non-obstacles
  3. Random Forest → classifier chaque candidat
  4. Post-filtrage métier → rejeter les détections impossibles

Usage:
    python inference.py --file eval_sceneA_100.h5 --output predictions.csv
    python inference.py --folder eval_data/ --output-dir predictions/
"""

import argparse
import gc
import glob
import os
import pickle

import hdbscan
import numpy as np
import pandas as pd
import h5py

import lidar_utils

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

MAX_POINTS_HDBSCAN = 50000
VOXEL_SIZE         = 0.5    # mètres — 1 point par cube de 0.5m³
Z_GROUND_THRESHOLD = -5.0   # Filtre sol standard
Z_DEEP_THRESHOLD   = -80.0  # Pour pylônes/antennes qui descendent sous le Lidar
Z_MAX_THRESHOLD    = 200.0

# Stats training par classe (labels_train_clean.csv) — pour classification par distance
TRAIN_STATS = {
    # class_id: {feature: (mean, std)}
    0: {"height": (33.3, 16.2), "footprint": (47.1, 42.1),  "elongation": (4.98, 2.04)},  # Antenna
    1: {"height": (3.5,  2.8),  "footprint": (180., 300.),  "elongation": (0.15, 0.20)},  # Cable
    2: {"height": (42.4, 14.0), "footprint": (60.,  60.),   "elongation": (3.5,  1.5)},   # Electric Pole
    3: {"height": (66.8, 37.0), "footprint": (351., 391.),  "elongation": (2.49, 2.48)},  # Wind Turbine
}

CLASS_NAMES = {
    0: "Antenna",
    1: "Cable",
    2: "Electric Pole",
    3: "Wind Turbine",
}

# ─────────────────────────────────────────────
# RÈGLES GÉOMÉTRIQUES
# ─────────────────────────────────────────────

def check_geometric_rules(h, w, l):
    """
    Retourne True si le cluster ressemble à au moins un type d'obstacle.
    Basé sur les stats réelles du dataset.
    """
    max_wl     = max(max(w, l), 1e-6)
    footprint  = w * l
    elongation = h / max_wl
    flatness   = min(w, l) / max_wl

    # Antenna : vertical compact, h=5-91m, width<16m
    if 5.0 <= h <= 95.0 and max_wl <= 20.0 and elongation >= 1.5:
        return True

    # Electric pole : vertical, h=3-65m, width<15m
    if 3.0 <= h <= 68.0 and max_wl <= 20.0 and elongation >= 1.0:
        return True

    # Wind turbine : grand, h=15-165m
    if h >= 15.0 and footprint >= 5.0:
        return True

    # Cable : horizontal, long et fin
    if h <= 20.0 and max(w, l) >= 2.0 and flatness <= 0.3:
        return True

    return False


# ─────────────────────────────────────────────
# FEATURES (identiques à extract_features.py)
# ─────────────────────────────────────────────

def extract_features(pts_xyz):
    x, y, z = pts_xyz[:, 0], pts_xyz[:, 1], pts_xyz[:, 2]

    w  = float(x.max() - x.min())
    l  = float(y.max() - y.min())
    h  = float(z.max() - z.min())
    cx = float(x.mean())
    cy = float(y.mean())
    cz = float((z.max() + z.min()) / 2)

    max_wl = max(max(w, l), 1e-6)
    min_wl = min(w, l)
    volume = max(w * l * h, 1e-6)
    z_min  = cz - h / 2
    z_max  = cz + h / 2

    return {
        "height":                 h,
        "width":                  w,
        "length":                 l,
        "z_center":               cz,
        "z_min_approx":           z_min,
        "z_max_approx":           z_max,
        "elongation":             h / max_wl,
        "flatness":               min_wl / max_wl,
        "aspect_wl":              w / max(l, 1e-6),
        "volume":                 volume,
        "log_volume":             np.log1p(volume),
        "footprint":              w * l,
        "slenderness":            h / max(np.sqrt(w * l), 1e-6),
        "height_x_footprint":     h * w * l,
        "hw_ratio":               h / max(w, 1e-6),
        "dist_lidar":             float(np.sqrt(cx**2 + cy**2 + cz**2)),
        "height_vol_ratio":       h / max(volume ** (1/3), 1e-6),
        "z_min_abs":              abs(z_min),
        "compactness":            (w + l + h) / max(3 * (volume ** (1/3)), 1e-6),
        "footprint_height_ratio": (w * l) / max(h, 1e-6),
        "density":                len(pts_xyz) / volume,
        "log_npts":               np.log1p(len(pts_xyz)),
        "_cx": cx, "_cy": cy, "_cz": cz,
        "_w": w, "_l": l, "_h": h,
    }


def compute_yaw(pts_xyz):
    xy = pts_xyz[:, :2] - pts_xyz[:, :2].mean(axis=0)
    if len(xy) < 2:
        return 0.0
    cov = np.cov(xy.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    main_axis = eigenvectors[:, np.argmax(eigenvalues)]
    return float(np.arctan2(main_axis[1], main_axis[0]))


# ─────────────────────────────────────────────
# DÉTECTION VERTICALE
# ─────────────────────────────────────────────

def detect_vertical_objects(xyz, cell_size=2.0, z_range_min=15.0, min_pts=8):
    """
    Détecte les objets verticaux (antennes, pylônes) par projection XY.
    Retourne une liste de nuages de points, un par objet détecté.
    """
    from collections import defaultdict

    dist_xy = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2)
    mask = (xyz[:, 2] > -80) & (xyz[:, 2] < 200) & (dist_xy > 2.0)
    pts = xyz[mask]

    if len(pts) < 10:
        return []

    # Grille XY
    cells = defaultdict(list)
    for i, (x, y, z) in enumerate(pts):
        cells[(int(x // cell_size), int(y // cell_size))].append(i)

    # Cellules candidates : grande plage Z
    candidate_cells = set()
    for (xi, yi), idxs in cells.items():
        zvals = pts[idxs, 2]
        xvals = pts[idxs, 0]
        yvals = pts[idxs, 1]
        xy_size = max(xvals.max() - xvals.min(), yvals.max() - yvals.min())
        if zvals.max() - zvals.min() >= z_range_min and len(idxs) >= min_pts and xy_size <= cell_size:
            candidate_cells.add((xi, yi))

    if not candidate_cells:
        return []

    # Fusionner les cellules adjacentes (connected components)
    visited = set()
    objects = []

    for start in candidate_cells:
        if start in visited:
            continue
        # BFS
        queue = [start]
        group = []
        while queue:
            cell = queue.pop()
            if cell in visited:
                continue
            visited.add(cell)
            if cell not in candidate_cells:
                continue
            group.append(cell)
            xi, yi = cell
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nb = (xi+dx, yi+dy)
                    if nb not in visited:
                        queue.append(nb)

        # Collecter tous les points du groupe
        all_idxs = []
        for cell in group:
            all_idxs.extend(cells[cell])
        if len(all_idxs) < min_pts:
            continue
        cluster_pts = pts[all_idxs]
        # Filtrer : un vrai objet vertical est fin en XY
        cluster_w = cluster_pts[:, 0].max() - cluster_pts[:, 0].min()
        cluster_l = cluster_pts[:, 1].max() - cluster_pts[:, 1].min()
        if max(cluster_w, cluster_l) > 30.0:  # trop large → végétation/terrain
            continue
        objects.append(cluster_pts)

    return objects

# ─────────────────────────────────────────────
# PIPELINE PAR FRAME
# ─────────────────────────────────────────────

def process_frame(frame_df, model, feature_cols):
    results = []

    frame_df = frame_df[frame_df["distance_cm"] > 0].copy()
    if len(frame_df) == 0:
        return results

    xyz = lidar_utils.spherical_to_local_cartesian(frame_df)
    dist_xy = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2)

    # ── ÉTAPE 1 : Détection verticale (antennes, pylônes) ──
    vertical_clusters = detect_vertical_objects(xyz, cell_size=2.0, z_range_min=25.0, min_pts=10)

    # ── ÉTAPE 2 : HDBSCAN pour les objets non-verticaux (câbles, éoliennes) ──
    mask_hdb = (xyz[:, 2] > Z_GROUND_THRESHOLD) & (xyz[:, 2] < Z_MAX_THRESHOLD) & (dist_xy > 2.0)
    xyz_hdb = xyz[mask_hdb]
    hdb_clusters = []
    if len(xyz_hdb) >= 10:
        vi = np.floor(xyz_hdb / VOXEL_SIZE).astype(np.int32)
        _, ui = np.unique(vi, axis=0, return_index=True)
        xyz_v = xyz_hdb[ui]
        if len(xyz_v) > MAX_POINTS_HDBSCAN:
            idx = np.random.choice(len(xyz_v), MAX_POINTS_HDBSCAN, replace=False)
            xyz_v = xyz_v[idx]
        try:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=5, min_samples=3,
                cluster_selection_epsilon=10.0, core_dist_n_jobs=-1,
            )
            labels = clusterer.fit_predict(xyz_v)
            hdb_clusters = [xyz_v[labels == cid] for cid in set(labels) - {-1}]
        except Exception:
            pass

    # Supprimer les clusters HDBSCAN qui chevauchent un cluster vertical
    vert_centers = [(p[:, 0].mean(), p[:, 1].mean()) for p in vertical_clusters]
    filtered_hdb = []
    for pts in hdb_clusters:
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
        overlap = any(abs(cx-vx) < 20 and abs(cy-vy) < 20 for vx, vy in vert_centers)
        if not overlap:
            filtered_hdb.append(pts)

    # Combiner tous les clusters : verticaux (is_vertical=True) + HDBSCAN
    all_clusters = [(pts, True)  for pts in vertical_clusters] + \
                   [(pts, False) for pts in filtered_hdb]

    for pts, is_vertical in all_clusters:
        if len(pts) < 5:
            continue

        # Rogner outliers Z
        if len(pts) >= 15:
            z_lo = np.percentile(pts[:, 2], 2)
            z_hi = np.percentile(pts[:, 2], 98)
            pts_t = pts[(pts[:, 2] >= z_lo) & (pts[:, 2] <= z_hi)]
            if len(pts_t) >= 5:
                pts = pts_t

        feats = extract_features(pts)
        cx = feats.pop("_cx")
        cy = feats.pop("_cy")
        cz = feats.pop("_cz")
        w  = feats.pop("_w")
        l  = feats.pop("_l")
        h  = feats.pop("_h")

        if not check_geometric_rules(h, w, l):
            continue

        feat_vector = np.array(
            [feats.get(col, 0.0) for col in feature_cols]
        ).reshape(1, -1)
        class_id = int(model.predict(feat_vector)[0])

        # Les objets verticaux ne peuvent pas être des câbles
        if is_vertical and class_id == 1:
            proba = model.predict_proba(feat_vector)[0]
            # Prendre la meilleure classe hors cable
            proba[1] = 0
            class_id = int(np.argmax(proba))

        if class_id == 4:
            continue

        # Post-filtrage
        if class_id == 0:  # Antenna
            if h < 12.0 or h > 95.0:           continue
            if max(w, l) > 20.0:                continue
            if w * l > 200.0:                   continue
            if h / max(max(w, l), 1e-6) < 2.0: continue
        elif class_id == 1:  # Cable
            if h > 25.0:                        continue
            if min(w, l) > 20.0:                continue
            if max(w, l) < 2.0:                 continue
        elif class_id == 2:  # Electric Pole
            if h < 8.0 or h > 70.0:            continue
            if max(w, l) > 25.0:               continue
            if h / max(max(w, l), 1e-6) < 0.8: continue
        elif class_id == 3:  # Wind Turbine
            if h < 15.0 or h > 175.0:          continue
            if max(w, l) < 8.0:                continue

        results.append({
            "cx": cx, "cy": cy, "cz": cz,
            "w": w, "l": l, "h": h,
            "yaw":         compute_yaw(pts),
            "class_id":    class_id,
            "class_label": CLASS_NAMES[class_id],
            "n_pts":       len(pts),
        })

    return results


# ─────────────────────────────────────────────
# FUSION DES DOUBLONS
# ─────────────────────────────────────────────

def merge_duplicates(df, distance_threshold=15.0):
    if df.empty:
        return df

    merged_rows = []
    for class_id in df["class_ID"].unique():
        df_class = df[df["class_ID"] == class_id].copy()
        if len(df_class) == 1:
            merged_rows.append(df_class)
            continue

        centers = df_class[["bbox_center_x", "bbox_center_y", "bbox_center_z"]].values
        used    = np.zeros(len(df_class), dtype=bool)

        for i in range(len(df_class)):
            if used[i]:
                continue
            dists      = np.sqrt(((centers - centers[i]) ** 2).sum(axis=1))
            close_mask = dists < distance_threshold
            group      = df_class[close_mask]
            best       = group.loc[group["bbox_height"].idxmax()]
            merged_rows.append(best.to_frame().T)
            used[close_mask] = True

    return pd.concat(merged_rows, ignore_index=True) if merged_rows else df


# ─────────────────────────────────────────────
# INFÉRENCE SUR UN FICHIER
# ─────────────────────────────────────────────

def run_inference(h5_path, model, feature_cols, output_csv, max_frames=None):
    print(f"\n📂 {os.path.basename(h5_path)}", flush=True)

    with h5py.File(h5_path, "r") as f:
        dataset = f["lidar_points"]
        ego_x   = dataset["ego_x"][:]
        ego_y   = dataset["ego_y"][:]
        ego_z   = dataset["ego_z"][:]
        ego_yaw = dataset["ego_yaw"][:]

    poses = np.unique(np.column_stack([ego_x, ego_y, ego_z, ego_yaw]), axis=0)
    del ego_x, ego_y, ego_z, ego_yaw
    gc.collect()

    if max_frames:
        poses = poses[:max_frames]
    print(f"   {len(poses)} frames à traiter", flush=True)

    all_rows   = []
    scene_name = os.path.splitext(os.path.basename(h5_path))[0]

    for i, pose in enumerate(poses):
        if i % 10 == 0:
            print(f"   Frame {i+1}/{len(poses)}...", flush=True)

        with h5py.File(h5_path, "r") as f:
            dataset = f["lidar_points"]
            ex  = dataset["ego_x"][:]
            ey  = dataset["ego_y"][:]
            ez  = dataset["ego_z"][:]
            ew  = dataset["ego_yaw"][:]
            msk = (ex == pose[0]) & (ey == pose[1]) & (ez == pose[2]) & (ew == pose[3])
            del ex, ey, ez, ew
            frame_pts = dataset[msk]
            del msk

        frame_df = pd.DataFrame(
            {name: frame_pts[name] for name in frame_pts.dtype.names}
        )
        del frame_pts
        gc.collect()

        detections = process_frame(frame_df, model, feature_cols)
        del frame_df
        gc.collect()

        for det in detections:
            all_rows.append({
                "ego_x":         pose[0],
                "ego_y":         pose[1],
                "ego_z":         pose[2],
                "ego_yaw":       pose[3],
                "bbox_center_x": det["cx"],
                "bbox_center_y": det["cy"],
                "bbox_center_z": det["cz"],
                "bbox_width":    det["w"],
                "bbox_length":   det["l"],
                "bbox_height":   det["h"],
                "bbox_yaw":      det["yaw"],
                "class_ID":      det["class_id"],
                "class_label":   det["class_label"],
            })

    if all_rows:
        result_df = pd.DataFrame(all_rows)
        before    = len(result_df)
        result_df = merge_duplicates(result_df)
        print(f"   🔀 Fusion doublons : {before} → {len(result_df)} détections")
        result_df.to_csv(output_csv, index=False)
        print(f"   ✅ {len(result_df)} détections → {output_csv}")
        print(f"   {result_df['class_label'].value_counts().to_string()}")
    else:
        print("   ⚠️  Aucune détection")
        pd.DataFrame(columns=[
            "ego_x", "ego_y", "ego_z", "ego_yaw",
            "bbox_center_x", "bbox_center_y", "bbox_center_z",
            "bbox_width", "bbox_length", "bbox_height",
            "bbox_yaw", "class_ID", "class_label"
        ]).to_csv(output_csv, index=False)

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file",       default=None)
    parser.add_argument("--folder",     default=None)
    parser.add_argument("--output",     default=None)
    parser.add_argument("--output-dir", default="predictions")
    parser.add_argument("--model",      default="classifier.pkl")
    parser.add_argument("--max-frames", type=int, default=None, help="Limiter le nb de frames (debug)")
    args = parser.parse_args()

    print(f"🤖 Chargement du modèle {args.model}...")
    with open(args.model, "rb") as f:
        model_data = pickle.load(f)

    model        = model_data["model"]
    feature_cols = model_data["feature_cols"]
    print(f"   Features : {feature_cols}")

    if args.file:
        h5_files = [args.file]
    elif args.folder:
        h5_files = sorted(glob.glob(os.path.join(args.folder, "*.h5")))
        print(f"   {len(h5_files)} fichiers H5 trouvés")
        os.makedirs(args.output_dir, exist_ok=True)
    else:
        print("❌ Spécifie --file ou --folder")
        return

    for h5_path in h5_files:
        if args.file:
            output_csv = args.output or h5_path.replace(".h5", "_predictions.csv")
        else:
            fname      = os.path.splitext(os.path.basename(h5_path))[0]
            output_csv = os.path.join(args.output_dir, f"{fname}_predictions.csv")

        run_inference(h5_path, model, feature_cols, output_csv, max_frames=args.max_frames)

    print("\n🏁 Inférence terminée !")


if __name__ == "__main__":
    main()
