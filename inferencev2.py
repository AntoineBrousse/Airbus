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

MAX_POINTS_HDBSCAN = 8000
VOXEL_SIZE         = 0.5   # mètres — 1 point par cube de 0.5m³
Z_GROUND_THRESHOLD = -5.0   # Filtre sol standard
Z_DEEP_THRESHOLD   = -80.0  # Pour pylônes/antennes qui descendent sous le Lidar
Z_MAX_THRESHOLD    = 200.0

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
# PIPELINE PAR FRAME
# ─────────────────────────────────────────────

def process_frame(frame_df, model, feature_cols):
    results = []

    frame_df = frame_df[frame_df["distance_cm"] > 0].copy()
    if len(frame_df) == 0:
        return results

    xyz = lidar_utils.spherical_to_local_cartesian(frame_df)
    dist_xy = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2)

    mask = (
        (xyz[:, 2] > Z_DEEP_THRESHOLD) &
        (xyz[:, 2] < Z_MAX_THRESHOLD) &
        (dist_xy > 2.0)
    )
    xyz_f = xyz[mask]

    if len(xyz_f) < 10:
        return results

    # Voxel grid : 1 point par voxel de VOXEL_SIZE mètres
    voxel_indices = np.floor(xyz_f / VOXEL_SIZE).astype(np.int32)
    _, unique_idx = np.unique(voxel_indices, axis=0, return_index=True)
    xyz_f = xyz_f[unique_idx]

    # Sécurité : si encore trop de points, sous-échantillonnage aléatoire
    if len(xyz_f) > MAX_POINTS_HDBSCAN:
        idx   = np.random.choice(len(xyz_f), MAX_POINTS_HDBSCAN, replace=False)
        xyz_f = xyz_f[idx]

    try:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=5, min_samples=3,
            cluster_selection_epsilon=10.0, core_dist_n_jobs=-1,
        )
        labels = clusterer.fit_predict(xyz_f)
    except Exception:
        return results

    all_pts_clusters = [xyz_f[labels == cid] for cid in set(labels) - {-1}]

    if not all_pts_clusters:
        return results

    for pts in all_pts_clusters:
        if len(pts) < 5:
            continue

        # Rogner les outliers Z (percentiles 1-99) pour éviter les bboxes trop hautes
        if len(pts) >= 20:
            z_lo = np.percentile(pts[:, 2], 1)
            z_hi = np.percentile(pts[:, 2], 99)
            pts_trimmed = pts[(pts[:, 2] >= z_lo) & (pts[:, 2] <= z_hi)]
            if len(pts_trimmed) >= 5:
                pts = pts_trimmed

        feats = extract_features(pts)
        cx = feats.pop("_cx")
        cy = feats.pop("_cy")
        cz = feats.pop("_cz")
        w  = feats.pop("_w")
        l  = feats.pop("_l")
        h  = feats.pop("_h")

        # Pré-filtrage géométrique
        if not check_geometric_rules(h, w, l):
            continue

        # Random Forest
        feat_vector = np.array(
            [feats.get(col, 0.0) for col in feature_cols]
        ).reshape(1, -1)

        class_id = int(model.predict(feat_vector)[0])

        if class_id == 4:
            continue

        if class_id == 0 and h < 12.0:                        continue  # Antenna trop petite
        if class_id == 0 and h > 95.0:                        continue  # Antenna trop grande
        if class_id == 0 and max(w, l) > 20.0:                continue  # Antenna trop large
        if class_id == 0 and w * l > 150.0:                   continue  # Antenna footprint trop grand
        if class_id == 0 and h / max(max(w,l),1e-6) < 2.0:   continue  # Antenna pas assez élancée
        if class_id == 1 and h > 20.0:                        continue  # Cable trop haut
        if class_id == 1 and min(w, l) > 15.0:                continue  # Cable trop épais → pas un câble
        if class_id == 1 and max(w, l) < 2.0:                 continue  # Cable trop court
        if class_id == 2 and h < 10.0:                        continue  # Pole trop petit
        if class_id == 2 and h > 68.0:                        continue  # Pole trop grand
        if class_id == 2 and max(w, l) > 20.0:                continue  # Pole trop large
        if class_id == 2 and h / max(max(w,l),1e-6) < 1.0:   continue  # Pole pas assez vertical
        if class_id == 3 and h < 15.0:                        continue  # Turbine trop petite
        if class_id == 3 and h > 170.0:                       continue  # Turbine trop grande
        if class_id == 3 and max(w, l) < 10.0:                continue  # Turbine trop fine → antenne

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

def run_inference(h5_path, model, feature_cols, output_csv, max_frames=None, start_frame=0, single_frame=None):
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

    if single_frame is not None:
        poses = poses[single_frame:single_frame + 1]
        print(f"   Frame unique : {single_frame}", flush=True)
    else:
        poses = poses[start_frame:]
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
    parser.add_argument("--max-frames",   type=int, default=None, help="Nb max de frames")
    parser.add_argument("--start-frame",  type=int, default=0,    help="Frame de départ")
    parser.add_argument("--frame",         type=int, default=None, help="Frame unique")
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

        run_inference(h5_path, model, feature_cols, output_csv, max_frames=args.max_frames, start_frame=args.start_frame, single_frame=args.frame)

    print("\n🏁 Inférence terminée !")


if __name__ == "__main__":
    main()
