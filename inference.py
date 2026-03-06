"""
inference.py - Génération des CSV de soumission pour le hackathon Airbus
Usage:
    # Un fichier
    python inference.py --file eval_sceneA_100.h5 --output predictions_sceneA_100.csv

    # Tous les fichiers d'un dossier
    python inference.py --folder eval_data/ --output-dir predictions/
"""

import argparse
import gc
import glob
import os
import pickle

import h5py
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

import lidar_utils

# ─────────────────────────────────────────────
# CONFIG DBSCAN (sans couleur → on cherche tout)
# ─────────────────────────────────────────────

# Sans labels couleur, on applique DBSCAN sur TOUS les points non-sol
# Paramètres adaptés pour capturer tous types d'obstacles
DBSCAN_EPS         = 3.0   # Distance max entre points du même cluster (mètres)
DBSCAN_MIN_SAMPLES = 5     # Minimum de points pour former un cluster
MAX_POINTS_DBSCAN  = 5000  # Sous-échantillonnage si trop de points

# Filtre sol : on ignore les points trop proches du sol
# Le Lidar est sur l'hélico, les obstacles sont au-dessus du sol
Z_GROUND_THRESHOLD = -5.0  # Ignorer points en dessous de cette altitude (mètres)
Z_MAX_THRESHOLD    = 150.0 # Ignorer points trop hauts (artefacts)

# Taille minimale d'un cluster pour être considéré comme obstacle
MIN_CLUSTER_POINTS = 10

CLASS_NAMES = {
    0: "Antenna",
    1: "Cable",
    2: "Electric pole",
    3: "Wind turbine"
}


# ─────────────────────────────────────────────
# FEATURES (même que extract_features.py)
# ─────────────────────────────────────────────

def extract_features_from_cluster(pts_xyz: np.ndarray) -> dict:
    """Calcule les features géométriques d'un cluster de points."""
    x, y, z = pts_xyz[:, 0], pts_xyz[:, 1], pts_xyz[:, 2]

    w = float(x.max() - x.min())
    l = float(y.max() - y.min())
    h = float(z.max() - z.min())
    cx = float(x.mean())
    cy = float(y.mean())
    cz = float((z.max() + z.min()) / 2)

    max_wl  = max(max(w, l), 1e-6)
    min_wl  = min(w, l)
    volume  = max(w * l * h, 1e-6)

    z_min   = cz - h / 2
    z_max   = cz + h / 2

    return {
        "height":               h,
        "width":                w,
        "length":               l,
        "z_center":             cz,
        "z_min_approx":         z_min,
        "z_max_approx":         z_max,
        "elongation":           h / max_wl,
        "flatness":             min_wl / max_wl,
        "aspect_wl":            w / max(l, 1e-6),
        "volume":               volume,
        "log_volume":           np.log1p(volume),
        "footprint":            w * l,
        "slenderness":          h / max(np.sqrt(w * l), 1e-6),
        "height_x_footprint":   h * w * l,
        "hw_ratio":             h / max(w, 1e-6),
        "dist_lidar":           float(np.sqrt(cx**2 + cy**2 + cz**2)),
        "height_vol_ratio":     h / max(volume ** (1/3), 1e-6),
        "z_min_abs":            abs(z_min),
        "compactness":          (w + l + h) / max(3 * (volume ** (1/3)), 1e-6),
        "footprint_height_ratio": (w * l) / max(h, 1e-6),
        "density":              0.0,
        "log_npts":             0.0,
        # Bbox info
        "_cx": cx, "_cy": cy, "_cz": cz,
        "_w": w, "_l": l, "_h": h,
    }


def compute_yaw(pts_xyz: np.ndarray) -> float:
    """Calcule le yaw via PCA sur le plan XY."""
    xy = pts_xyz[:, :2] - pts_xyz[:, :2].mean(axis=0)
    if len(xy) < 2:
        return 0.0
    cov = np.cov(xy.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    main_axis = eigenvectors[:, np.argmax(eigenvalues)]
    return float(np.arctan2(main_axis[1], main_axis[0]))


# ─────────────────────────────────────────────
# PIPELINE D'INFÉRENCE
# ─────────────────────────────────────────────

def filter_ground_points(xyz: np.ndarray, reflectivity: np.ndarray = None) -> np.ndarray:
    """
    Filtre les points sol pour ne garder que les obstacles potentiels.
    Sans labels couleur, on utilise le critère d'altitude.
    """
    mask = (xyz[:, 2] > Z_GROUND_THRESHOLD) & (xyz[:, 2] < Z_MAX_THRESHOLD)

    # Filtre distance minimale (points trop proches = bruit)
    dist = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2)
    mask &= (dist > 2.0)

    return mask


def process_frame_inference(frame_df: pd.DataFrame, model, feature_cols: list) -> list:
    """
    Traite une frame en inférence :
    1. Filtre les points sol
    2. DBSCAN sur tous les points restants
    3. Extrait features + prédit classe
    4. Retourne liste de bboxes prédites
    """
    results = []

    # Filtrer points invalides
    frame_df = frame_df[frame_df["distance_cm"] > 0].copy()
    if len(frame_df) == 0:
        return results

    # Convertir en XYZ cartésien
    xyz = lidar_utils.spherical_to_local_cartesian(frame_df)

    # Filtre sol
    ground_mask = filter_ground_points(xyz)
    xyz_filtered = xyz[ground_mask]

    if len(xyz_filtered) < MIN_CLUSTER_POINTS:
        return results

    # Sous-échantillonnage strict — toujours plafonner à MAX_POINTS_DBSCAN
    if len(xyz_filtered) > MAX_POINTS_DBSCAN:
        idx = np.random.choice(len(xyz_filtered), MAX_POINTS_DBSCAN, replace=False)
        xyz_filtered = xyz_filtered[idx]

    # DBSCAN sur tous les points non-sol
    try:
        db = DBSCAN(
            eps=DBSCAN_EPS,
            min_samples=DBSCAN_MIN_SAMPLES,
            n_jobs=-1
        ).fit(xyz_filtered)
    except MemoryError:
        print("    ⚠️  MemoryError DBSCAN → skip frame", flush=True)
        return results

    labels = db.labels_
    unique_labels = set(labels) - {-1}

    if not unique_labels:
        return results

    for cluster_id in unique_labels:
        cluster_mask = labels == cluster_id
        pts = xyz_filtered[cluster_mask]

        if len(pts) < MIN_CLUSTER_POINTS:
            continue

        # Extraire features
        feats = extract_features_from_cluster(pts)

        # Sauvegarder bbox info avant de supprimer du dict
        cx   = feats.pop("_cx")
        cy   = feats.pop("_cy")
        cz   = feats.pop("_cz")
        w    = feats.pop("_w")
        l    = feats.pop("_l")
        h    = feats.pop("_h")

        # Préparer vecteur de features dans le bon ordre
        feat_vector = np.array([feats.get(col, 0.0) for col in feature_cols]).reshape(1, -1)

        # Prédiction
        class_id = int(model.predict(feat_vector)[0])

        # Rejeter le background
        if class_id == 4:
            continue

        class_label = CLASS_NAMES[class_id]

        # Calculer yaw
        yaw = compute_yaw(pts)

        results.append({
            "cx": cx, "cy": cy, "cz": cz,
            "w": w, "l": l, "h": h,
            "yaw": yaw,
            "class_id": class_id,
            "class_label": class_label,
            "n_pts": len(pts),
        })

    return results


# ─────────────────────────────────────────────
# GÉNÉRATION CSV SOUMISSION
# ─────────────────────────────────────────────

def run_inference_on_file(h5_path: str, model, feature_cols: list, output_csv: str):
    """Lance l'inférence sur un fichier H5 et génère le CSV de soumission."""
    print(f"\n📂 {os.path.basename(h5_path)}", flush=True)

    # Charger poses
    with h5py.File(h5_path, "r") as f:
        dataset = f["lidar_points"]
        ego_x   = dataset["ego_x"][:]
        ego_y   = dataset["ego_y"][:]
        ego_z   = dataset["ego_z"][:]
        ego_yaw = dataset["ego_yaw"][:]

    poses = np.unique(np.column_stack([ego_x, ego_y, ego_z, ego_yaw]), axis=0)
    del ego_x, ego_y, ego_z, ego_yaw
    gc.collect()

    print(f"   {len(poses)} frames à traiter", flush=True)

    all_rows = []

    for i, pose in enumerate(poses):
        if i % 10 == 0:
            print(f"   Frame {i+1}/{len(poses)}...", flush=True)

        # Charger frame
        with h5py.File(h5_path, "r") as f:
            dataset = f["lidar_points"]
            ex = dataset["ego_x"][:]
            ey = dataset["ego_y"][:]
            ez = dataset["ego_z"][:]
            ew = dataset["ego_yaw"][:]
            mask = (ex == pose[0]) & (ey == pose[1]) & (ez == pose[2]) & (ew == pose[3])
            del ex, ey, ez, ew
            frame_points = dataset[mask]
            del mask

        frame_df = pd.DataFrame(
            {name: frame_points[name] for name in frame_points.dtype.names}
        )
        del frame_points
        gc.collect()

        # Inférence
        detections = process_frame_inference(frame_df, model, feature_cols)
        del frame_df
        gc.collect()

        # Formater les résultats au format hackathon
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

    # Sauvegarder CSV
    if all_rows:
        result_df = pd.DataFrame(all_rows)
        result_df.to_csv(output_csv, index=False)
        print(f"   ✅ {len(all_rows)} détections → {output_csv}")
        print(f"   {result_df['class_label'].value_counts().to_string()}")
    else:
        print("   ⚠️  Aucune détection")
        # Créer CSV vide avec les bonnes colonnes
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
    parser.add_argument("--file",       default=None, help="Un fichier H5 à traiter")
    parser.add_argument("--folder",     default=None, help="Dossier contenant les H5")
    parser.add_argument("--output",     default=None, help="CSV de sortie (pour --file)")
    parser.add_argument("--output-dir", default="predictions", help="Dossier de sortie (pour --folder)")
    parser.add_argument("--model",      default="classifier.pkl", help="Modèle pkl")
    args = parser.parse_args()

    # Charger le modèle
    print(f"🤖 Chargement du modèle {args.model}...")
    with open(args.model, "rb") as f:
        model_data = pickle.load(f)

    model        = model_data["model"]
    feature_cols = model_data["feature_cols"]
    print(f"   Features : {feature_cols}")

    # Déterminer les fichiers à traiter
    if args.file:
        h5_files = [args.file]
    elif args.folder:
        h5_files = sorted(glob.glob(os.path.join(args.folder, "*.h5")))
        print(f"   {len(h5_files)} fichiers H5 trouvés dans {args.folder}")
        os.makedirs(args.output_dir, exist_ok=True)
    else:
        print("❌ Spécifie --file ou --folder")
        return

    # Lancer l'inférence
    for h5_path in h5_files:
        if args.file:
            output_csv = args.output or h5_path.replace(".h5", "_predictions.csv")
        else:
            fname = os.path.splitext(os.path.basename(h5_path))[0]
            output_csv = os.path.join(args.output_dir, f"{fname}_predictions.csv")

        run_inference_on_file(h5_path, model, feature_cols, output_csv)

    print("\n🏁 Inférence terminée !")


if __name__ == "__main__":
    main()
