"""
visualize_predictions.py - Visualise les bboxes prédites sur le nuage de points
Affiche :
  - Les points avec leurs couleurs RGB originales
  - Les bboxes ground truth (couleurs officielles, traits pleins)
  - Les bboxes prédites (mêmes couleurs, traits pointillés/transparents)

Usage:
    python visualize_predictions.py --file scene_1.h5 --predictions test_scene1.csv --frame 0
"""

import argparse

import numpy as np
import open3d as o3d
import pandas as pd

import lidar_utils

# ─────────────────────────────────────────────
# COULEURS OFFICIELLES (normalisées 0-1)
# ─────────────────────────────────────────────

CLASS_COLORS = {
    0: [38/255,  23/255,  180/255],  # Antenna      → bleu
    1: [177/255, 132/255, 47/255],   # Cable         → orange
    2: [129/255, 81/255,  97/255],   # Electric pole → violet
    3: [66/255,  132/255, 9/255],    # Wind turbine  → vert
}

CLASS_RGB = {
    0: (38, 23, 180),
    1: (177, 132, 47),
    2: (129, 81, 97),
    3: (66, 132, 9),
}

CLASS_NAMES = {
    0: "Antenna",
    1: "Cable",
    2: "Electric pole",
    3: "Wind turbine",
}


# ─────────────────────────────────────────────
# CRÉATION D'UNE BBOX OPEN3D
# ─────────────────────────────────────────────

def make_bbox_lineset(cx, cy, cz, w, l, h, yaw, color):
    """
    Crée une bounding box 3D orientée sous forme de LineSet Open3D.
    On utilise un LineSet plutôt qu'une OrientedBoundingBox pour
    pouvoir contrôler finement l'apparence (couleur, épaisseur).
    """
    # Les 8 coins de la bbox dans le repère local
    hl, hw, hh = w / 2, l / 2, h / 2
    corners_local = np.array([
        [-hw, -hl, -hh], [ hw, -hl, -hh],
        [ hw,  hl, -hh], [-hw,  hl, -hh],
        [-hw, -hl,  hh], [ hw, -hl,  hh],
        [ hw,  hl,  hh], [-hw,  hl,  hh],
    ])

    # Rotation autour de Z (yaw)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    R = np.array([
        [cos_y, -sin_y, 0],
        [sin_y,  cos_y, 0],
        [0,      0,     1],
    ])
    corners_world = corners_local @ R.T + np.array([cx, cy, cz])

    # Les 12 arêtes de la bbox
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Face bas
        [4, 5], [5, 6], [6, 7], [7, 4],  # Face haut
        [0, 4], [1, 5], [2, 6], [3, 7],  # Piliers
    ]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners_world)
    line_set.lines  = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))

    return line_set


# ─────────────────────────────────────────────
# BBOXES GROUND TRUTH (depuis les points colorés)
# ─────────────────────────────────────────────

def get_gt_bboxes(df_frame):
    """Reconstruit les bboxes ground truth depuis les couleurs RGB."""
    import compute_boxes

    geoms = []

    obstacle_clusters = compute_boxes.cluster_obstacles(df_frame)
    for cluster in obstacle_clusters:
        bbox  = cluster["bbox"]
        # GT = VERT vif → "ce qu'il y a vraiment"
        color = [0.0, 1.0, 0.0]
        ls = make_bbox_lineset(
            bbox["cx"], bbox["cy"], bbox["cz"],
            bbox["w"],  bbox["l"],  bbox["h"],
            bbox["yaw"], color
        )
        geoms.append(ls)

    return geoms


# ─────────────────────────────────────────────
# BBOXES PRÉDITES (depuis le CSV d'inférence)
# ─────────────────────────────────────────────

def get_pred_bboxes(pred_df, pose):
    """Filtre les prédictions pour la frame courante et crée les LineSet."""
    geoms = []

    # Filtrer les prédictions pour cette frame
    mask = (
        (np.abs(pred_df["ego_x"]   - pose["ego_x"])   < 0.01) &
        (np.abs(pred_df["ego_y"]   - pose["ego_y"])   < 0.01) &
        (np.abs(pred_df["ego_z"]   - pose["ego_z"])   < 0.01) &
        (np.abs(pred_df["ego_yaw"] - pose["ego_yaw"]) < 0.01)
    )
    frame_preds = pred_df[mask]

    for _, row in frame_preds.iterrows():
        class_id = int(row["class_ID"])
        if class_id not in CLASS_COLORS:
            continue

        # Prédiction = ROUGE vif → "ce que le modèle prédit"
        pred_color = [1.0, 1.0, 0.0]

        ls = make_bbox_lineset(
            row["bbox_center_x"], row["bbox_center_y"], row["bbox_center_z"],
            row["bbox_width"],    row["bbox_length"],   row["bbox_height"],
            row["bbox_yaw"],      pred_color
        )
        geoms.append(ls)

    print(f"   {len(frame_preds)} prédictions pour cette frame")
    if len(frame_preds) > 0:
        print(f"   {frame_preds['class_label'].value_counts().to_string()}")

    return geoms


# ─────────────────────────────────────────────
# LÉGENDE DANS LE TERMINAL
# ─────────────────────────────────────────────

def print_legend():
    print("\n=== LÉGENDE ===")
    print("Points colorés  → couleurs RGB originales du dataset")
    print("Bbox VERTE      → Ground Truth (ce qu'il y a vraiment)")
    print("Bbox ROUGE      → Prédictions du modèle")
    print()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file",        required=True,  help="Fichier H5")
    parser.add_argument("--predictions", required=True,  help="CSV de prédictions")
    parser.add_argument("--frame",       type=int, default=0, help="Index de frame (0-based)")
    parser.add_argument("--no-gt",       action="store_true", help="Masquer le ground truth")
    args = parser.parse_args()

    # ── Charger les données ──
    print(f"\n📂 Chargement de {args.file}...")
    df = lidar_utils.load_h5_data(args.file)
    pose_counts = lidar_utils.get_unique_poses(df)

    if args.frame >= len(pose_counts):
        print(f"❌ Frame {args.frame} invalide (max: {len(pose_counts)-1})")
        return

    selected_pose = pose_counts.iloc[args.frame]
    df_frame = lidar_utils.filter_by_pose(df, selected_pose)
    df_frame = df_frame[df_frame["distance_cm"] > 0].copy()

    xyz = lidar_utils.spherical_to_local_cartesian(df_frame)
    df_frame[["x", "y", "z"]] = xyz

    print(f"   Frame {args.frame} : {len(df_frame):,} points")

    # ── Charger les prédictions ──
    print(f"📂 Chargement de {args.predictions}...")
    pred_df = pd.read_csv(args.predictions)

    # ── Nuage de points avec couleurs originales ──
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    if {"r", "g", "b"}.issubset(df_frame.columns):
        rgb_norm = np.column_stack([
            df_frame["r"] / 255.0,
            df_frame["g"] / 255.0,
            df_frame["b"] / 255.0,
        ])
        pcd.colors = o3d.utility.Vector3dVector(rgb_norm)

    geometries = [pcd]

    # ── Bboxes Ground Truth ──
    if not args.no_gt:
        print("\n🟦 Calcul des bboxes Ground Truth...")
        gt_geoms = get_gt_bboxes(df_frame)
        print(f"   {len(gt_geoms)} bboxes GT")
        geometries.extend(gt_geoms)

    # ── Bboxes Prédites ──
    print("\n🔮 Chargement des prédictions...")
    pred_geoms = get_pred_bboxes(pred_df, selected_pose)
    geometries.extend(pred_geoms)

    # ── Légende ──
    print_legend()

    # ── Visualisation ──
    print("🖥️  Ouverture de la fenêtre Open3D...")
    print("   Appuie sur Q pour fermer\n")

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=f"Prédictions vs GT — Frame {args.frame}",
        width=1280, height=720
    )
    for g in geometries:
        vis.add_geometry(g)

    render_opt = vis.get_render_option()
    if render_opt is not None:
        render_opt.point_size = 2.0
        render_opt.background_color = np.array([0.05, 0.05, 0.05])

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
