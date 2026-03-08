"""
visualize_predictions.py - Visualise les bboxes prédites sur le nuage de points
Supporte deux formats de CSV :
  - inference.py     : bbox_center_x/y/z, bbox_width/length/height, class_ID
  - inference_debug.py : cx/cy/cz, w/l/h, rf_class_id, source

Usage:
    python visualize_predictions.py --file scene_1.h5 --predictions predictions.csv --frame 0
    python visualize_predictions.py --file scene_1.h5 --predictions debug.csv --frame 3 --debug
"""

import argparse

import numpy as np
import open3d as o3d
import pandas as pd

import lidar_utils

# ─────────────────────────────────────────────
# COULEURS (normalisées 0-1)
# ─────────────────────────────────────────────

# Couleurs officielles par classe
CLASS_COLORS = {
    0:  [38/255,  23/255,  180/255],  # Antenna       → bleu
    1:  [177/255, 132/255, 47/255],   # Cable          → orange
    2:  [129/255, 81/255,  97/255],   # Electric pole  → violet
    3:  [66/255,  132/255, 9/255],    # Wind turbine   → vert
    4:  [0.4,     0.4,     0.4],      # Background     → gris
    99: [1.0,     0.0,     0.0],      # Unknown        → rouge vif
}

# Couleurs par source (mode debug)
SOURCE_COLORS = {
    "vertical": [1.0,  0.9,  0.0],   # jaune  → détection verticale
    "hdbscan":  [0.0,  0.9,  0.9],   # cyan   → HDBSCAN
}

CLASS_NAMES = {
    0:  "Antenna",
    1:  "Cable",
    2:  "Electric pole",
    3:  "Wind turbine",
    4:  "Background",
    99: "Unknown",
}


# ─────────────────────────────────────────────
# CRÉATION D'UNE BBOX OPEN3D
# ─────────────────────────────────────────────

def make_bbox_lineset(cx, cy, cz, w, l, h, yaw, color):
    hl, hw, hh = w / 2, l / 2, h / 2
    corners_local = np.array([
        [-hw, -hl, -hh], [ hw, -hl, -hh],
        [ hw,  hl, -hh], [-hw,  hl, -hh],
        [-hw, -hl,  hh], [ hw, -hl,  hh],
        [ hw,  hl,  hh], [-hw,  hl,  hh],
    ])
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    R = np.array([
        [cos_y, -sin_y, 0],
        [sin_y,  cos_y, 0],
        [0,      0,     1],
    ])
    corners_world = corners_local @ R.T + np.array([cx, cy, cz])
    lines = [
        [0,1],[1,2],[2,3],[3,0],
        [4,5],[5,6],[6,7],[7,4],
        [0,4],[1,5],[2,6],[3,7],
    ]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners_world)
    ls.lines  = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return ls


# ─────────────────────────────────────────────
# DÉTECTION DU FORMAT CSV
# ─────────────────────────────────────────────

def detect_csv_format(pred_df):
    """Retourne 'debug' ou 'inference' selon les colonnes présentes."""
    if "cx" in pred_df.columns and "rf_class_id" in pred_df.columns:
        return "debug"
    return "inference"


# ─────────────────────────────────────────────
# BBOXES GROUND TRUTH
# ─────────────────────────────────────────────

def get_gt_bboxes(df_frame):
    import compute_boxes
    geoms = []
    for cluster in compute_boxes.cluster_obstacles(df_frame):
        bbox = cluster["bbox"]
        ls = make_bbox_lineset(
            bbox["cx"], bbox["cy"], bbox["cz"],
            bbox["w"],  bbox["l"],  bbox["h"],
            bbox["yaw"], [1.0, 1.0, 1.0]  # blanc = GT
        )
        geoms.append(ls)
    return geoms


# ─────────────────────────────────────────────
# BBOXES PRÉDITES — format inference.py
# ─────────────────────────────────────────────

def get_pred_bboxes_inference(pred_df, pose):
    geoms = []
    mask = (
        (np.abs(pred_df["ego_x"]   - pose["ego_x"])   < 0.01) &
        (np.abs(pred_df["ego_y"]   - pose["ego_y"])   < 0.01) &
        (np.abs(pred_df["ego_z"]   - pose["ego_z"])   < 0.01) &
        (np.abs(pred_df["ego_yaw"] - pose["ego_yaw"]) < 0.01)
    )
    frame_preds = pred_df[mask]
    for _, row in frame_preds.iterrows():
        class_id = int(row["class_ID"])
        color = CLASS_COLORS.get(class_id, [1.0, 0.0, 0.0])
        ls = make_bbox_lineset(
            row["bbox_center_x"], row["bbox_center_y"], row["bbox_center_z"],
            row["bbox_width"],    row["bbox_length"],   row["bbox_height"],
            row["bbox_yaw"],      color
        )
        geoms.append(ls)
    print(f"   {len(frame_preds)} prédictions pour cette frame")
    if len(frame_preds) > 0:
        print(f"   {frame_preds['class_label'].value_counts().to_string()}")
    return geoms


# ─────────────────────────────────────────────
# BBOXES PRÉDITES — format inference_debug.py
# ─────────────────────────────────────────────

def get_pred_bboxes_debug(pred_df, frame_idx, color_by="class"):
    """
    color_by = 'class'  → couleur par classe RF
    color_by = 'source' → couleur par source (vertical=jaune, hdbscan=cyan)
    """
    geoms = []
    frame_preds = pred_df[pred_df["frame"] == frame_idx]
    for _, row in frame_preds.iterrows():
        if color_by == "source":
            color = SOURCE_COLORS.get(str(row.get("source", "")), [1.0, 0.0, 1.0])
        else:
            class_id = int(row.get("rf_class_id", 99))
            color = CLASS_COLORS.get(class_id, [1.0, 0.0, 0.0])

        ls = make_bbox_lineset(
            float(row["cx"]), float(row["cy"]), float(row["cz"]),
            float(row["w"]),  float(row["l"]),  float(row["h"]),
            0.0, color  # yaw=0 en debug
        )
        geoms.append(ls)

    print(f"   {len(frame_preds)} clusters debug pour frame {frame_idx}")
    if len(frame_preds) > 0:
        print(f"   Par source  : {frame_preds['source'].value_counts().to_string()}")
        print(f"   Par classe  : {frame_preds['rf_class_label'].value_counts().to_string()}")
    return geoms


# ─────────────────────────────────────────────
# LÉGENDE
# ─────────────────────────────────────────────

def print_legend(debug=False, color_by="class"):
    print("\n=== LÉGENDE ===")
    print("Points        → couleurs RGB originales")
    print("Bbox BLANCHE  → Ground Truth")
    if debug and color_by == "source":
        print("Bbox JAUNE    → Détection verticale")
        print("Bbox CYAN     → HDBSCAN")
    else:
        print("Bbox BLEUE    → Antenna       (class 0)")
        print("Bbox ORANGE   → Cable         (class 1)")
        print("Bbox VIOLETTE → Electric pole (class 2)")
        print("Bbox VERTE    → Wind turbine  (class 3)")
        print("Bbox GRISE    → Background    (class 4)")
        print("Bbox ROUGE    → Unknown       (class 99)")
    print()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file",        required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--frame",       type=int, default=0)
    parser.add_argument("--no-gt",       action="store_true")
    parser.add_argument("--debug",       action="store_true", help="Forcer mode debug CSV")
    parser.add_argument("--color-by",    choices=["class", "source"], default="class",
                        help="Mode debug : colorier par classe RF ou par source")
    args = parser.parse_args()

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

    print(f"📂 Chargement de {args.predictions}...")
    pred_df = pd.read_csv(args.predictions)
    csv_format = detect_csv_format(pred_df)
    if args.debug:
        csv_format = "debug"
    print(f"   Format CSV détecté : {csv_format}")

    # Nuage de points
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

    # GT
    if not args.no_gt:
        print("\n🟦 Calcul des bboxes Ground Truth...")
        gt_geoms = get_gt_bboxes(df_frame)
        print(f"   {len(gt_geoms)} bboxes GT")
        geometries.extend(gt_geoms)

    # Prédictions
    print("\n🔮 Chargement des prédictions...")
    if csv_format == "debug":
        pred_geoms = get_pred_bboxes_debug(pred_df, args.frame, color_by=args.color_by)
    else:
        pred_geoms = get_pred_bboxes_inference(pred_df, selected_pose)
    geometries.extend(pred_geoms)

    print_legend(debug=(csv_format == "debug"), color_by=args.color_by)

    print("🖥️  Ouverture Open3D... (Q pour fermer)\n")
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=f"{'DEBUG ' if csv_format=='debug' else ''}Prédictions vs GT — Frame {args.frame}",
        width=1280, height=720
    )
    for g in geometries:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    if opt:
        opt.point_size = 2.0
        opt.background_color = np.array([0.05, 0.05, 0.05])
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()

