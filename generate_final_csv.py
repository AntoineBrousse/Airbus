"""
generate_final_csv.py - Génération du dataset d'entraînement complet
Usage:
    python generate_final_csv.py
"""

import glob
import os

import pandas as pd

import compute_boxes
import lidar_utils


def generate_dataset_csv(input_folder, output_csv):
    files = sorted(glob.glob(os.path.join(input_folder, "*.h5")))
    all_results = []

    if not files:
        print(f"Aucun fichier trouvé dans : {input_folder}")
        return

    for file_path in files:
        file_name = os.path.basename(file_path)
        print(f"\n>>> Début du traitement : {file_name}")

        df_raw = lidar_utils.load_h5_data(file_path)
        poses = lidar_utils.get_unique_poses(df_raw)

        if poses is None:
            continue

        num_poses = len(poses)

        for i, (_, pose) in enumerate(poses.iterrows()):
            if i % 5 == 0:
                print(f"  [{file_name}] Progression : Frame {i}/{num_poses}")

            df_frame = lidar_utils.filter_by_pose(df_raw, pose)

            # ── Nouveau : filtrer les points invalides ──
            df_frame = df_frame[df_frame["distance_cm"] > 0].copy()

            if df_frame.empty:
                continue

            xyz = lidar_utils.spherical_to_local_cartesian(df_frame)
            df_frame[['x', 'y', 'z']] = xyz

            # Filtre RAM : garder seulement les points colorés pour les obstacles
            # Les points noirs (0,0,0) sont du sol non labelisé → inutiles pour DBSCAN
            df_obstacles = df_frame[
                (df_frame['r'] > 0) | (df_frame['g'] > 0) | (df_frame['b'] > 0)
            ].copy()

            # ── utiliser cluster_obstacles() avec eps par classe ──
            obstacle_clusters = compute_boxes.cluster_obstacles(df_obstacles)

            for cluster in obstacle_clusters:
                bbox = cluster["bbox"]

                all_results.append({
                    "file_name":     file_name,
                    "frame_index":   i,          # ← Nouveau
                    "ego_x":         pose["ego_x"],
                    "ego_y":         pose["ego_y"],
                    "ego_z":         pose["ego_z"],
                    "ego_yaw":       pose["ego_yaw"],
                    "bbox_center_x": bbox["cx"],
                    "bbox_center_y": bbox["cy"],
                    "bbox_center_z": bbox["cz"],
                    "bbox_width":    bbox["w"],
                    "bbox_length":   bbox["l"],
                    "bbox_height":   bbox["h"],
                    "bbox_yaw":      bbox["yaw"],
                    "class_ID":      cluster["class_id"],
                    "class_label":   cluster["class_label"],
                })

            # ── Nouveau : extraire les clusters background ──
            bg_clusters = compute_boxes.extract_background_clusters(df_frame)

            for cluster in bg_clusters:
                bbox = cluster["bbox"]

                all_results.append({
                    "file_name":     file_name,
                    "frame_index":   i,
                    "ego_x":         pose["ego_x"],
                    "ego_y":         pose["ego_y"],
                    "ego_z":         pose["ego_z"],
                    "ego_yaw":       pose["ego_yaw"],
                    "bbox_center_x": bbox["cx"],
                    "bbox_center_y": bbox["cy"],
                    "bbox_center_z": bbox["cz"],
                    "bbox_width":    bbox["w"],
                    "bbox_length":   bbox["l"],
                    "bbox_height":   bbox["h"],
                    "bbox_yaw":      bbox["yaw"],
                    "class_ID":      cluster["class_id"],
                    "class_label":   cluster["class_label"],
                })

        # Sauvegarde intermédiaire après chaque fichier H5
        pd.DataFrame(all_results).to_csv(output_csv, index=False)
        print(f"--- Sauvegarde intermédiaire : {output_csv} ({len(all_results)} lignes) ---")

# Stats finales
    df_final = pd.DataFrame(all_results)
    print(f"\n{'='*50}")
    print(f"Traitement terminé → {output_csv}")
    print(f"Total : {len(df_final)} exemples")
    print("\nDistribution des classes :")
    counts = df_final["class_label"].value_counts()
    for label, count in counts.items():
        pct = 100 * count / len(df_final)
        print(f"  {label:<20} {count:>6} ({pct:5.1f}%)")

    # Nettoyage automatique → labels_train_clean.csv
    clean_path = output_csv.replace(".csv", "_clean.csv")
    df_clean = df_final[df_final["bbox_height"] > 0.1].reset_index(drop=True)
    df_clean.to_csv(clean_path, index=False)
    print(f"\n✅ Version nettoyée → {clean_path} ({len(df_clean)} lignes, {len(df_final)-len(df_clean)} supprimées)")


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUT_DIR  = os.path.join(SCRIPT_DIR, "airbus_hackathon_trainingdata")
    OUTPUT_FILE = os.path.join(SCRIPT_DIR, "labels_train.csv")

    if not os.path.exists(INPUT_DIR):
        print(f"ERREUR : Dossier {INPUT_DIR} introuvable.")
    else:
        generate_dataset_csv(INPUT_DIR, OUTPUT_FILE)
