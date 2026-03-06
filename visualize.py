import argparse

import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN

import lidar_utils

# Configuration des classes officielle
CLASS_SETTINGS = {
    (38, 23, 180):  {"label": "Antenna",       "color": [0, 0, 1],   "eps": 3.0, "min": 5},
    (177, 132, 47): {"label": "Cable",         "color": [1, 0.5, 0], "eps": 1.2, "min": 2},
    (129, 81, 97):  {"label": "Electric pole", "color": [1, 0, 1],   "eps": 4.0, "min": 10},
    (66, 132, 9):   {"label": "Wind turbine",  "color": [0, 1, 0],   "eps": 10.0, "min": 20}
}


def check_drone_passage(pts_cable, min_dist_drone=3.0):
    """Sépare les fils et calcule les distances avec sécurité sur le nombre de points."""
    geoms = []
    if len(pts_cable) < 2: return geoms


    model = DBSCAN(eps=1.2, min_samples=2).fit(pts_cable)
    labels = model.labels_
    unique_labels = [l for l in set(labels) if l != -1]

    clusters_info = []

    for label in unique_labels:
        pts_obj = pts_cable[labels == label]
        pcd_tmp = o3d.geometry.PointCloud()
        pcd_tmp.points = o3d.utility.Vector3dVector(pts_obj)

        # SÉCURITÉ : get_oriented_bounding_box() nécessite min 4 points non-coplanaires
        if len(pts_obj) >= 4:
            try:
                obb = pcd_tmp.get_oriented_bounding_box()
                obb.color = [1, 0.5, 0]
                geoms.append(obb)
                clusters_info.append({'center': obb.get_center()})
            except RuntimeError: # Cas où les 4+ points sont parfaitement alignés
                aabb = pcd_tmp.get_axis_aligned_bounding_box()
                aabb.color = [1, 0.5, 0]
                geoms.append(aabb)
                clusters_info.append({'center': aabb.get_center()})
        else:
            # Pour 2 ou 3 points, on fait une boîte simple alignée sur les axes
            aabb = pcd_tmp.get_axis_aligned_bounding_box()
            aabb.color = [1, 0.5, 0]
            geoms.append(aabb)
            clusters_info.append({'center': aabb.get_center()})


    # Calcul des distances
    print("\n--- ANALYSE DE PASSAGE POUR DRONE ---")
    for i in range(len(clusters_info)):
        for j in range(i + 1, len(clusters_info)):
            p1 = clusters_info[i]['center']
            p2 = clusters_info[j]['center']
            dist = np.linalg.norm(p1 - p2)

            status = "✅ PASSABLE" if dist >= min_dist_drone else "❌ TROP ÉTROIT"
            print(f"Distance Fil {i} <-> Fil {j}: {dist:.2f}m | {status}")

            points = [p1, p2]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points)
            line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
            line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0] if dist >= min_dist_drone else [1, 0, 0]])
            geoms.append(line_set)

    return geoms


def main():
    parser = argparse.ArgumentParser(description="LiDAR Cable Analysis & Drone Safety")
    parser.add_argument("--file", required=True)
    parser.add_argument("--pose-index", type=int, default=0)
    parser.add_argument("--min-gap", type=float, default=3.0)
    args = parser.parse_args()


    # Chargement et filtrage
    df = lidar_utils.load_h5_data(args.file)
    pose_counts = lidar_utils.get_unique_poses(df)
    selected_pose = pose_counts.iloc[args.pose_index]
    df = lidar_utils.filter_by_pose(df, selected_pose)

    # Conversion XYZ
    xyz = lidar_utils.spherical_to_local_cartesian(df)
    df[['x', 'y', 'z']] = xyz


    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)


    if {"r", "g", "b"}.issubset(df.columns):
        rgb = np.column_stack((df["r"]/255.0, df["g"]/255.0, df["b"]/255.0))
        pcd.colors = o3d.utility.Vector3dVector(rgb)


    geometries = [pcd]

    if {"r", "g", "b"}.issubset(df.columns):
        for rgb_val, settings in CLASS_SETTINGS.items():
            mask = (df['r'] == rgb_val[0]) & (df['g'] == rgb_val[1]) & (df['b'] == rgb_val[2])
            pts_class = df[mask][['x', 'y', 'z']].to_numpy()


            if len(pts_class) > 0:
                if settings["label"] == "Cable":
                    geometries.extend(check_drone_passage(pts_class, min_dist_drone=args.min_gap))
                else:
                    model = DBSCAN(eps=settings["eps"], min_samples=settings["min"]).fit(pts_class)
                    labels = model.labels_
                    for label in set(labels):
                        if label == -1: continue
                        pts_obj = pts_class[labels == label]
                        pcd_tmp = o3d.geometry.PointCloud()
                        pcd_tmp.points = o3d.utility.Vector3dVector(pts_obj)
                        # Même sécurité ici pour les autres classes
                        if len(pts_obj) >= 4:
                            try:
                                bbox = pcd_tmp.get_oriented_bounding_box()
                            except:
                                bbox = pcd_tmp.get_axis_aligned_bounding_box()
                        else:
                            bbox = pcd_tmp.get_axis_aligned_bounding_box()
                        bbox.color = settings["color"]
                        geometries.append(bbox)


    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Airbus Hackathon - Fixed", width=1280, height=720)
    for g in geometries: vis.add_geometry(g)
    vis.get_render_option().point_size = 2.0
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
