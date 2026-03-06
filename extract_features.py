"""
extract_features.py - Extraction des features géométriques pour le Random Forest
Usage:
    python extract_features.py --csv labels_train_clean.csv --output features_train.csv
"""

import argparse

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# EXTRACTION DES FEATURES
# ─────────────────────────────────────────────

def extract_features_from_bbox(row: pd.Series) -> dict:
    """
    Calcule les features géométriques à partir d'une ligne du CSV.
    Ces features permettent au Random Forest de distinguer les 5 classes
    (Antenna, Cable, Electric pole, Wind turbine, background).
    """
    w = row["bbox_width"]
    l = row["bbox_length"]
    h = row["bbox_height"]
    cx = row["bbox_center_x"]
    cy = row["bbox_center_y"]
    cz = row["bbox_center_z"]
    n  = row.get("num_points", 0)

    # Sécurité division par zéro
    max_wl  = max(max(w, l), 1e-6)
    min_wl  = min(w, l)
    volume  = max(w * l * h, 1e-6)
    z_min   = cz - h / 2
    z_max   = cz + h / 2

    features = {
        # ── Dimensions brutes ──
        "height":       h,
        "width":        w,
        "length":       l,

        # ── Position verticale ──
        "z_center":     cz,
        "z_min_approx": z_min,   # Altitude du bas
        "z_max_approx": z_max,   # Altitude du haut

        # ── Ratios de forme ──
        "elongation":   h / max_wl,            # > 1 = vertical (antenne)
        "flatness":     min_wl / max_wl,        # ~0 = câble (asymétrique)
        "aspect_wl":    w / max(l, 1e-6),       # ratio w/l

        # ── Volume ──
        "volume":       volume,
        "log_volume":   np.log1p(volume),

        # ── Compacité ──
        "footprint":    w * l,                              # Surface au sol
        "slenderness":  h / max(np.sqrt(w * l), 1e-6),     # Finesse verticale

        # ── Features supplémentaires ──
        "height_x_footprint": h * w * l,        # Éolienne = haute ET large
        "hw_ratio":           h / max(w, 1e-6), # Antenne = haute ET fine

        # ── Distance au Lidar ──
        "dist_lidar":   np.sqrt(cx**2 + cy**2 + cz**2),

        # ── Features anti-background ──
        # Les arbres/rochers sont ronds et proches du sol
        # Les obstacles sont verticaux et peuvent être suspendus

        # Ratio hauteur/volume → obstacles = grands pour leur volume
        "height_vol_ratio": h / max(volume ** (1/3), 1e-6),

        # z_min élevé = objet suspendu (câble) ou sur une colline
        # z_min bas = au niveau du sol (arbre, rocher, pylône)
        "z_min_abs":    abs(z_min),

        # Compacité 3D : sphère = 1, objet allongé > 1
        # Arbre ≈ sphère, antenne = très allongée
        "compactness":  (w + l + h) / max(3 * (volume ** (1/3)), 1e-6),

        # Ratio surface/hauteur : arbre = large pour sa hauteur
        "footprint_height_ratio": (w * l) / max(h, 1e-6),
    }

    # Ajouter num_points si disponible
    if n > 0:
        features["density"]  = n / volume
        features["log_npts"] = np.log1p(n)
    else:
        features["density"]  = 0.0
        features["log_npts"] = 0.0

    return features


# ─────────────────────────────────────────────
# ANALYSE DES FEATURES
# ─────────────────────────────────────────────

def print_feature_analysis(df_feat: pd.DataFrame, labels: pd.Series):
    """Affiche les moyennes par classe pour comprendre le pouvoir discriminant."""
    df_analysis = df_feat.copy()
    df_analysis["class"] = labels

    print("\n=== Moyennes des features par classe ===\n")
    means = df_analysis.groupby("class").mean()
    print(means[["height", "elongation", "flatness", "z_min_approx",
                 "volume", "slenderness"]].round(2).to_string())

    print("\n=== Features les plus discriminantes ===")
    # Variance inter-classe normalisée
    global_mean = df_feat.mean()
    global_std  = df_feat.std().replace(0, 1)
    class_means = df_analysis.groupby("class").mean()
    scores = ((class_means - global_mean) / global_std).abs().max()
    top_features = scores.sort_values(ascending=False).head(8)
    for feat, score in top_features.items():
        bar = "█" * int(score * 3)
        print(f"  {feat:<20} {score:.2f}  {bar}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",    required=True, help="CSV nettoyé (labels_train_clean.csv)")
    parser.add_argument("--output", default="features_train.csv", help="CSV de sortie avec features")
    parser.add_argument("--analyze", action="store_true", help="Afficher l'analyse des features")
    args = parser.parse_args()

    print(f"\n📂 Chargement de {args.csv}...")
    df = pd.read_csv(args.csv)
    print(f"   {len(df)} bboxes chargées")

    # Nettoyer les bboxes nulles si pas encore fait
    before = len(df)
    df = df[df["bbox_height"] > 0.1].reset_index(drop=True)
    if before != len(df):
        print(f"   ⚠️  {before - len(df)} bboxes nulles supprimées")

    print("\n🔧 Extraction des features...")
    feature_rows = []
    for _, row in df.iterrows():
        feats = extract_features_from_bbox(row)
        feature_rows.append(feats)

    df_features = pd.DataFrame(feature_rows)

    # Ajouter les colonnes de label et d'identification
    df_features["class_ID"]    = df["class_ID"].values
    df_features["class_label"] = df["class_label"].values

    # Ajouter ego pose pour traçabilité
    for col in ["ego_x", "ego_y", "ego_z", "ego_yaw"]:
        if col in df.columns:
            df_features[col] = df[col].values

    print(f"   ✅ {len(df_features)} features extraites")
    print(f"   Features : {[c for c in df_features.columns if c not in ['class_ID','class_label','ego_x','ego_y','ego_z','ego_yaw']]}")

    # Analyse optionnelle
    if args.analyze:
        feat_cols = [c for c in df_features.columns
                     if c not in ["class_ID", "class_label", "ego_x", "ego_y", "ego_z", "ego_yaw"]]
        print_feature_analysis(df_features[feat_cols], df_features["class_label"])

    # Distribution finale
    print("\n📊 Distribution des classes dans le dataset :")
    counts = df_features["class_label"].value_counts()
    total  = len(df_features)
    for label, count in counts.items():
        pct = 100 * count / total
        bar = "█" * int(pct / 2)
        print(f"  {label:<20} {count:>5} ({pct:5.1f}%)  {bar}")

    min_count = counts.min()
    max_count = counts.max()
    ratio = max_count / max(min_count, 1)
    if ratio > 3:
        print(f"\n  ⚠️  Déséquilibre détecté (ratio {ratio:.1f}x)")
        print("     → On utilisera class_weight='balanced' dans le Random Forest")

    # Sauvegarder
    df_features.to_csv(args.output, index=False)
    print(f"\n✅ Features sauvegardées → {args.output}")


if __name__ == "__main__":
    main()
