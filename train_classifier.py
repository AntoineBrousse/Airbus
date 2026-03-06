"""
train_classifier.py - Entraînement du classifieur Random Forest
Usage:
    python train_classifier.py --features features_train.csv
"""

import argparse
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

FEATURE_COLS = [
    "height", "width", "length",
    "z_center", "z_min_approx", "z_max_approx",
    "elongation", "flatness", "aspect_wl",
    "volume", "log_volume", "footprint",
    "slenderness", "height_x_footprint", "hw_ratio",
    "dist_lidar",
    "height_vol_ratio", "z_min_abs",
    "compactness", "footprint_height_ratio",
    "density", "log_npts"
]

LABEL_COL = "class_ID"

CLASS_NAMES = {
    0: "Antenna",
    1: "Cable",
    2: "Electric pole",
    3: "Wind turbine",
    4: "background"
}


# ─────────────────────────────────────────────
# FONCTIONS
# ─────────────────────────────────────────────

def print_confusion_matrix(y_true, y_pred, class_names):
    """Affiche une matrice de confusion lisible."""
    cm = confusion_matrix(y_true, y_pred)
    labels = sorted(class_names.keys())
    names  = [class_names[l] for l in labels]

    print("\n=== Matrice de confusion ===")
    print(f"{'':>15}", end="")
    for n in names:
        print(f"{n[:10]:>12}", end="")
    print()

    for i, label in enumerate(labels):
        print(f"{names[i]:>15}", end="")
        for j in range(len(labels)):
            val = cm[i][j]
            marker = " ✅" if i == j else "   "
            print(f"{val:>10}{marker if i==j else '  '}", end="")
        print()


def print_feature_importance(model, feature_cols):
    """Affiche l'importance des features."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\n=== Importance des features ===")
    for i, idx in enumerate(indices):
        bar = "█" * int(importances[idx] * 100)
        print(f"  {feature_cols[idx]:<20} {importances[idx]:.3f}  {bar}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="features_train.csv")
    parser.add_argument("--output",   default="classifier.pkl")
    parser.add_argument("--n-trees",  type=int, default=200)
    parser.add_argument("--test-size",type=float, default=0.2)
    args = parser.parse_args()

    # ── 1. Chargement ──────────────────────────
    print(f"\n📂 Chargement de {args.features}...")
    df = pd.read_csv(args.features)
    print(f"   {len(df)} exemples, {len(FEATURE_COLS)} features")

    # Vérifier que toutes les features sont présentes
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"❌ Features manquantes : {missing}")
        return

    X = df[FEATURE_COLS].values
    y = df[LABEL_COL].values

    print("\n📊 Distribution des classes :")
    for cid, cname in CLASS_NAMES.items():
        count = (y == cid).sum()
        pct   = 100 * count / len(y)
        print(f"   {cname:<20} {count:>5} ({pct:.1f}%)")

    # ── 2. Split train/test ────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=42,
        stratify=y  # Garder les proportions dans train et test
    )
    print(f"\n   Train : {len(X_train)} exemples")
    print(f"   Test  : {len(X_test)} exemples")

    # ── 3. Entraînement ───────────────────────
    print(f"\n🌲 Entraînement Random Forest ({args.n_trees} arbres)...")
    model = RandomForestClassifier(
        n_estimators=args.n_trees,
        max_depth=None,          # Arbres profonds
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced", # Compense le déséquilibre des classes
        random_state=42,
        n_jobs=-1,               # Utilise tous les cœurs CPU
        verbose=0
    )
    model.fit(X_train, y_train)
    print("   ✅ Entraînement terminé !")

    # ── 4. Évaluation ─────────────────────────
    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    print(f"\n🎯 Accuracy sur le test set : {accuracy*100:.1f}%")

    print("\n=== Rapport par classe ===")
    print(classification_report(
        y_test, y_pred,
        target_names=[CLASS_NAMES[i] for i in range((5))]
    ))

    print_confusion_matrix(y_test, y_pred, CLASS_NAMES)
    print_feature_importance(model, FEATURE_COLS)

    # ── 5. Cross-validation ───────────────────
    print("\n🔁 Cross-validation 5-fold...")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="f1_macro", n_jobs=-1)
    print(f"   F1-macro moyen : {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    # ── 6. Sauvegarde ─────────────────────────
    model_data = {
        "model":        model,
        "feature_cols": FEATURE_COLS,
        "class_names":  CLASS_NAMES,
    }
    with open(args.output, "wb") as f:
        pickle.dump(model_data, f)

    print(f"\n✅ Modèle sauvegardé → {args.output}")
    print(f"   Nombre de paramètres (arbres × feuilles) : "
          f"{sum(e.tree_.n_node_samples.sum() for e in model.estimators_):,}")


if __name__ == "__main__":
    main()
