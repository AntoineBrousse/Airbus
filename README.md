# 🚁 Airbus AI Hackathon 2026 — Détection d'Obstacles Lidar

Détection et classification d'obstacles 3D (antennes, câbles, pylônes, éoliennes) dans des nuages de points Lidar pour hélicoptères en vol basse altitude.

---

## 📖 Comprendre les données

### Scène vs Frame
- **Scène** = un environnement géographique complet. 10 scènes d'entraînement, chacune dans un fichier `.h5`.
- **Frame** = un scan Lidar depuis **une position précise de l'hélicoptère** (~575 000 points, 100 frames par scène). Chaque frame est identifiée par un quadruplet unique `(ego_x, ego_y, ego_z, ego_yaw)`.

### Labels couleur
| Phase | Couleur des points | Signification |
|---|---|---|
| Entraînement | RGB propre (ex: rouge, bleu...) | Indique la classe de l'obstacle |
| Évaluation | RGB = 128 partout | Labels masqués → le modèle doit deviner |

### Les 4 classes cibles
| ID | Classe | Couleur RGB |
|---|---|---|
| 0 | Antenna | (38, 23, 180) |
| 1 | Cable | (177, 132, 47) |
| 2 | Electric pole | (129, 81, 97) |
| 3 | Wind turbine | (66, 132, 9) |
| 4 | background | points non labelisés (arbres, sol...) |

---

## 🏗️ Architecture du pipeline

```
📁 PHASE 1 — GÉNÉRATION DU GROUND TRUTH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
compute_boxes.py + generate_final_csv.py
  → DBSCAN sur points colorés par classe (eps différent par classe)
  → DBSCAN sur points non colorés → clusters background
  → Calcul bbox orientée via PCA pour chaque cluster
  → Filtres qualité : min(w,l) > 0.5m, cz > -50m
  ↓
labels_train.csv + labels_train_clean.csv

📁 PHASE 2 — ENTRAÎNEMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
extract_features.py
  → 22 features géométriques par bbox
     (height, elongation, flatness, compactness, z_min_abs...)
  ↓
features_train.csv
  ↓
train_classifier.py
  → Random Forest 200 arbres, class_weight=balanced
  ↓
classifier.pkl

📁 PHASE 3 — INFÉRENCE (Jour J)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
inference.py
  → DBSCAN sur tous les points non-sol (sans filtre couleur)
  → Extraction des 22 features par cluster
  → Random Forest prédit la classe
  → Rejet automatique du background (class_ID=4)
  ↓
predictions/scene_X_predictions.csv
```

---

## 📊 Résultats actuels

### Modèle (Random Forest v2)
| Métrique | Score |
|---|---|
| Accuracy | 92.1% |
| F1-macro | 0.849 |
| F1 Cable | 0.95 |
| F1 Antenna | 0.81 |
| F1 Wind turbine | 0.87 |
| F1 Electric pole | 0.70 |
| F1 background | 0.97 |

### Dataset d'entraînement
| Classe | Exemples | % |
|---|---|---|
| background | 3 308 | 61.9% |
| Antenna | 712 | 13.3% |
| Wind turbine | 691 | 12.9% |
| Electric pole | 333 | 6.2% |
| Cable | 303 | 5.7% |
| **Total** | **5 347** | |

### Test inférence sur scene_5
| Classe | Ground truth | Prédictions | Status |
|---|---|---|---|
| Antenna | 95 | 146 | 1.5x trop |
| Cable | 11 | 7 | sous-détecté |
| Electric pole | 0 | 60 | faux positifs |
| Wind turbine | 73 | 31 | sous-détecté |
| background | 361 | 0 | ✅ rejeté |

---

## 🚀 Installation

```bash
pip install scikit-learn numpy pandas h5py
```

---

## ▶️ Utilisation

```bash
# Phase 1 — Générer le dataset (long ~30 min)
python generate_final_csv.py

# Phase 2 — Entraîner le modèle
python extract_features.py --csv labels_train_clean.csv --output features_train.csv --analyze
python train_classifier.py --features features_train.csv

# Phase 3 — Inférence sur les fichiers d'évaluation
python inference.py --folder eval_data/ --output-dir predictions/

# Test sur une scène d'entraînement
python inference.py --file airbus_hackathon_trainingdata/scene_5.h5 --output test.csv
```

---

## 🔧 Pistes d'amélioration

- **Faux positifs Electric pole** → ajuster `DBSCAN_EPS` dans `inference.py`
- **Wind turbine sous-détecté** → augmenter `MAX_BACKGROUND_CLUSTERS` dans `compute_boxes.py`
- **Câbles difficiles à détecter** → problème fondamental sans couleur (points très épars)
- **Ajouter plus de scènes background** → améliorer la qualité des exemples négatifs
- **Tuner les hyperparamètres RF** → `n_estimators`, `max_depth`, `min_samples_leaf`

---

## 📁 Structure des fichiers

```
├── lidar_utils.py          # Fourni par Airbus — chargement H5, conversion XYZ
├── compute_boxes.py        # DBSCAN + calcul bbox orientée + extraction background
├── generate_final_csv.py   # Génère labels_train.csv + labels_train_clean.csv
├── extract_features.py     # Calcule 22 features géométriques par bbox
├── train_classifier.py     # Entraîne le Random Forest
├── inference.py            # Pipeline d'inférence complet
├── visualize.py            # Visualisation Open3D
├── classifier.pkl          # Modèle entraîné (92.1% accuracy)
├── labels_train.csv        # Dataset brut (5 378 bboxes)
├── labels_train_clean.csv  # Dataset nettoyé (5 347 bboxes)
└── features_train.csv      # Features extraites prêtes pour l'entraînement
```
