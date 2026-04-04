## <span style="color:#D85A30">Equipe</span>

- **Carole ADEPOH N'GUESSAN**
- **Florence Liliane MATIP**
- **Grâce Marie-Paule LASM**


## 1. Contexte du projet

Nous sommes une agence immobilière **CFM Immobilier**, qui souhaite exploiter ses données analytiques pour automatiser l'estimation des prix de biens immobiliers.
En tant qu'équipe Data Engineering & Data Science, nous avons conçu et déployé un pipeline ML complet directement dans l'environnement Snowflake, sans déplacer les données vers un système externe.

L'objectif est de fournir aux équipes métier un outil fiable, accessible et interactif pour estimer la valeur d'un bien en temps réel, à partir de ses caractéristiques.


## 2. Dataset

Les données proviennent du bucket S3 : `s3://logbrain-datalake/datasets/house_price/`

Le dataset contient **1 090 biens immobiliers** décrits par 13 variables :

| Variable | Description |
|---|---|
| `price` | Prix de vente du bien |
| `area` | Surface totale en m² (33 à 324 m²) |
| `bedrooms` | Nombre de chambres |
| `bathrooms` | Nombre de salles de bain |
| `stories` | Nombre d'étages |
| `mainroad` | Accès à une route principale (oui/non) |
| `guestroom` | Présence d'une chambre d'amis (oui/non) |
| `basement` | Présence d'un sous-sol (oui/non) |
| `hotwaterheating` | Système de chauffage à eau chaude (oui/non) |
| `airconditioning` | Climatisation (oui/non) |
| `parking` | Nombre de places de parking |
| `prefarea` | Localisation en zone privilégiée (oui/non) |
| `furnishingstatus` | État d'ameublement (meublé / semi-meublé / non meublé) |


## 3.Analyse exploratoire — points clés
L'analyse des données révèle les éléments suivants:
- **Prix minimum** : 87 500 € — **Prix maximum** : 665 000 €
- **Prix médian** : 213 500 € — **Prix moyen** : 237 663 €
- La moyenne est supérieure à la médiane, ce qui indique la présence de biens 
  haut de gamme qui tirent les chiffres vers le haut.
- **Aucune valeur manquante** dans le dataset.
- Les variables les plus corrélées au prix sont la surface (0.55) 
  et le nombre de salles de bain (0.53).

---

## Pipeline ML — étapes réalisées

### 1. Ingestion des données
Chargement du dataset depuis S3 vers une table Snowflake `HOUSES_PRICES` 
via Snowpark, directement dans un Snowflake Notebook.

### 2. Exploration et visualisation
- Distribution du prix de vente (histogramme + boxplot)
- Matrice de corrélation entre les variables numériques
- Impact des variables catégorielles sur le prix (boxplots groupés)

### 3. Feature Engineering
- Encodage binaire des variables yes/no (`mainroad`, `guestroom`, `basement`, 
  `hotwaterheating`, `airconditioning`, `prefarea`) : yes → 1, no → 0
- Encodage ordinal de `furnishingstatus` : meublé → 2, semi-meublé → 1, 
  non meublé → 0
- Split train/test : 80% / 20% avec `random_state=42`
- Normalisation via `StandardScaler` appliquée sur la régression linéaire

### 4. Entraînement de trois modèles
Plutôt que de retenir un seul modèle par défaut, nous avons choisi d'en entraîner 
trois de natures différentes afin de comparer objectivement leurs performances 
et de sélectionner celui qui s'adapte le mieux à notre dataset, car aucun algorithme 
n'est universellement supérieur.

- **Linear Regression** — modèle de référence, simple et interprétable
- **Random Forest** — ensemble d'arbres de décision, robuste aux non-linéarités
- **XGBoost** — boosting par gradient, performant sur les données tabulaires

### 5. Évaluation des modèles
Ce problème étant une régression et non une classification, nous utilisons RMSE, 
MAE et R² à la place d'Accuracy, Precision et Recall qui ne s'appliquent 
pas à la prédiction d'une valeur continue.

- **RMSE** : erreur quadratique moyenne — mesure l'amplitude des erreurs 
  en pénalisant les grosses erreurs
- **MAE** : erreur absolue moyenne — mesure l'erreur moyenne sans pénalisation
- **R²** : coefficient de détermination — mesure la part de variance expliquée 
  par le modèle (0 à 1, plus c'est élevé mieux c'est)

### 6. Optimisation des hyperparamètres
Nous avons optimisé Random Forest et XGBoost via `RandomizedSearchCV` 
(10 itérations, 3 folds). La régression linéaire n'a pas été optimisée 
car elle dispose de très peu d'hyperparamètres ayant un impact significatif 
sur ses performances.

---

## Résultats — Tableau comparatif final

| Modèle | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | 53 985 | 40 253 | 0.6732 |
| Random Forest (base) | 32 483 | 19 500 | 0.8817 |
| XGBoost (base) | **29 314** | **18 497** | **0.9036** |
| Random Forest (opt.) | 32 483 | 19 500 | 0.8817 |
| XGBoost (opt.) | 32 314 | 22 011 | 0.8829 |

### Meilleur modèle : XGBoost Base
RMSE : 29 314 MAE : 18 497 R² : 0.9036
XGBoost dans sa version de base est le modèle retenu. Sa version optimisée 
s'est révélée légèrement moins performante, ce qui arrive lorsque la recherche 
aléatoire ne tombe pas sur de meilleures combinaisons de paramètres que celles 
par défaut. L'optimisation de Random Forest n'a quant à elle rien changé — 
les paramètres par défaut étaient déjà optimaux pour ce dataset.

---

## Importance des features

| Feature | Importance |
|---|---|
| BATHROOMS | 0.378 |
| AREA | 0.090 |
| AIRCONDITIONING | 0.077 |
| GUESTROOM | 0.076 |
| HOTWATERHEATING | 0.061 |
| BASEMENT | 0.059 |
| FURNISHINGSTATUS | 0.048 |
| PREFAREA | 0.047 |
| STORIES | 0.046 |
| PARKING | 0.045 |
| BEDROOMS | 0.040 |
| MAINROAD | 0.033 |

Le nombre de salles de bain est le critère le plus déterminant pour estimer 
le prix d'un bien dans ce dataset, avec une importance presque quatre fois 
supérieure à la deuxième variable (surface).

---

## Stockage dans le Model Registry

Le modèle XGBoost Base a été enregistré dans le Snowflake Model Registry 
sous le nom `HOUSE_PRICE_PREDICTOR` (version `ODD_TREEFROG_2`) avec ses 
métriques attachées, ce qui permet à toute l'organisation de le recharger 
et de l'utiliser sans réentraînement.

---

## Application Streamlit — CFM Immobilier

Une application Streamlit a été déployée directement dans Snowflake pour 
permettre aux équipes métier d'interagir avec le modèle sans connaissance 
technique.

L'application permet de :
- Saisir les caractéristiques d'un bien via une interface intuitive
- Obtenir une estimation du prix en temps réel
- Consulter le prix au m² estimé
- Accéder à l'historique des estimations réalisées

---

## Limites connues du modèle

Le modèle présente une limite identifiée sur la variable surface. De légères 
variations de surface peuvent produire des prédictions contre-intuitives 
(une maison légèrement plus grande estimée moins chère qu'une plus petite 
avec les mêmes critères). Cette limite s'explique par deux facteurs :

1. La taille réduite du dataset (1 090 exemples)
2. La faible importance relative de la surface (9%) dans le modèle 
   face au nombre de salles de bain (38%)

Le modèle reste fiable dans la plage d'entraînement : **33 à 324 m²**.

---

## Structure du dépôt

Projet-Data_Engineering_MLops_MBA_ESG/
│
├── README.md
├── .gitignore
├── notebook/
│   └── house_price_ml_pipeline.py
├── app/
│   └── streamlit_app.py
└── data/
    └── .gitkeep

---

## Équipe

Projet réalisé dans le cadre du Workshop Data Engineering & Machine Learning 
avec Snowflake — MBA ESG.

Livrable : `MBAESG_[PROMOTION]_[CLASSE]_EVALUATION_DATAENGINEER_MLOPS`


