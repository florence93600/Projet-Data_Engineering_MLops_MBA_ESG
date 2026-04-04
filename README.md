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


## 3. Analyse exploratoire — points clés
L'analyse du dataset révèle les éléments suivants:
- **Prix minimum** : 87 500 € — **Prix maximum** : 665 000 €
- **Prix médian** : 213 500 € — **Prix moyen** : 237 663 €
- La moyenne est supérieure à la médiane, ce qui indique la présence de biens  haut de gamme qui tirent les chiffres vers le haut.
- **Aucune valeur manquante** dans le dataset.
- Les variables les plus corrélées au prix sont la surface (0.55) et le nombre de salles de bain (0.53).


## 4. Pipeline ML — étapes réalisées

### a. Ingestion des données
Chargement du dataset depuis S3 vers une table Snowflake `HOUSES_PRICES` via Snowpark, directement dans un Snowflake Notebook.

### b. Exploration et visualisation
Nous avons produit plusieurs visualisations pour comprendre les données :
- Distribution du prix de vente (histogramme + boxplot) pour voir la répartition des prix et identifier les valeurs extrêmes
- Matrice de corrélation entre les variables numériques pour mesurer les liens entre chaque variable et le prix
- Boxplots groupés par variable catégorielle pour mesurer l'impact de chaque équipement ou caractéristique sur le prix

### c. Feature Engineering
**Encodage des variables texte (yes/no → 0/1)**
Les colonnes `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning` et `prefarea` contenaient les mots "yes" ou "no". Nous les avons transformées en 1 (yes) et 0 (no) pour que le modèle puisse les lire.

**Encodage ordinal de l'ameublement**
La colonne `furnishingstatus` avait trois modalités avec un ordre logique : meublé vaut plus qu'un semi-meublé, qui vaut plus qu'un non-meublé. Nous avons donc appliqué un encodage ordonné : furnished → 2, semi-furnished → 1, unfurnished → 0.

**Split train/test : 80% / 20%**
Le dataset a été divisé en deux parties. 80% des maisons (872 lignes) servent à entraîner le modèle, et 20% (218 lignes) sont mises de côté pour tester ses performances sur des données qu'il n'a jamais vues. Le paramètre `random_state=42` fixe la répartition aléatoire pour garantir des résultats identiques à chaque exécution du code.

**Normalisation via `StandardScaler`**
La normalisation consiste à ramener toutes les variables sur la même échelle. Sans ça, la surface (33 à 324 m²) et le nombre de chambres (1 à 6) n'auraient pas le même poids dans les calculs de la régression linéaire, ce qui fausserait le modèle. Le `StandardScaler` transforme chaque variable pour qu'elle ait une moyenne de 0 et un écart type de 1. Cette étape est appliquée uniquement à la régression linéaire — Random Forest et XGBoost sont basés sur des arbres de décision et n'en ont pas besoin.

### d. Entraînement de trois modèles
Plutôt que de retenir un seul modèle par défaut, nous avons choisi d'en entraîner trois de natures différentes afin de comparer objectivement leurs performances et de sélectionner celui qui s'adapte le mieux à notre dataset, car aucun algorithme n'est universellement supérieur.

| Modèle | Description |
|---|---|
| **Linear Regression** | Modèle de référence, simple et interprétable. Cherche une relation linéaire entre les variables et le prix. |
| **Random Forest** | Construit des centaines d'arbres de décision et fait la moyenne de leurs prédictions. |
| **XGBoost** | Construit les arbres un par un, chaque nouvel arbre apprenant des erreurs du précédent. |


### e. Évaluation des modèles
Ce problème étant une régression et non une classification, nous utilisons RMSE, MAE et R² à la place d'Accuracy, Precision et Recall qui ne s'appliquent pas à la prédiction d'une valeur continue.

- **RMSE** : erreur quadratique moyenne — mesure l'amplitude des erreurs en pénalisant les grosses erreurs
- **MAE** : erreur absolue moyenne — mesure l'erreur moyenne sans pénalisation
- **R²** : coefficient de détermination — mesure la part de variance expliquée par le modèle (0 à 1, plus c'est élevé mieux c'est)

### f. Optimisation des hyperparamètres
Pour améliorer les performances de Random Forest et XGBoost, nous avons utilisé RandomizedSearchCV, une technique qui teste automatiquement différentes combinaisons de paramètres pour trouver la meilleure configuration possible.
Concrètement, nous lui avons demandé de tester 10 combinaisons de paramètres différentes ("10 itérations"). Pour chaque combinaison testée, le modèle est évalué 3 fois sur des portions différentes des données d'entraînement ("3 folds") afin d'avoir un résultat fiable et non dépendant d'un seul découpage. Au total cela représente 30 entraînements par modèle. La meilleure combinaison est ensuite retenue automatiquement.
La régression linéaire n'a pas été optimisée car elle dispose de très peu de paramètres ajustables ayant un impact réel sur ses performances — l'optimiser n'aurait pas apporté de gain significatif.


## 5. Résultats — Tableau comparatif final

| Modèle | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | 53 985 | 40 253 | 0.6732 |
| Random Forest (base) | 32 483 | 19 500 | 0.8817 |
| XGBoost (base) | **29 314** | **18 497** | **0.9036** |
| Random Forest (opt.) | 32 483 | 19 500 | 0.8817 |
| XGBoost (opt.) | 32 314 | 22 011 | 0.8829 |

### Meilleur modèle : XGBoost Base
RMSE : 29 314 MAE : 18 497 R² : 0.9036
XGBoost dans sa version de base est le modèle retenu. Il se trompe en moyenne de 29 314 euros sur le prix d'un bien et explique 90% des variations de prix observées dans le dataset.
Sa version optimisée s'est révélée légèrement moins performante, ce qui arrive lorsque la recherche aléatoire ne tombe pas sur de meilleures combinaisons de paramètres que celles par défaut. L'optimisation de Random Forest n'a quant à elle rien changé, les paramètres par défaut étaient déjà optimaux pour ce dataset.


## 6. Importance des features
Après avoir sélectionné XGBoost comme meilleur modèle, nous avons analysé quelles variables ont le plus influencé ses décisions.

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

Le nombre de salles de bain est de loin le critère le plus déterminant, avec une importance de 0.378 — soit presque quatre fois plus que la surface. Ce résultat montre que dans ce dataset, les salles de bain sont le principal marqueur du standing d'un bien.


## 7. Stockage dans le Model Registry

Le modèle XGBoost Base a été enregistré dans le Snowflake Model Registry sous le nom `HOUSE_PRICE_PREDICTOR` (version `V1`) avec ses métriques attachées, ce qui permet à toute l'organisation de le recharger et de l'utiliser sans réentraînement.
(ou exécuter la commande suivante dans SQL: "SHOW VERSIONS IN MODEL HOUSES_PRICES_DB.ML_SCHEMA.HOUSE_PRICE_PREDICTOR;")


## 8. Application Streamlit — CFM Immobilier

Une application Streamlit a été déployée directement dans Snowflake pour permettre aux équipes métier d'interagir avec le modèle sans connaissance technique.

L'application permet de :
- Saisir les caractéristiques d'un bien via une interface intuitive
- Obtenir une estimation du prix en temps réel, calculée par le modèle rechargé depuis le registry
- Consulter le prix au m² estimé
- Accéder à l'historique des estimations réalisées


## 9. Limites connues du modèle
Le modèle présente une limite identifiée sur la variable surface. De légères variations de surface peuvent produire des prédictions contre-intuitives. Par exemple une maison légèrement plus grande estimée moins chère qu'une plus petite avec les mêmes critères par ailleurs.
Cette limite s'explique par deux facteurs :
-La taille réduite du dataset (1 090 exemples) ne permet pas au modèle de capturer finement toutes les relations entre variables
-La faible importance relative de la surface dans le modèle (9%) face au nombre de salles de bain (38%) — XGBoost a appris depuis les données que la surface n'est pas le facteur dominant, même si c'est contre-intuitif

Le modèle est fiable dans la plage d'entraînement : **33 à 324 m²**. Toute valeur en dehors de cette plage produit une estimation indicative uniquement. Cette limite vient de la nature du dataset fourni et non d'une erreur de modélisation.


## 10. Structure du dépôt

```
Projet-Data_Engineering_MLops_MBA_ESG/
│
├── README.md
├── .gitignore
├── Interface Streamlit CFM Immobilier (visualisation du rendu de l'application Streamlit)
├── notebook/
│   └── house_price_ml_pipeline.py
├── app/
│   └── streamlit_app.py
└── data/
    └── .gitkeep
```







