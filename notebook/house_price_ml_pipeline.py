# HOUSE PRICE ML PIPELINE — Snowflake Notebook

# CELLULE 1 — Imports & Connexion

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from snowflake.snowpark.context import get_active_session
from snowflake.snowpark import functions as F
from snowflake.snowpark.types import IntegerType, FloatType, StringType

session = get_active_session()

print("=" * 50)
print("Connexion Snowpark OK")
print(f"   Base     : {session.get_current_database()}")
print(f"   Schéma   : {session.get_current_schema()}")
print(f"   Warehouse: {session.get_current_warehouse()}")
print("=" * 50)

# CELLULE 2 — Chargement des données depuis Snowflake

# Chargement de la table en DataFrame Snowpark
df_snow = session.table("HOUSES_PRICES")

print(f" Nombre de lignes  : {df_snow.count()}")
print(f" Nombre de colonnes: {len(df_snow.columns)}")
print("\n Aperçu des données :")
df_snow.show(5)

# Convertir en pandas pour l'exploration visuelle
df = df_snow.to_pandas()
print("\n Conversion en Pandas OK")

# CELLULE 3 — Exploration : infos générales

print("=" * 50)
print(" INFORMATIONS GÉNÉRALES")
print("=" * 50)
print(df.info())

print("\n" + "=" * 50)
print(" STATISTIQUES DESCRIPTIVES")
print("=" * 50)
print(df.describe().round(2))

print("\n" + "=" * 50)
print(" VALEURS MANQUANTES")
print("=" * 50)
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else " Aucune valeur manquante")

print("\n" + "=" * 50)
print(" VALEURS UNIQUES (colonnes catégorielles)")
print("=" * 50)
cat_cols = ['MAINROAD','GUESTROOM','BASEMENT','HOTWATERHEATING',
            'AIRCONDITIONING','PREFAREA','FURNISHINGSTATUS']
for col in cat_cols:
    print(f"  {col}: {df[col].unique()}")

# CELLULE 4 — EDA : Distribution du prix (variable cible)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Distribution du Prix de Vente", fontsize=16, fontweight='bold')

# Histogramme
axes[0].hist(df['PRICE'], bins=40, color='#2196F3', edgecolor='white', alpha=0.85)
axes[0].set_title("Distribution du prix")
axes[0].set_xlabel("Prix")
axes[0].set_ylabel("Fréquence")
axes[0].axvline(df['PRICE'].mean(), color='red', linestyle='--', label=f"Moyenne: {df['PRICE'].mean():,.0f}")
axes[0].axvline(df['PRICE'].median(), color='orange', linestyle='--', label=f"Médiane: {df['PRICE'].median():,.0f}")
axes[0].legend()

# Boxplot
axes[1].boxplot(df['PRICE'], vert=True, patch_artist=True,
                boxprops=dict(facecolor='#2196F3', alpha=0.6))
axes[1].set_title("Boxplot du prix")
axes[1].set_ylabel("Prix")

plt.tight_layout()
plt.savefig('price_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n Prix min    : {df['PRICE'].min():,}")
print(f" Prix max    : {df['PRICE'].max():,}")
print(f" Prix moyen  : {df['PRICE'].mean():,.0f}")
print(f" Prix médian : {df['PRICE'].median():,.0f}")

# CELLULE 5 — EDA : Corrélations

# Corrélation sur les variables numériques
num_cols = ['PRICE', 'AREA', 'BEDROOMS', 'BATHROOMS', 'STORIES', 'PARKING']
corr_matrix = df[num_cols].corr()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Analyse des Corrélations", fontsize=16, fontweight='bold')

# Heatmap
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=axes[0], square=True,
            cbar_kws={'shrink': 0.8})
axes[0].set_title("Matrice de corrélation")

# Corrélation avec le prix uniquement
corr_price = corr_matrix['PRICE'].drop('PRICE').sort_values(ascending=True)
colors = ['#F44336' if x < 0 else '#4CAF50' for x in corr_price]
axes[1].barh(corr_price.index, corr_price.values, color=colors, edgecolor='white')
axes[1].set_title("Corrélation avec PRICE")
axes[1].set_xlabel("Coefficient de corrélation")
axes[1].axvline(0, color='black', linewidth=0.8)
for i, v in enumerate(corr_price.values):
    axes[1].text(v + 0.01 if v >= 0 else v - 0.06, i, f'{v:.2f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('correlations.png', dpi=150, bbox_inches='tight')
plt.show()

# CELLULE 6 — EDA : Variables catégorielles vs Prix

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle("Impact des variables catégorielles sur le Prix", fontsize=16, fontweight='bold')
axes = axes.flatten()

cat_cols = ['MAINROAD','GUESTROOM','BASEMENT','HOTWATERHEATING',
            'AIRCONDITIONING','PREFAREA','FURNISHINGSTATUS']

for i, col in enumerate(cat_cols):
    df.boxplot(column='PRICE', by=col, ax=axes[i],
               patch_artist=True)
    axes[i].set_title(f'Prix par {col}')
    axes[i].set_xlabel('')
    axes[i].tick_params(axis='x', rotation=15)

axes[-1].set_visible(False)
plt.tight_layout()
plt.savefig('categorical_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# CELLULE 7 — Feature Engineering

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

print(" Début du Feature Engineering...")

# Copie du dataframe
df_ml = df.copy()

# -- Encodage des variables binaires (yes=1, no=0) --
binary_cols = ['MAINROAD','GUESTROOM','BASEMENT','HOTWATERHEATING',
               'AIRCONDITIONING','PREFAREA']
for col in binary_cols:
    df_ml[col] = df_ml[col].map({'yes': 1, 'no': 0})
    print(f"   {col} encodé (yes→1, no→0)")

# -- Encodage ordinal de FURNISHINGSTATUS --
furnishing_map = {
    'furnished': 2,
    'semi-furnished': 1,
    'unfurnished': 0
}
df_ml['FURNISHINGSTATUS'] = df_ml['FURNISHINGSTATUS'].map(furnishing_map)
print("   FURNISHINGSTATUS encodé (furnished→2, semi→1, unfurnished→0)")

# -- Séparation X / y --
TARGET = 'PRICE'
FEATURES = [c for c in df_ml.columns if c != TARGET]

X = df_ml[FEATURES]
y = df_ml[TARGET]

print(f"\n Features ({len(FEATURES)}) : {FEATURES}")
print(f" Target : {TARGET}")

# -- Split Train / Test (80% / 20%) --
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n  Train : {X_train.shape[0]} lignes")
print(f"  Test  : {X_test.shape[0]} lignes")

# -- Normalisation --
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print("\n Normalisation appliquée (StandardScaler)")

# CELLULE 8 — Entraînement des 3 modèles

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(name, model, X_tr, X_te, y_tr, y_te):
    """Entraîne un modèle et retourne ses métriques."""
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    mae  = mean_absolute_error(y_te, y_pred)
    r2   = r2_score(y_te, y_pred)
    print(f"\n{'='*45}")
    print(f"   {name}")
    print(f"{'='*45}")
    print(f"  RMSE : {rmse:>12,.0f}")
    print(f"  MAE  : {mae:>12,.0f}")
    print(f"  R²   : {r2:>12.4f}")
    return {'name': name, 'model': model, 'rmse': rmse, 'mae': mae, 'r2': r2, 'y_pred': y_pred}

print(" Entraînement des modèles de base...\n")

results = []

# Modèle 1 : Régression linéaire
lr = LinearRegression()
results.append(evaluate_model("Linear Regression", lr,
                               X_train_scaled, X_test_scaled, y_train, y_test))

# Modèle 2 : Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
results.append(evaluate_model("Random Forest", rf,
                               X_train, X_test, y_train, y_test))

# Modèle 3 : XGBoost
xgb = XGBRegressor(n_estimators=100, random_state=42,
                   learning_rate=0.1, verbosity=0)
results.append(evaluate_model("XGBoost", xgb,
                               X_train, X_test, y_train, y_test))

print("\n Entraînement terminé !")


# CELLULE 9 — Comparaison visuelle des modèles

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Comparaison des modèles de base", fontsize=16, fontweight='bold')

names  = [r['name'] for r in results]
rmses  = [r['rmse'] for r in results]
maes   = [r['mae']  for r in results]
r2s    = [r['r2']   for r in results]
colors = ['#2196F3', '#4CAF50', '#FF9800']

# RMSE
bars = axes[0].bar(names, rmses, color=colors, edgecolor='white', width=0.5)
axes[0].set_title("RMSE (moins = mieux)")
axes[0].set_ylabel("RMSE")
for bar, val in zip(bars, rmses):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                 f'{val:,.0f}', ha='center', va='bottom', fontsize=9)

# MAE
bars = axes[1].bar(names, maes, color=colors, edgecolor='white', width=0.5)
axes[1].set_title("MAE (moins = mieux)")
axes[1].set_ylabel("MAE")
for bar, val in zip(bars, maes):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                 f'{val:,.0f}', ha='center', va='bottom', fontsize=9)

# R²
bars = axes[2].bar(names, r2s, color=colors, edgecolor='white', width=0.5)
axes[2].set_title("R² (plus = mieux, max=1)")
axes[2].set_ylabel("R²")
axes[2].set_ylim(0, 1.1)
for bar, val in zip(bars, r2s):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('models_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# Meilleur modèle de base
best_base = min(results, key=lambda x: x['rmse'])
print(f"\n Meilleur modèle de base : {best_base['name']} (RMSE={best_base['rmse']:,.0f}, R²={best_base['r2']:.4f})")

# CELLULE 10 — Prédictions vs Réalité (meilleur modèle de base)

best_result = min(results, key=lambda x: x['rmse'])
y_pred_best = best_result['y_pred']

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f"Analyse des prédictions — {best_result['name']}", fontsize=14, fontweight='bold')

# Scatter prédit vs réel
axes[0].scatter(y_test, y_pred_best, alpha=0.6, color='#2196F3', edgecolors='white', s=60)
min_val = min(y_test.min(), y_pred_best.min())
max_val = max(y_test.max(), y_pred_best.max())
axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Parfait')
axes[0].set_xlabel("Prix réel")
axes[0].set_ylabel("Prix prédit")
axes[0].set_title("Prédit vs Réel")
axes[0].legend()

# Résidus
residuals = y_test.values - y_pred_best
axes[1].hist(residuals, bins=30, color='#4CAF50', edgecolor='white', alpha=0.85)
axes[1].axvline(0, color='red', linestyle='--', linewidth=2)
axes[1].set_xlabel("Résidus (Réel - Prédit)")
axes[1].set_ylabel("Fréquence")
axes[1].set_title("Distribution des résidus")

plt.tight_layout()
plt.savefig('predictions_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# CELLULE 11 — Optimisation Hyperparamètres (GridSearch)

from sklearn.model_selection import RandomizedSearchCV

print("Optimisation par RandomizedSearch — Random Forest")
print("(peut prendre 1-2 minutes...)\n")

# Grille réduite
param_grid_rf = {
    'n_estimators'     : [50, 100],
    'max_depth'        : [None, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf' : [1, 2]
}

rf_base = RandomForestRegressor(random_state=42, n_jobs=1)  # ← n_jobs=1

grid_search_rf = RandomizedSearchCV(
    rf_base,
    param_distributions=param_grid_rf,
    n_iter=10,        # ← réduit de 20 à 10
    cv=3,             # ← réduit de 5 à 3
    scoring='neg_root_mean_squared_error',
    random_state=42,
    n_jobs=1,         # ← n_jobs=1 (pas de parallélisme)
    verbose=1
)

grid_search_rf.fit(X_train, y_train)

best_rf = grid_search_rf.best_estimator_
y_pred_rf_opt = best_rf.predict(X_test)
rmse_rf_opt = np.sqrt(mean_squared_error(y_test, y_pred_rf_opt))
mae_rf_opt  = mean_absolute_error(y_test, y_pred_rf_opt)
r2_rf_opt   = r2_score(y_test, y_pred_rf_opt)

print(f"\nMeilleurs paramètres RF :")
for k, v in grid_search_rf.best_params_.items():
    print(f"   {k}: {v}")
print(f"\nPerformances RF optimisé :")
print(f"   RMSE : {rmse_rf_opt:>12,.0f}")
print(f"   MAE  : {mae_rf_opt:>12,.0f}")
print(f"   R²   : {r2_rf_opt:>12.4f}")


# CELLULE 12 — Optimisation XGBoost

from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV

print("Optimisation XGBoost")

param_grid_xgb = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

xgb_base = XGBRegressor(random_state=42, verbosity=0)

grid_search_xgb = RandomizedSearchCV(
    xgb_base,
    param_distributions=param_grid_xgb,
    n_iter=10,
    cv=3,
    scoring='neg_root_mean_squared_error',
    random_state=42,
    n_jobs=1,
    verbose=1
)

grid_search_xgb.fit(X_train, y_train)

best_xgb = grid_search_xgb.best_estimator_
y_pred_xgb_opt = best_xgb.predict(X_test)
rmse_xgb_opt = np.sqrt(mean_squared_error(y_test, y_pred_xgb_opt))
mae_xgb_opt = mean_absolute_error(y_test, y_pred_xgb_opt)
r2_xgb_opt = r2_score(y_test, y_pred_xgb_opt)

for k, v in grid_search_xgb.best_params_.items():
    print(f"{k}: {v}")
print(f"RMSE : {rmse_xgb_opt:,.0f}")
print(f"MAE  : {mae_xgb_opt:,.0f}")
print(f"R2   : {r2_xgb_opt:.4f}")


# CELLULE 13 — Tableau comparatif final & sélection

print("\n" + "="*65)
print("   TABLEAU COMPARATIF FINAL")
print("="*65)
print(f"  {'Modèle':<30} {'RMSE':>12} {'MAE':>12} {'R²':>8}")
print("-"*65)

all_models = [
    ("Linear Regression",      results[0]['rmse'],  results[0]['mae'],  results[0]['r2']),
    ("Random Forest (base)",   results[1]['rmse'],  results[1]['mae'],  results[1]['r2']),
    ("XGBoost (base)",         results[2]['rmse'],  results[2]['mae'],  results[2]['r2']),
    ("Random Forest (opt.)",   rmse_rf_opt,         mae_rf_opt,         r2_rf_opt),
    ("XGBoost (opt.)",         rmse_xgb_opt,        mae_xgb_opt,        r2_xgb_opt),
]

for name, rmse, mae, r2 in all_models:
    print(f"  {name:<30} {rmse:>12,.0f} {mae:>12,.0f} {r2:>8.4f}")

print("="*65)

# Sélection automatique du meilleur
best_name, best_rmse, best_mae, best_r2 = min(all_models, key=lambda x: x[1])
print(f"\n MEILLEUR MODÈLE : {best_name}")
print(f"   RMSE = {best_rmse:,.0f} | MAE = {best_mae:,.0f} | R² = {best_r2:.4f}")

# On identifie l'objet modèle correspondant
if "XGBoost (opt.)" == best_name:
    BEST_MODEL = best_xgb
    BEST_MODEL_NAME = "XGBoost_Optimized"
elif "Random Forest (opt.)" == best_name:
    BEST_MODEL = best_rf
    BEST_MODEL_NAME = "RandomForest_Optimized"
elif "XGBoost (base)" == best_name:
    BEST_MODEL = xgb
    BEST_MODEL_NAME = "XGBoost_Base"
else:
    BEST_MODEL = rf
    BEST_MODEL_NAME = "RandomForest_Base"

print(f"\n Modèle sélectionné pour le registry : {BEST_MODEL_NAME}")


# CELLULE 14 — Feature Importance

if hasattr(BEST_MODEL, 'feature_importances_'):
    importances = pd.Series(BEST_MODEL.feature_importances_, index=FEATURES)
    importances = importances.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors_imp = ['#FF6B6B' if v > importances.median() else '#74B9FF' for v in importances.values]
    bars = ax.barh(importances.index, importances.values, color=colors_imp, edgecolor='white')
    ax.set_title(f"Importance des features — {BEST_MODEL_NAME}", fontsize=14, fontweight='bold')
    ax.set_xlabel("Importance")
    for bar, val in zip(bars, importances.values):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()


# CELLULE 15 — Sauvegarde dans le Model Registry

from snowflake.ml.registry import Registry
import json

print(" Sauvegarde dans le Snowflake Model Registry...")

# Initialiser le registry
reg = Registry(session=session, database_name="HOUSES_PRICES_DB", schema_name="ML_SCHEMA")

# Métriques à logguer
metrics = {
    "rmse" : float(best_rmse),
    "mae"  : float(best_mae),
    "r2"   : float(best_r2)
}

# Enregistrement du modèle
model_version = reg.log_model(
    model         = BEST_MODEL,
    model_name    = "HOUSE_PRICE_PREDICTOR",
    version_name  = "V1",
    comment       = f"Modèle {BEST_MODEL_NAME} — RMSE={best_rmse:,.0f}, R²={best_r2:.4f}",
    metrics       = metrics,
    sample_input_data = X_train[:5]   # exemple d'input pour la signature
)

print(f"\n Modèle enregistré !")
print(f"   Nom     : HOUSE_PRICE_PREDICTOR")
print(f"   Version : V1")
print(f"   RMSE    : {best_rmse:,.0f}")
print(f"   R²      : {best_r2:.4f}")


# CELLULE 16 — Vérification dans le registry

print(" Modèles dans le registry :\n")
models = reg.show_models()
print(models)

# Charger le modèle depuis le registry
loaded_mv = reg.get_model("HOUSE_PRICE_PREDICTOR").version("V1")
print(f"\n Modèle rechargé depuis le registry")
print(f"   Métriques : {loaded_mv.show_metrics()}")


# CELLULE 17 — Inférence sur nouvelles données

print(" Inférence sur nouvelles données...\n")

# Simuler 10 nouvelles maisons
new_houses = pd.DataFrame({
    'AREA'            : [3500, 7200, 2800, 5000, 6100, 4200, 8000, 3000, 4500, 6500],
    'BEDROOMS'        : [3, 4, 2, 3, 4, 3, 5, 2, 3, 4],
    'BATHROOMS'       : [1, 2, 1, 2, 2, 1, 3, 1, 2, 2],
    'STORIES'         : [1, 2, 1, 2, 2, 1, 3, 1, 2, 2],
    'MAINROAD'        : [1, 1, 0, 1, 1, 0, 1, 1, 0, 1],
    'GUESTROOM'       : [0, 1, 0, 0, 1, 0, 1, 0, 0, 1],
    'BASEMENT'        : [0, 1, 0, 1, 1, 0, 1, 0, 1, 1],
    'HOTWATERHEATING' : [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
    'AIRCONDITIONING' : [0, 1, 0, 1, 1, 0, 1, 0, 1, 1],
    'PARKING'         : [0, 2, 0, 1, 2, 1, 3, 0, 1, 2],
    'PREFAREA'        : [0, 1, 0, 0, 1, 0, 1, 0, 1, 1],
    'FURNISHINGSTATUS': [0, 2, 1, 1, 2, 0, 2, 1, 1, 2]
})

# Prédiction
predictions = BEST_MODEL.predict(new_houses)
new_houses['PREDICTED_PRICE'] = predictions.astype(int)

print(" Prédictions pour 10 nouvelles maisons :")
print(new_houses[['AREA', 'BEDROOMS', 'BATHROOMS', 'AIRCONDITIONING',
                   'PREFAREA', 'PREDICTED_PRICE']].to_string(index=False))

# Sauvegarder dans Snowflake
new_houses_snow = session.create_dataframe(new_houses)
new_houses_snow.write.save_as_table(
    "HOUSES_PRICES_DB.ML_SCHEMA.INFERENCE_RESULTS",
    mode="overwrite"
)

print(f"\n {len(new_houses)} prédictions sauvegardées dans INFERENCE_RESULTS")

# CELLULE 18 — Vérification table d'inférence

results_table = session.table("INFERENCE_RESULTS")
print(" Table INFERENCE_RESULTS :")
results_table.show(10)
print(f"\n Total : {results_table.count()} lignes")
print("\n Pipeline ML complet terminé avec succès !")

