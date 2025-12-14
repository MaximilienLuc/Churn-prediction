"""
=============================================================================
PIPELINE DE PRÉDICTION DE CHURN - Service de Streaming Musical
=============================================================================
Ce script construit un modèle prédictif pour identifier les utilisateurs
susceptibles de résilier leur abonnement dans les 10 jours suivant le 2018-11-20.

Auteur: Claude
Date: 2024
=============================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, precision_recall_curve,
                             average_precision_score, f1_score)

# Configuration
plt.style.use('seaborn-v0_8-whitegrid')
pd.set_option('display.max_columns', None)

print("="*70)
print("CHARGEMENT ET EXPLORATION DES DONNÉES")
print("="*70)

# =============================================================================
# 1. CHARGEMENT DES DONNÉES
# =============================================================================
df = pd.read_csv('df_train_sample.csv')

print(f"\n📊 Dimensions du dataset: {df.shape[0]:,} événements, {df.shape[1]} colonnes")
print(f"📅 Période: du {df['time'].min()} au {df['time'].max()}")
print(f"👥 Nombre d'utilisateurs uniques: {df['userId'].nunique():,}")

# Aperçu des données
print("\n📋 Aperçu des colonnes:")
print(df.dtypes)

# =============================================================================
# 2. NETTOYAGE ET PRÉPARATION
# =============================================================================
print("\n" + "="*70)
print("NETTOYAGE ET PRÉPARATION DES DONNÉES")
print("="*70)

# Conversion des timestamps
df['time'] = pd.to_datetime(df['time'])
df['registration'] = pd.to_datetime(df['registration'])

# Date de référence (fin de la période d'observation)
REFERENCE_DATE = pd.to_datetime('2018-11-20')

# Vérification de la variable cible
print(f"\n🎯 Distribution de la variable cible (will_churn_10days):")
churn_dist = df.groupby('userId')['will_churn_10days'].first().value_counts()
print(churn_dist)
print(f"\nTaux de churn: {churn_dist[1] / churn_dist.sum() * 100:.2f}%")

# Analyse des pages visitées
print(f"\n📄 Types de pages visitées:")
print(df['page'].value_counts())

# =============================================================================
# 3. FEATURE ENGINEERING
# =============================================================================
print("\n" + "="*70)
print("FEATURE ENGINEERING - AGRÉGATION PAR UTILISATEUR")
print("="*70)

def create_user_features(df, reference_date):
    """
    Crée des features agrégées au niveau utilisateur.
    """
    
    # Filtrer les événements avant la date de référence
    df_filtered = df[df['time'] <= reference_date].copy()
    
    # DataFrame pour stocker les features
    user_features = pd.DataFrame()
    user_features['userId'] = df_filtered['userId'].unique()
    
    # ----- FEATURES DÉMOGRAPHIQUES -----
    user_info = df_filtered.groupby('userId').agg({
        'gender': 'first',
        'level': 'last',  # Dernier niveau connu
        'registration': 'first',
        'will_churn_10days': 'first'
    }).reset_index()
    
    user_features = user_features.merge(user_info, on='userId', how='left')
    
    # Ancienneté (jours depuis inscription)
    user_features['days_since_registration'] = (
        reference_date - user_features['registration']
    ).dt.days
    
    # ----- FEATURES D'ACTIVITÉ GLOBALE -----
    activity = df_filtered.groupby('userId').agg({
        'ts': 'count',  # Nombre total d'événements
        'sessionId': 'nunique',  # Nombre de sessions uniques
        'time': ['min', 'max'],  # Première et dernière activité
        'length': ['sum', 'mean', 'count'],  # Durée d'écoute
    })
    activity.columns = ['_'.join(col).strip() for col in activity.columns]
    activity = activity.reset_index()
    activity.columns = ['userId', 'total_events', 'unique_sessions', 
                        'first_activity', 'last_activity',
                        'total_listening_time', 'avg_song_length', 'songs_with_length']
    
    user_features = user_features.merge(activity, on='userId', how='left')
    
    # Récence (jours depuis la dernière activité)
    user_features['days_since_last_activity'] = (
        reference_date - pd.to_datetime(user_features['last_activity'])
    ).dt.days
    
    # Durée d'activité (jours entre première et dernière activité)
    user_features['activity_span_days'] = (
        pd.to_datetime(user_features['last_activity']) - 
        pd.to_datetime(user_features['first_activity'])
    ).dt.days + 1
    
    # Fréquence d'utilisation
    user_features['events_per_day'] = (
        user_features['total_events'] / user_features['activity_span_days']
    ).replace([np.inf, -np.inf], 0)
    
    user_features['sessions_per_day'] = (
        user_features['unique_sessions'] / user_features['activity_span_days']
    ).replace([np.inf, -np.inf], 0)
    
    # ----- FEATURES PAR TYPE DE PAGE -----
    # Pages importantes pour le churn
    important_pages = [
        'NextSong', 'Home', 'Thumbs Up', 'Thumbs Down', 
        'Add to Playlist', 'Add Friend', 'Roll Advert',
        'Downgrade', 'Cancel', 'Submit Downgrade', 'Error',
        'Help', 'Settings', 'Logout'
    ]
    
    page_counts = df_filtered.groupby(['userId', 'page']).size().unstack(fill_value=0)
    
    for page in important_pages:
        col_name = f'page_{page.lower().replace(" ", "_")}'
        if page in page_counts.columns:
            page_counts_temp = page_counts[[page]].reset_index()
            page_counts_temp.columns = ['userId', col_name]
            user_features = user_features.merge(page_counts_temp, on='userId', how='left')
            user_features[col_name] = user_features[col_name].fillna(0)
        else:
            user_features[col_name] = 0
    
    # ----- FEATURES DE SIGNAUX DE CHURN -----
    # Visites sur pages de résiliation/downgrade
    user_features['churn_signals'] = (
        user_features.get('page_downgrade', 0) + 
        user_features.get('page_cancel', 0) + 
        user_features.get('page_submit_downgrade', 0)
    )
    
    # Ratio Thumbs Down / Thumbs Up
    user_features['thumbs_ratio'] = (
        user_features.get('page_thumbs_down', 0) / 
        (user_features.get('page_thumbs_up', 0) + 1)
    )
    
    # Taux d'erreurs
    user_features['error_rate'] = (
        user_features.get('page_error', 0) / user_features['total_events']
    ).fillna(0)
    
    # ----- FEATURES D'ENGAGEMENT -----
    # Ratio de chansons écoutées sur total d'événements
    user_features['song_event_ratio'] = (
        user_features.get('page_nextsong', 0) / user_features['total_events']
    ).fillna(0)
    
    # Interactions positives (Thumbs Up + Add to Playlist + Add Friend)
    user_features['positive_interactions'] = (
        user_features.get('page_thumbs_up', 0) + 
        user_features.get('page_add_to_playlist', 0) + 
        user_features.get('page_add_friend', 0)
    )
    
    # Taux d'interactions positives
    user_features['positive_interaction_rate'] = (
        user_features['positive_interactions'] / user_features['total_events']
    ).fillna(0)
    
    # ----- FEATURES TEMPORELLES -----
    # Dernière semaine vs reste de la période
    one_week_before = reference_date - pd.Timedelta(days=7)
    
    last_week = df_filtered[df_filtered['time'] >= one_week_before].groupby('userId').size()
    last_week = last_week.reset_index()
    last_week.columns = ['userId', 'events_last_week']
    
    user_features = user_features.merge(last_week, on='userId', how='left')
    user_features['events_last_week'] = user_features['events_last_week'].fillna(0)
    
    # Tendance d'activité (activité récente vs ancienne)
    user_features['activity_trend'] = (
        user_features['events_last_week'] / (user_features['total_events'] + 1)
    )
    
    # ----- FEATURES PAR NIVEAU (paid/free) -----
    # Historique des changements de niveau
    level_changes = df_filtered.groupby('userId')['level'].nunique().reset_index()
    level_changes.columns = ['userId', 'level_changes']
    user_features = user_features.merge(level_changes, on='userId', how='left')
    user_features['has_changed_level'] = (user_features['level_changes'] > 1).astype(int)
    
    # ----- FEATURES DE LOCALISATION (simplifiées) -----
    # On peut extraire l'état/région de la location si nécessaire
    
    return user_features


# Création des features
print("\n⚙️ Création des features utilisateur...")
user_df = create_user_features(df, REFERENCE_DATE)

print(f"✅ Dataset agrégé: {user_df.shape[0]} utilisateurs, {user_df.shape[1]} features")

# Affichage des features créées
print(f"\n📊 Features créées:")
feature_cols = [col for col in user_df.columns if col not in ['userId', 'will_churn_10days', 'registration', 'first_activity', 'last_activity']]
for i, col in enumerate(feature_cols, 1):
    print(f"  {i:2d}. {col}")

# =============================================================================
# 4. ANALYSE EXPLORATOIRE DES FEATURES
# =============================================================================
print("\n" + "="*70)
print("ANALYSE EXPLORATOIRE")
print("="*70)

# Création des visualisations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Distribution du churn
ax1 = axes[0, 0]
churn_counts = user_df['will_churn_10days'].value_counts()
colors = ['#2ecc71', '#e74c3c']
ax1.pie(churn_counts, labels=['Non-Churn', 'Churn'], autopct='%1.1f%%', colors=colors, startangle=90)
ax1.set_title('Distribution du Churn', fontsize=12, fontweight='bold')

# 2. Churn par niveau (paid/free)
ax2 = axes[0, 1]
churn_by_level = user_df.groupby(['level', 'will_churn_10days']).size().unstack(fill_value=0)
churn_by_level_pct = churn_by_level.div(churn_by_level.sum(axis=1), axis=0) * 100
churn_by_level_pct.plot(kind='bar', ax=ax2, color=colors)
ax2.set_title('Taux de Churn par Niveau', fontsize=12, fontweight='bold')
ax2.set_xlabel('Niveau')
ax2.set_ylabel('Pourcentage')
ax2.legend(['Non-Churn', 'Churn'])
ax2.tick_params(axis='x', rotation=0)

# 3. Churn par genre
ax3 = axes[0, 2]
churn_by_gender = user_df.groupby(['gender', 'will_churn_10days']).size().unstack(fill_value=0)
churn_by_gender_pct = churn_by_gender.div(churn_by_gender.sum(axis=1), axis=0) * 100
churn_by_gender_pct.plot(kind='bar', ax=ax3, color=colors)
ax3.set_title('Taux de Churn par Genre', fontsize=12, fontweight='bold')
ax3.set_xlabel('Genre')
ax3.set_ylabel('Pourcentage')
ax3.legend(['Non-Churn', 'Churn'])
ax3.tick_params(axis='x', rotation=0)

# 4. Distribution des événements par statut de churn
ax4 = axes[1, 0]
user_df.boxplot(column='total_events', by='will_churn_10days', ax=ax4)
ax4.set_title('Événements Totaux par Statut de Churn', fontsize=12, fontweight='bold')
ax4.set_xlabel('Churn (0=Non, 1=Oui)')
ax4.set_ylabel('Nombre d\'événements')
plt.suptitle('')

# 5. Récence par statut de churn
ax5 = axes[1, 1]
user_df.boxplot(column='days_since_last_activity', by='will_churn_10days', ax=ax5)
ax5.set_title('Récence par Statut de Churn', fontsize=12, fontweight='bold')
ax5.set_xlabel('Churn (0=Non, 1=Oui)')
ax5.set_ylabel('Jours depuis dernière activité')
plt.suptitle('')

# 6. Signaux de churn
ax6 = axes[1, 2]
user_df.boxplot(column='churn_signals', by='will_churn_10days', ax=ax6)
ax6.set_title('Signaux de Churn par Statut', fontsize=12, fontweight='bold')
ax6.set_xlabel('Churn (0=Non, 1=Oui)')
ax6.set_ylabel('Nombre de signaux')
plt.suptitle('')

plt.tight_layout()
plt.savefig('eda_churn_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

print("✅ Graphiques sauvegardés dans 'eda_churn_analysis.png'")

# Statistiques comparatives
print("\n📈 Statistiques comparatives Churn vs Non-Churn:")
comparison_cols = ['total_events', 'unique_sessions', 'days_since_last_activity', 
                   'events_per_day', 'page_nextsong', 'positive_interactions',
                   'churn_signals', 'thumbs_ratio', 'activity_trend']

comparison = user_df.groupby('will_churn_10days')[comparison_cols].mean().T
comparison.columns = ['Non-Churn', 'Churn']
comparison['Diff %'] = ((comparison['Churn'] - comparison['Non-Churn']) / comparison['Non-Churn'] * 100).round(1)
print(comparison.round(2))

# =============================================================================
# 5. PRÉPARATION POUR LA MODÉLISATION
# =============================================================================
print("\n" + "="*70)
print("PRÉPARATION POUR LA MODÉLISATION")
print("="*70)

# Sélection des features
exclude_cols = ['userId', 'will_churn_10days', 'registration', 
                'first_activity', 'last_activity', 'gender', 'level']

feature_cols = [col for col in user_df.columns if col not in exclude_cols]

# Encodage des variables catégorielles
user_df['gender_encoded'] = LabelEncoder().fit_transform(user_df['gender'].fillna('Unknown'))
user_df['level_encoded'] = LabelEncoder().fit_transform(user_df['level'].fillna('Unknown'))

feature_cols.extend(['gender_encoded', 'level_encoded'])

# Préparation X et y
X = user_df[feature_cols].fillna(0)
y = user_df['will_churn_10days']

print(f"\n📊 Dimensions finales:")
print(f"  - Features (X): {X.shape}")
print(f"  - Target (y): {y.shape}")
print(f"  - Taux de churn: {y.mean()*100:.2f}%")

# Division train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📂 Division des données:")
print(f"  - Train: {X_train.shape[0]} utilisateurs ({y_train.mean()*100:.2f}% churn)")
print(f"  - Test: {X_test.shape[0]} utilisateurs ({y_test.mean()*100:.2f}% churn)")

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =============================================================================
# 6. MODÉLISATION
# =============================================================================
print("\n" + "="*70)
print("MODÉLISATION")
print("="*70)

# Dictionnaire des modèles
models = {
    'Logistic Regression': LogisticRegression(
        class_weight='balanced', 
        max_iter=1000, 
        random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
}

# Résultats
results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n🔄 Entraînement et évaluation des modèles...\n")

for name, model in models.items():
    print(f"{'='*50}")
    print(f"📌 {name}")
    print(f"{'='*50}")
    
    # Utiliser les données scalées pour la régression logistique
    if name == 'Logistic Regression':
        X_tr, X_te = X_train_scaled, X_test_scaled
    else:
        X_tr, X_te = X_train, X_test
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_tr, y_train, cv=cv, scoring='roc_auc')
    print(f"  CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    # Entraînement final
    model.fit(X_tr, y_train)
    
    # Prédictions
    y_pred = model.predict(X_te)
    y_pred_proba = model.predict_proba(X_te)[:, 1]
    
    # Métriques
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    ap = average_precision_score(y_test, y_pred_proba)
    
    print(f"  Test ROC-AUC: {roc_auc:.4f}")
    print(f"  Test F1-Score: {f1:.4f}")
    print(f"  Test Average Precision: {ap:.4f}")
    
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-Churn', 'Churn']))
    
    results[name] = {
        'model': model,
        'cv_auc': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'test_auc': roc_auc,
        'test_f1': f1,
        'test_ap': ap,
        'y_pred_proba': y_pred_proba
    }

# =============================================================================
# 7. COMPARAISON DES MODÈLES
# =============================================================================
print("\n" + "="*70)
print("COMPARAISON DES MODÈLES")
print("="*70)

# Tableau récapitulatif
results_df = pd.DataFrame({
    'Model': list(results.keys()),
    'CV ROC-AUC': [r['cv_auc'] for r in results.values()],
    'CV Std': [r['cv_std'] for r in results.values()],
    'Test ROC-AUC': [r['test_auc'] for r in results.values()],
    'Test F1': [r['test_f1'] for r in results.values()],
    'Test AP': [r['test_ap'] for r in results.values()]
})
results_df = results_df.sort_values('Test ROC-AUC', ascending=False)
print("\n📊 Récapitulatif des performances:")
print(results_df.to_string(index=False))

# Meilleur modèle
best_model_name = results_df.iloc[0]['Model']
best_model = results[best_model_name]['model']
print(f"\n🏆 Meilleur modèle: {best_model_name}")

# Visualisation des courbes ROC
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC Curves
ax1 = axes[0]
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_pred_proba'])
    ax1.plot(fpr, tpr, label=f"{name} (AUC={res['test_auc']:.3f})", linewidth=2)

ax1.plot([0, 1], [0, 1], 'k--', linewidth=1)
ax1.set_xlabel('False Positive Rate', fontsize=11)
ax1.set_ylabel('True Positive Rate', fontsize=11)
ax1.set_title('Courbes ROC', fontsize=13, fontweight='bold')
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)

# Precision-Recall Curves
ax2 = axes[1]
for name, res in results.items():
    precision, recall, _ = precision_recall_curve(y_test, res['y_pred_proba'])
    ax2.plot(recall, precision, label=f"{name} (AP={res['test_ap']:.3f})", linewidth=2)

ax2.set_xlabel('Recall', fontsize=11)
ax2.set_ylabel('Precision', fontsize=11)
ax2.set_title('Courbes Precision-Recall', fontsize=13, fontweight='bold')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print("✅ Graphiques sauvegardés dans 'model_comparison.png'")

# =============================================================================
# 8. ANALYSE DES FEATURES IMPORTANTES
# =============================================================================
print("\n" + "="*70)
print("IMPORTANCE DES FEATURES")
print("="*70)

# Feature importance du Random Forest
rf_model = results['Random Forest']['model']
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🔍 Top 15 features les plus importantes (Random Forest):")
print(feature_importance.head(15).to_string(index=False))

# Visualisation
fig, ax = plt.subplots(figsize=(10, 8))
top_features = feature_importance.head(15)
colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top_features)))
bars = ax.barh(range(len(top_features)), top_features['importance'], color=colors)
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['feature'])
ax.invert_yaxis()
ax.set_xlabel('Importance', fontsize=11)
ax.set_title('Top 15 Features les Plus Importantes', fontsize=13, fontweight='bold')
ax.grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

print("✅ Graphique sauvegardé dans 'feature_importance.png'")

# =============================================================================
# 9. MATRICE DE CONFUSION DU MEILLEUR MODÈLE
# =============================================================================
print("\n" + "="*70)
print("MATRICE DE CONFUSION - MEILLEUR MODÈLE")
print("="*70)

# Prédictions avec le meilleur modèle
if best_model_name == 'Logistic Regression':
    y_pred_best = best_model.predict(X_test_scaled)
else:
    y_pred_best = best_model.predict(X_test)

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred_best)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Non-Churn', 'Churn'],
            yticklabels=['Non-Churn', 'Churn'])
ax.set_xlabel('Prédit', fontsize=11)
ax.set_ylabel('Réel', fontsize=11)
ax.set_title(f'Matrice de Confusion - {best_model_name}', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\n📊 Matrice de confusion ({best_model_name}):")
print(f"  - Vrais Négatifs (TN): {cm[0,0]}")
print(f"  - Faux Positifs (FP): {cm[0,1]}")
print(f"  - Faux Négatifs (FN): {cm[1,0]}")
print(f"  - Vrais Positifs (TP): {cm[1,1]}")

print("\n✅ Graphique sauvegardé dans 'confusion_matrix.png'")

# =============================================================================
# 10. RÉSUMÉ ET RECOMMANDATIONS
# =============================================================================
print("\n" + "="*70)
print("RÉSUMÉ ET RECOMMANDATIONS")
print("="*70)

print(f"""
📋 RÉSUMÉ DU PROJET
{'='*50}
• Dataset: {df.shape[0]:,} événements de {user_df.shape[0]:,} utilisateurs
• Période d'observation: jusqu'au 2018-11-20
• Fenêtre de prédiction: 10 jours après le 2018-11-20
• Taux de churn: {y.mean()*100:.2f}%

🏆 MEILLEUR MODÈLE: {best_model_name}
• ROC-AUC: {results[best_model_name]['test_auc']:.4f}
• F1-Score: {results[best_model_name]['test_f1']:.4f}

🔑 FEATURES LES PLUS PRÉDICTIVES:
{chr(10).join([f"  {i+1}. {row['feature']} ({row['importance']:.4f})" for i, row in feature_importance.head(5).iterrows()])}

💡 RECOMMANDATIONS:
1. Les signaux de churn (pages Cancel, Downgrade) sont très prédictifs
2. La récence d'activité est un indicateur clé
3. L'engagement (interactions positives) protège contre le churn
4. Les utilisateurs "paid" peuvent avoir un comportement différent

📁 FICHIERS GÉNÉRÉS:
• eda_churn_analysis.png - Analyse exploratoire
• model_comparison.png - Comparaison des modèles
• feature_importance.png - Importance des features
• confusion_matrix.png - Matrice de confusion
""")

# Sauvegarde du modèle et du scaler
import pickle

with open('best_model.pkl', 'wb') as f:
    pickle.dump({
        'model': best_model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'model_name': best_model_name
    }, f)

print("✅ Modèle sauvegardé dans 'best_model.pkl'")

# =============================================================================
# 11. FONCTION DE PRÉDICTION POUR DE NOUVEAUX UTILISATEURS
# =============================================================================
print("\n" + "="*70)
print("CODE POUR PRÉDICTION SUR DONNÉES DE TEST")
print("="*70)

print("""
# Pour appliquer le modèle sur le fichier de test:

import pickle
import pandas as pd

# Charger le modèle
with open('best_model.pkl', 'rb') as f:
    saved = pickle.load(f)

model = saved['model']
scaler = saved['scaler']
feature_cols = saved['feature_cols']

# Charger les données de test
df_test = pd.read_csv('df_test.csv')

# Appliquer la même fonction de feature engineering
test_features = create_user_features(df_test, REFERENCE_DATE)

# Préparer les features
X_new = test_features[feature_cols].fillna(0)

# Normaliser si nécessaire (pour Logistic Regression)
# X_new_scaled = scaler.transform(X_new)

# Prédictions
predictions = model.predict_proba(X_new)[:, 1]

# Créer le fichier de soumission
submission = pd.DataFrame({
    'userId': test_features['userId'],
    'churn_probability': predictions
})
submission.to_csv('submission.csv', index=False)
""")

print("\n" + "="*70)
print("✅ PIPELINE TERMINÉ AVEC SUCCÈS!")
print("="*70)
