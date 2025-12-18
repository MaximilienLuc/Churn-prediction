"""
ANALYSE GÉOGRAPHIQUE DU CHURN PAR RÉGION US
==============================================

Analyse complète du churn par région Census US (9 régions)
- Distribution du dataset
- Churn rate par région
- Comportements par région
- Performance du modèle
- Visualisations

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# MAPPING DES ÉTATS VERS RÉGIONS CENSUS US
# ============================================================================

STATE_TO_REGION = {
    # Pacific
    'CA': 'Pacific', 'OR': 'Pacific', 'WA': 'Pacific', 'AK': 'Pacific', 'HI': 'Pacific',
    
    # Mountain
    'MT': 'Mountain', 'ID': 'Mountain', 'WY': 'Mountain', 'NV': 'Mountain',
    'UT': 'Mountain', 'CO': 'Mountain', 'AZ': 'Mountain', 'NM': 'Mountain',
    
    # West North Central
    'ND': 'West North Central', 'SD': 'West North Central', 'NE': 'West North Central',
    'KS': 'West North Central', 'MN': 'West North Central', 'IA': 'West North Central',
    'MO': 'West North Central',
    
    # West South Central
    'OK': 'West South Central', 'AR': 'West South Central', 
    'LA': 'West South Central', 'TX': 'West South Central',
    
    # East North Central
    'WI': 'East North Central', 'IL': 'East North Central', 'IN': 'East North Central',
    'MI': 'East North Central', 'OH': 'East North Central',
    
    # East South Central
    'KY': 'East South Central', 'TN': 'East South Central',
    'MS': 'East South Central', 'AL': 'East South Central',
    
    # South Atlantic
    'WV': 'South Atlantic', 'MD': 'South Atlantic', 'DE': 'South Atlantic',
    'VA': 'South Atlantic', 'NC': 'South Atlantic', 'SC': 'South Atlantic',
    'GA': 'South Atlantic', 'FL': 'South Atlantic', 'DC': 'South Atlantic',
    
    # Middle Atlantic
    'NY': 'Middle Atlantic', 'PA': 'Middle Atlantic', 'NJ': 'Middle Atlantic',
    
    # New England
    'ME': 'New England', 'NH': 'New England', 'VT': 'New England',
    'MA': 'New England', 'RI': 'New England', 'CT': 'New England'
}


def extract_state(location):
    """
    Extrait le code état depuis location
    Ex: "New York-Newark-Jersey City, NY-NJ-PA" → "NY"
    """
    if pd.isna(location):
        return None
    
    # Format: "City, STATE" ou "City, STATE-STATE"
    # Prendre après la dernière virgule
    parts = str(location).split(',')
    if len(parts) < 2:
        return None
    
    # Extraire les états (après la virgule)
    state_part = parts[-1].strip()
    
    # Prendre le premier état mentionné
    # Ex: "NY-NJ-PA" → "NY"
    states = state_part.split('-')
    primary_state = states[0].strip()
    
    # Vérifier que c'est bien un code état (2 lettres)
    if len(primary_state) == 2 and primary_state.isalpha():
        return primary_state.upper()
    
    return None


def map_to_region(state):
    """Mapper état vers région Census"""
    if state is None:
        return 'Unknown'
    return STATE_TO_REGION.get(state, 'Unknown')


# ============================================================================
# ANALYSE COMPLÈTE
# ============================================================================

def geographic_analysis(df_raw, df_features=None):
    """
    Analyse géographique complète
    
    Args:
        df_raw: DataFrame original avec colonne 'location'
        df_features: DataFrame avec features + target (optionnel)
    """
    
    print("="*80)
    print("ANALYSE GÉOGRAPHIQUE DU CHURN PAR RÉGION US")
    print("="*80)
    print()
    
    # ========================================================================
    # PRÉPARATION
    # ========================================================================
    
    print("Extraction des régions...")
    df_raw = df_raw.copy()
    df_raw['state'] = df_raw['location'].apply(extract_state)
    df_raw['region'] = df_raw['state'].apply(map_to_region)
    
    # Statistiques de parsing
    total_rows = len(df_raw)
    parsed = (df_raw['region'] != 'Unknown').sum()
    print(f"Lignes parsées: {parsed:,} / {total_rows:,} ({parsed/total_rows*100:.1f}%)")
    print()
    
    # Filter unknown
    df_raw_clean = df_raw[df_raw['region'] != 'Unknown'].copy()
    
    # ========================================================================
    # 1. DISTRIBUTION DU DATASET
    # ========================================================================
    
    print("="*80)
    print("1. DISTRIBUTION DU DATASET PAR RÉGION")
    print("="*80)
    print()
    
    # Par région
    region_counts = df_raw_clean['region'].value_counts()
    region_pct = (region_counts / len(df_raw_clean) * 100).round(2)
    
    distribution_df = pd.DataFrame({
        'Sessions': region_counts,
        '% Total': region_pct
    })
    
    # Nombre de users uniques par région
    users_by_region = df_raw_clean.groupby('region')['userId'].nunique()
    distribution_df['Users'] = users_by_region
    distribution_df['Sessions/User'] = (distribution_df['Sessions'] / distribution_df['Users']).round(1)
    
    print(distribution_df.sort_values('Sessions', ascending=False))
    print()
    
    # ========================================================================
    # 2. CHURN RATE PAR RÉGION
    # ========================================================================
    
    print("="*80)
    print("2. CHURN RATE PAR RÉGION")
    print("="*80)
    print()
    
    # Identifier les churners
    churners = df_raw_clean[df_raw_clean['page'] == 'Cancellation Confirmation']['userId'].unique()
    df_raw_clean['is_churner'] = df_raw_clean['userId'].isin(churners).astype(int)
    
    # Churn rate par région (au niveau user)
    user_region = df_raw_clean.groupby('userId').agg({
        'region': 'first',
        'is_churner': 'max'  # 1 si user a churné
    })
    
    churn_by_region = user_region.groupby('region').agg({
        'is_churner': ['sum', 'count', 'mean']
    })
    churn_by_region.columns = ['Churners', 'Total_Users', 'Churn_Rate']
    churn_by_region['Churn_Rate'] = (churn_by_region['Churn_Rate'] * 100).round(2)
    churn_by_region = churn_by_region.sort_values('Churn_Rate', ascending=False)
    
    # Baseline
    baseline_churn = user_region['is_churner'].mean() * 100
    churn_by_region['vs_Baseline'] = (churn_by_region['Churn_Rate'] - baseline_churn).round(2)
    
    print(f"Baseline churn rate: {baseline_churn:.2f}%")
    print()
    print(churn_by_region)
    print()
    
    # Test statistique (Chi-square)
    contingency_table = pd.crosstab(user_region['region'], user_region['is_churner'])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"Chi-square test: χ²={chi2:.2f}, p-value={p_value:.4f}")
    if p_value < 0.05:
        print("✅ Différences significatives entre régions (p < 0.05)")
    else:
        print("❌ Pas de différence significative entre régions (p >= 0.05)")
    print()
    
    # ========================================================================
    # 3. COMPORTEMENTS PAR RÉGION
    # ========================================================================
    
    if df_features is not None:
        print("="*80)
        print("3. COMPORTEMENTS PAR RÉGION")
        print("="*80)
        print()
        
        # Ajouter région aux features
        if 'userId' in df_features.columns:
            user_to_region = df_raw_clean.groupby('userId')['region'].first().to_dict()
            df_features['region'] = df_features['userId'].map(user_to_region)
            df_features_clean = df_features[df_features['region'].notna()].copy()
            
            # Features clés à analyser
            key_features = [
                'frustration_score',
                'thumbs_down_last_14days',
                'songs_listened_last_14days',
                'consecutive_days_inactive',
                'days_since_registration',
                'is_paid'
            ]
            
            # Moyennes par région
            behavior_by_region = df_features_clean.groupby('region')[key_features].mean().round(2)
            
            print("Moyennes des features clés par région:")
            print()
            print(behavior_by_region)
            print()
            
            # ANOVA pour voir si différences significatives
            print("Tests ANOVA (différences entre régions):")
            for feature in key_features:
                groups = [df_features_clean[df_features_clean['region'] == r][feature].dropna() 
                         for r in df_features_clean['region'].unique()]
                f_stat, p_val = stats.f_oneway(*groups)
                sig = "✅ Significatif" if p_val < 0.05 else "❌ Non significatif"
                print(f"{feature:30s}: F={f_stat:.2f}, p={p_val:.4f} {sig}")
            print()
    
    # ========================================================================
    # 4. PERFORMANCE DU MODÈLE PAR RÉGION
    # ========================================================================
    
    if df_features is not None and 'will_churn_10days' in df_features.columns:
        print("="*80)
        print("4. DISTRIBUTION DU TARGET PAR RÉGION")
        print("="*80)
        print()
        
        target_by_region = df_features_clean.groupby('region')['will_churn_10days'].agg([
            'sum', 'count', 'mean'
        ])
        target_by_region.columns = ['Positives', 'Total', 'Positive_Rate']
        target_by_region['Positive_Rate'] = (target_by_region['Positive_Rate'] * 100).round(2)
        target_by_region = target_by_region.sort_values('Positive_Rate', ascending=False)
        
        print(target_by_region)
        print()
    
    # ========================================================================
    # 5. VISUALISATIONS
    # ========================================================================
    
    print("="*80)
    print("5. GÉNÉRATION DES VISUALISATIONS")
    print("="*80)
    print()
    
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: Distribution des sessions
    ax1 = plt.subplot(2, 3, 1)
    region_counts_sorted = region_counts.sort_values(ascending=True)
    region_counts_sorted.plot(kind='barh', ax=ax1, color='steelblue')
    ax1.set_xlabel('Nombre de sessions')
    ax1.set_title('Distribution des sessions par région', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Plot 2: Churn rate par région
    ax2 = plt.subplot(2, 3, 2)
    churn_sorted = churn_by_region.sort_values('Churn_Rate', ascending=True)
    colors = ['red' if x > baseline_churn else 'green' for x in churn_sorted['Churn_Rate']]
    churn_sorted['Churn_Rate'].plot(kind='barh', ax=ax2, color=colors)
    ax2.axvline(baseline_churn, color='black', linestyle='--', label=f'Baseline ({baseline_churn:.1f}%)')
    ax2.set_xlabel('Churn rate (%)')
    ax2.set_title('Churn rate par région', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='x', alpha=0.3)
    
    # Plot 3: Sessions par user
    ax3 = plt.subplot(2, 3, 3)
    sessions_per_user = distribution_df.sort_values('Sessions/User', ascending=True)
    sessions_per_user['Sessions/User'].plot(kind='barh', ax=ax3, color='coral')
    ax3.set_xlabel('Sessions par user')
    ax3.set_title('Engagement moyen par région', fontsize=12, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    
    # Plot 4: Frustration par région
    if df_features is not None and 'frustration_score' in behavior_by_region.columns:
        ax4 = plt.subplot(2, 3, 4)
        frustration_sorted = behavior_by_region['frustration_score'].sort_values(ascending=True)
        frustration_sorted.plot(kind='barh', ax=ax4, color='orange')
        ax4.set_xlabel('Frustration score moyen')
        ax4.set_title('Frustration moyenne par région', fontsize=12, fontweight='bold')
        ax4.grid(axis='x', alpha=0.3)
    
    # Plot 5: Activité par région
    if df_features is not None and 'songs_listened_last_14days' in behavior_by_region.columns:
        ax5 = plt.subplot(2, 3, 5)
        songs_sorted = behavior_by_region['songs_listened_last_14days'].sort_values(ascending=True)
        songs_sorted.plot(kind='barh', ax=ax5, color='purple')
        ax5.set_xlabel('Songs listened (14d)')
        ax5.set_title('Activité moyenne par région', fontsize=12, fontweight='bold')
        ax5.grid(axis='x', alpha=0.3)
    
    # Plot 6: Distribution users paid vs free
    if df_features is not None and 'is_paid' in behavior_by_region.columns:
        ax6 = plt.subplot(2, 3, 6)
        paid_pct = behavior_by_region['is_paid'].sort_values(ascending=True) * 100
        paid_pct.plot(kind='barh', ax=ax6, color='gold')
        ax6.set_xlabel('% Users paid')
        ax6.set_title('Proportion users payants par région', fontsize=12, fontweight='bold')
        ax6.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/geographic_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Graphiques sauvegardés: geographic_analysis.png")
    print()
    
    # ========================================================================
    # 6. RECOMMANDATIONS
    # ========================================================================
    
    print("="*80)
    print("6. RECOMMANDATIONS BUSINESS")
    print("="*80)
    print()
    
    # Régions à risque (churn > baseline + 2%)
    high_risk = churn_by_region[churn_by_region['vs_Baseline'] > 2]
    if len(high_risk) > 0:
        print("⚠️  RÉGIONS À RISQUE (churn > baseline + 2%):")
        for region in high_risk.index:
            churn_rate = high_risk.loc[region, 'Churn_Rate']
            diff = high_risk.loc[region, 'vs_Baseline']
            users = high_risk.loc[region, 'Total_Users']
            print(f"  - {region}: {churn_rate:.1f}% (+{diff:.1f}pp), {users:,} users")
        print()
    
    # Régions performantes (churn < baseline - 2%)
    low_risk = churn_by_region[churn_by_region['vs_Baseline'] < -2]
    if len(low_risk) > 0:
        print("✅ RÉGIONS PERFORMANTES (churn < baseline - 2%):")
        for region in low_risk.index:
            churn_rate = low_risk.loc[region, 'Churn_Rate']
            diff = low_risk.loc[region, 'vs_Baseline']
            users = low_risk.loc[region, 'Total_Users']
            print(f"  - {region}: {churn_rate:.1f}% ({diff:.1f}pp), {users:,} users")
        print()
    
    # Actions recommandées
    print("💡 ACTIONS RECOMMANDÉES:")
    print("  1. Cibler les régions à risque avec campagnes de rétention")
    print("  2. Étudier les best practices des régions performantes")
    print("  3. Adapter les features du modèle par région (si différences significatives)")
    print("  4. Considérer des modèles régionaux si performances très différentes")
    print()
    
    print("="*80)
    print("✅ ANALYSE TERMINÉE")
    print("="*80)
    
    return {
        'distribution': distribution_df,
        'churn_by_region': churn_by_region,
        'behavior_by_region': behavior_by_region if df_features is not None else None
    }


# ============================================================================
# UTILISATION
# ============================================================================
import pandas as pd

if __name__ == "__main__":
    # Charger les données
    print("Chargement des données...")
    df_raw = pd.read_parquet("data/train.parquet")  # Dataset original avec location
    
    # Optionnel : charger les features
    try:
        df_features = pd.read_csv('features_with_target.csv')
        print("Features chargées")
    except:
        df_features = None
        print("Pas de features (analyse limitée)")
    
    print()
    
    # Lancer l'analyse
    results = geographic_analysis(df_raw, df_features)
    
    # Sauvegarder les résultats
    results['distribution'].to_csv('/mnt/user-data/outputs/distribution_by_region.csv')
    results['churn_by_region'].to_csv('/mnt/user-data/outputs/churn_by_region.csv')
    
    if results['behavior_by_region'] is not None:
        results['behavior_by_region'].to_csv('/mnt/user-data/outputs/behavior_by_region.csv')
    
    print("\n✅ Résultats sauvegardés dans /mnt/user-data/outputs/")
