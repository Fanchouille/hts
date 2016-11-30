import pandas as pd

import hierarchyHandler as hh
import tsutils as tsu

data_path = 'C:/cygwin64/home/FanchUser/Logistique/data_log.csv'

# LOAD DATA
data_log = pd.read_csv(data_path, delimiter=';')
# Handle datetimes and create a filed with just Date (day only)
data_log.loc[:, 'Date'] = pd.to_datetime(data_log.loc[:, 'Date'], format="%d/%m/%Y %H:%M")
data_log.loc[:, 'DateDay'] = pd.to_datetime(data_log.loc[:, 'Date'].dt.date)
# Add info of destination for each transporteur
data_log.loc[:, 'Pays'] = 'France'
data_log.loc[data_log['Transporteur'] == 'LPS', 'Pays'] = 'Suisse'
data_log.loc[data_log['Transporteur'] == 'TAX', 'Pays'] = 'Belgique'
data_log.loc[data_log['Transporteur'] == 'CHI', 'Pays'] = 'International'
data_log.loc[:, 'All'] = 'All'

# BUILD HIERARCHY
hierarchy = data_log.loc[:, ['Transporteur', 'Pays', 'All']].drop_duplicates()
hierarchyOrder = {'Transporteur': 0, 'Pays': 1, 'All': 2}

# HIERARCHY HANDLER USE
cHH = hh.cHierarchyHandler(hierarchy, hierarchyOrder)
structure = cHH.create_structure()
summing_matrix = cHH.create_summing_matrix()

# CREATE A FULL DATASETS WITH NO "HOLES"
lDateRange = pd.date_range(start=data_log.loc[:, 'DateDay'].min(), end=data_log.loc[:, 'DateDay'].max(), freq='D')
featuresDict = {'DateDay': lDateRange, 'Transporteur': hierarchy.loc[:, 'Transporteur'].unique()}
# Use tsutils lib to create full dataset
fullDf = tsu.cross_join_from_dict(featuresDict)

# Add info of destination for each transporteur
fullDf = pd.merge(fullDf, hierarchy, left_on='Transporteur', right_on='Transporteur', how='left')
# Fill with data
fullDf = pd.merge(fullDf, data_log.groupby(['DateDay', 'Transporteur'], as_index=False)[['NbColis']].sum(),
                  on=['Transporteur', 'DateDay'], how='left').fillna(0)

# BUILD DATASETS FOR EACH LEVELS
level_dfs = {}

for level in cHH.mRevHierarchyOrder.keys():
    level_dfs[level] = fullDf.groupby(['DateDay', cHH.mRevHierarchyOrder[level]], as_index=False)[['NbColis']].sum()

featured_level_dfs = level_dfs.copy()
for level in cHH.mRevHierarchyOrder.keys():
    for i in range(1, 5):
        # Add lagged features with data from Day-1, Day-2, .., Day-4
        featured_level_dfs[level] = pd.merge(featured_level_dfs[level], level_dfs[level].set_index('DateDay')
                                             .groupby(cHH.mRevHierarchyOrder[level])[['NbColis']]
                                             .tshift(i, 'D')
                                             .rename(columns={'NbColis': 'NbColis_lag' + str(i)})
                                             .reset_index(), on=['DateDay', cHH.mRevHierarchyOrder[level]], how='left')
