import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import hierarchyHandler as hh
import htsMethods as htsm
import tsUtils as tsu

data_path = '../data/data_log.csv'


# LOAD DATA
data_log = pd.read_csv(data_path, delimiter=';')
# Handle datetimes and create a filed with just Date (day only)
data_log.loc[:, 'Date'] = pd.to_datetime(data_log.loc[:, 'Date'], format="%d/%m/%Y %H:%M")
data_log.loc[:, 'DateDay'] = pd.to_datetime(data_log.loc[:, 'Date'].dt.date)
# Add info of destination for each carrier
data_log.loc[:, 'Pays'] = 'France'
data_log.loc[data_log['Transporteur'] == 'LPS', 'Pays'] = 'Suisse'
data_log.loc[data_log['Transporteur'] == 'TAX', 'Pays'] = 'Belgique'
data_log.loc[data_log['Transporteur'] == 'CHI', 'Pays'] = 'International'
data_log.loc[:, 'All'] = 'All'

# BUILD HIERARCHY
hierarchy = data_log.loc[:, ['Transporteur', 'Pays', 'All']].drop_duplicates()
hierarchyOrder = {'Transporteur': 0, 'Pays': 1, 'All': 2}
revHierarchyOrder = dict((v, k) for k, v in hierarchyOrder.iteritems())

# HIERARCHY HANDLER USE
cHH = hh.cHierarchyHandler(hierarchy, hierarchyOrder)
structure = cHH.create_structure()


# CREATE A FULL DATASETS WITH NO "HOLES"
lDateRange = pd.date_range(start=data_log.loc[:, 'DateDay'].min(), end=data_log.loc[:, 'DateDay'].max(), freq='D')
featuresDict = {'DateDay': lDateRange, 'Transporteur': hierarchy.loc[:, 'Transporteur'].unique()}
# Use tsUtils lib to create full dataset
fullDf = tsu.cross_join_from_dict(featuresDict)

# Add info of destination for each transporteur
fullDf = pd.merge(fullDf, hierarchy, left_on='Transporteur', right_on='Transporteur', how='left')
# Fill with data
fullDf = pd.merge(fullDf, data_log.groupby(['DateDay', 'Transporteur'], as_index=False)[['NbColis']].sum(),
                  on=['Transporteur', 'DateDay'], how='left').fillna(0)

# BUILD DATASETS FOR EACH LEVELS
level_dfs = {}

# Aggregate data for each level
for level in cHH.mRevHierarchyOrder.keys():
    level_dfs[level] = fullDf.groupby(['DateDay', cHH.mRevHierarchyOrder[level]], as_index=False)[['NbColis']].sum()

# Featurize data for each level
featured_level_dfs = level_dfs.copy()
for level in cHH.mRevHierarchyOrder.keys():
    for i in range(1, 5):
        # Add lagged features with data from Day-1, Day-2, .., Day-4
        featured_level_dfs[level] = pd.merge(featured_level_dfs[level], level_dfs[level].set_index('DateDay')
                                             .groupby(cHH.mRevHierarchyOrder[level])[['NbColis']]
                                             .tshift(i, 'D')
                                             .rename(columns={'NbColis': 'NbColis_lag' + str(i)})
                                             .reset_index(), on=['DateDay', cHH.mRevHierarchyOrder[level]], how='left') \
            .dropna()

# Split date for training and testing sets
date_threshold = pd.to_datetime('2016-06-30')
# Create models for each levels
rf = RandomForestRegressor(n_jobs=-1, random_state=123)
results_level_dfs = {}

for level in revHierarchyOrder.keys():
    results_current_lvl = []

    # For each level
    current_level_tr, current_level_ts = tsu.split_tr_test(featured_level_dfs[level], 'DateDay', date_threshold)
    current_level_tr = current_level_tr.groupby(revHierarchyOrder[level])
    current_level_ts = current_level_ts.groupby(revHierarchyOrder[level])

    # For each value in current level
    for name, group in current_level_tr:
        current_tr_df = current_level_tr.get_group(name)
        current_ts_df = current_level_ts.get_group(name).copy()
        # Fit !
        rf.fit(current_tr_df.loc[:, ['NbColis_lag' + str(i) for i in range(1, 5)]].values,
               current_tr_df.loc[:, 'NbColis'].values)
        current_ts_df.loc[:, 'Forecast'] = rf.predict(
            current_ts_df.loc[:, ['NbColis_lag' + str(i) for i in range(1, 5)]].values)

        results_current_lvl.append(
            current_ts_df.loc[:, ['DateDay', revHierarchyOrder[level], 'NbColis', 'Forecast']])

    results_level_dfs[level] = pd.concat(results_current_lvl)

##################################################################################################################
# USE OF HTS HERE
##################################################################################################################
hts_optim = htsm.cHtsOptimizer(results_level_dfs, hierarchy, hierarchyOrder)

print hts_optim.mStructure

# TOP DOWN APPROACH
p1, p2 = hts_optim.computeTopDownHistoricalProportions(results_level_dfs, iTsCol='NbColis')
td_res = hts_optim.computeTopDownForecasts(p1, '_TD_p1', iInitialForecastCol='Forecast')

# BOTTOM - UP APPROACH
bu_res = hts_optim.computeBottomUpForecasts('DateDay', iInitialForecastCol='Forecast')

# MIDDLE OUT APPROACH
mo_res = hts_optim.computeMiddleOutForecasts('DateDay', p1, iMidLevel=1, iPrefix='MO', iInitialForecastCol='Forecast')
