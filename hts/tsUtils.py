# Simple script with useful functions to manipulate DFs

import pandas as pd


def cross_join_2_dfs(iDf1, iDf2, **kwargs):
    """

    :param iDf1: 1st dataframe
    :param iDf2: 2nd dataframe
    :param kwargs:
    :return: cross joined DF
    """
    iDf1['_tmpkey'] = 1
    iDf2['_tmpkey'] = 1
    oRes = pd.merge(iDf1, iDf2, on='_tmpkey', **kwargs).drop('_tmpkey', axis=1)

    iDf1.drop('_tmpkey', axis=1, inplace=True)
    iDf2.drop('_tmpkey', axis=1, inplace=True)

    return oRes


def cross_join_from_dict(iFeatureDict):
    """

    :param iFeatureDict: Dict of features to cross join {ColName1 : [Col1 distinct values],
    ColName2 : [Col2 distinct values], ...}
    :return: a DF with cross joined-data : each combination of features in iFeatureDict is constructed
    """
    if len(iFeatureDict.keys()) < 2:
        print('Not enough data to cross join (less than 2 features)')
        return None

    oResultDf = pd.DataFrame([1])
    lSize = 1
    for col in iFeatureDict.keys():
        lCurrentDf = pd.DataFrame(iFeatureDict[col], columns=[col])
        lSize = lSize * len(iFeatureDict[col])
        oResultDf = cross_join_2_dfs(oResultDf, lCurrentDf)

    oResultDf.drop(0, axis=1, inplace=True)

    if lSize != oResultDf.shape[0]:
        print('Error encountered : DF has the wrong number of lines !.')
        return None
    return oResultDf


def split_tr_test(iDf, iDateCol, iDateThreshold):
    """

    :param iDf: dataframe with TS data
    :param iDateCol: date column
    :param iDateThreshold: data threshold to split training and testing sets
    :return: training / testing sets
    """
    oTrDf = iDf.loc[iDf[iDateCol] <= iDateThreshold, :]
    oTsDf = iDf.loc[iDf[iDateCol] > iDateThreshold, :]

    return oTrDf, oTsDf


def create_df_dict_for_each_level(iDf, iDateCol, iTargetCols, iHierarchyOrder):
    """

    :param iDf: input DF
    :param iDateCol: date column
    :param iTargetCol: target col to aggregate
    :param iHierarchyOrder: hierarchy
    :return: dict of df with 1 DF per level of the hierarchy
    """
    lRevHierarchyOrder = dict((v, k) for k, v in iHierarchyOrder.iteritems())
    oLevelDf = {}
    for level in lRevHierarchyOrder.keys():
        oLevelDf[level] = iDf.groupby([iDateCol, lRevHierarchyOrder[level]], as_index=False)[iTargetCols].sum()

    return oLevelDf


def create_full_df_with_hierarchy(iDf, iHierarchy, iHierarchyOrder, iTargetCols, iDateCol, iFreq, iFromDateCol=True,
                                  iDateRange=None):
    """

    :param iDf: input DF
    :param iHierarchy: hierarchy
    :param iHierarchyOrder: hierarchy order
    :param iTargetCols: iTargetCols to keep
    :param iDateCol: if iFromDateCol is True, get date range from iDateCol
    :param iFreq: frequency of dates (Days, Weeks, etc)
    :param iFromDateCol: boolean True or False : if False, get date range from iDateRange
    :param iDateRange: if iFromDateCol is False, use this date range
    :return: full df with all cross join between dates and level 0 + all upper levels
    """
    lRevHierarchyOrder = dict((v, k) for k, v in iHierarchyOrder.iteritems())

    if iFromDateCol:
        lDateRange = pd.date_range(start=iDf.loc[:, iDateCol].min(), end=iDf.loc[:, iDateCol].max(), freq=iFreq)
    else:
        lDateRange = pd.date_range(start=pd.to_datetime(iDateRange[0]), end=pd.to_datetime(iDateRange[1]), freq=iFreq)

    lLevel0Uniq = iHierarchy.loc[:, lRevHierarchyOrder[0]].unique()

    featuresDict = {iDateCol: lDateRange, lRevHierarchyOrder[0]: lLevel0Uniq}

    # Use tsUtils lib to create full dataset
    oFullDf = cross_join_from_dict(featuresDict)

    # Add info of destination for each transporteur
    oFullDf = pd.merge(oFullDf, iHierarchy, on=lRevHierarchyOrder[0], how='left')
    # Fill with data
    oFullDf = pd.merge(oFullDf, iDf.groupby([iDateCol, lRevHierarchyOrder[0]], as_index=False)[iTargetCols].sum(),
                       on=[iDateCol, lRevHierarchyOrder[0]], how='left').fillna(0)

    return oFullDf
