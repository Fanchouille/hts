## Simple script with useful functions to manipulate DFs

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
