import numpy as np
import pandas as pd

from hierarchyHandler import cHierarchyHandler


class cHtsOptimizer(cHierarchyHandler):
    def __init__(self, iLevelDfDict, iInitialForecastCol='Forecast', iDateCol=None, iHierarchyDf=None,
                 iHierarchyOrder=None):
        """

        :param iHierarchyDf: a Df with the hierarchy
        :param iHierarchyOrder: a dict with names and level values : 0 is the base level (most granular)
        """
        cHierarchyHandler.__init__(self, iHierarchyDf, iHierarchyOrder)
        self.mStructure = self.create_structure()
        self.mLevelDfDict = iLevelDfDict
        self.mInitialForecastCol = iInitialForecastCol
        self.mDateCol = iDateCol

    def computeTopDownHistoricalProportions(self, iLevelDfDictForProp, iTsCol=None):
        """

        :param iTscol: string of the column to be used to compute proportions
        With forecast DFs, computes the 2 kinds of proportions :
        :return: average of historical proportions, proportion of historical average
        """
        oAvgHistProp = {}
        oPropHistAvg = {}
        if iTsCol is not None:
            for level in self.mStructure.keys():
                if level > 0:
                    for col in self.mStructure[level].keys():
                        oAvgHistProp[col] = {}
                        oPropHistAvg[col] = {}
                        for col1 in self.mStructure[level][col]:
                            lYxx = iLevelDfDictForProp[level - 1].groupby(self.mRevHierarchyOrder[level - 1]).get_group(
                                col1).loc[:, iTsCol].values
                            lYt = iLevelDfDictForProp[level].groupby(self.mRevHierarchyOrder[level]).get_group(col).loc[
                                  :, iTsCol].values
                            # Trick if there is a zero in lYt
                            oAvgHistProp[col][col1] = (lYxx[np.where(lYt != 0)] / lYt[np.where(lYt != 0)]).mean()
                            # Assume lYt is not zero mean
                            oPropHistAvg[col][col1] = lYxx.mean() / lYt.mean()

        return oAvgHistProp, oPropHistAvg

    def computeTopDownForecasts(self, iProp, iPrefix):
        """
        TOP DOWN forecast : forecast is taken at maximum level and then allocated to lower levels according to iProp
        :param iProp: proportion used to computes TD
        :param iPrefix: prefix string to add to forecast col
        :param iInitialForecastCol: initial forecast column
        :return: same dict of Dfs with updated forecasts cols for the TD approach
        """
        # Copy initial forecasts (forecast for highest level will be unchanged)
        oLevelDfDictResultsTD = self.mLevelDfDict.copy()
        lLevelsReversed = sorted(self.mStructure.keys(), reverse=True)

        # Highest level (less granular)
        lHighestLevel = lLevelsReversed[0]

        # Forecast for highest level is initial forecast (because it is TOP DOWN approach)
        current_df = self.mLevelDfDict[lHighestLevel].copy()
        current_df.loc[:, self.mInitialForecastCol + "_" + iPrefix] = current_df.loc[:, self.mInitialForecastCol]
        oLevelDfDictResultsTD[lHighestLevel] = current_df

        # From less to most granular
        for level in lLevelsReversed:
            if level > 0:
                results_current_lvl = []
                for col in self.mStructure[level].keys():
                    l_new_TD_forecast_df = oLevelDfDictResultsTD[level].groupby(
                        self.mRevHierarchyOrder[level]).get_group(
                        col).copy()
                    # Get the forecast for the +1 level
                    for col1 in self.mStructure[level][col]:
                        # Replace the forecast for the current level by forecast lvl+1 * proportion
                        l_new_TD_forecast_df2 = oLevelDfDictResultsTD[level - 1].groupby(
                            self.mRevHierarchyOrder[level - 1], as_index=False).get_group(
                            col1).copy()
                        l_new_TD_forecast_df2.loc[:,
                        self.mInitialForecastCol + "_" + iPrefix] = l_new_TD_forecast_df.loc[:,
                                                                    self.mInitialForecastCol + "_" + iPrefix].values * \
                                                                    iProp[col][col1]
                        results_current_lvl.append(l_new_TD_forecast_df2)
                oLevelDfDictResultsTD[level - 1] = pd.concat(results_current_lvl)

        return oLevelDfDictResultsTD

    def computeBottomUpForecasts(self, iPrefix='BU'):
        """
        Bottom up forecasts : forecast are taken at lower levels and aggregated at upper ones
        :param iInitialForecastCol: initial forecast column
        :param iPrefix: prefix string to add to forecast col
        :return: same dict of Dfs with updated forecasts cols for the BU approach
        """
        # Copy initial forecasts (forecast for lowest level will be unchanged)
        oLevelDfDictResultsBU = self.mLevelDfDict.copy()
        lLevelsSorted = sorted(self.mStructure.keys(), reverse=False)

        # Lowest level (most granular)
        lLowestLevel = lLevelsSorted[0]

        # Forecast for lowest level is initial forecast (because it is BOTTOM UP approach)
        current_df = self.mLevelDfDict[lLowestLevel].copy()
        current_df.loc[:, self.mInitialForecastCol + "_" + iPrefix] = current_df.loc[:, self.mInitialForecastCol]
        oLevelDfDictResultsBU[lLowestLevel] = current_df

        for level in lLevelsSorted:
            if level > 0:
                results_current_lvl = []
                for col in self.mStructure[level].keys():
                    l_new_BU_forecast_df = oLevelDfDictResultsBU[level - 1].loc[
                                           oLevelDfDictResultsBU[level - 1][self.mRevHierarchyOrder[level - 1]].isin(
                                               self.mStructure[level][col]), :]
                    l_new_TD_forecast_df2 = oLevelDfDictResultsBU[level].groupby(
                        self.mRevHierarchyOrder[level]).get_group(
                        col).copy()
                    # print l_new_BU_forecast_df.groupby(iDateCol)[iInitialForecastCol + "_" + iPrefix].sum().head()
                    l_new_TD_forecast_df2.loc[:, self.mInitialForecastCol + "_" + iPrefix] = \
                        l_new_BU_forecast_df.groupby(self.mDateCol)[
                            self.mInitialForecastCol + "_" + iPrefix].sum().values
                    results_current_lvl.append(l_new_TD_forecast_df2)
                oLevelDfDictResultsBU[level] = pd.concat(results_current_lvl)

        return oLevelDfDictResultsBU

    def computeMiddleOutForecasts(self, iProp, iMidLevel, iPrefix='MO'):
        """

        :param iDateCol: string of column with date
        :param iProp: proportion computed for TD
        :param iMidLevel: number of mid level ( == number levels // 2)
        :param iPrefix: string to add to initial forecast column name
        :param iInitialForecastCol: string of column name containing initial forecast
        :return: dict of DF with midlle out forecasts
        """
        # Copy initial forecasts (forecast for mid level will be unchanged)
        oLevelDfDictResultsMO = self.mLevelDfDict.copy()
        # Forecast for mid level is initial forecast (because it is MIDDLE OUT approach)
        current_df = self.mLevelDfDict[iMidLevel].copy()
        current_df.loc[:, self.mInitialForecastCol + "_" + iPrefix] = current_df.loc[:, self.mInitialForecastCol]
        oLevelDfDictResultsMO[iMidLevel] = current_df

        if len(self.mStructure.keys()) > 2:
            # lower levels .... top-down starting from the middle.
            levels_below = sorted([level for level in self.mStructure.keys() if (level <= iMidLevel)],
                                  reverse=True)
            levels_above = sorted([level for level in self.mStructure.keys() if (level > iMidLevel)],
                                  reverse=False)

            # lower level : TD approach
            for level in levels_below:
                if level > 0:
                    results_current_lvl = []
                    for col in self.mStructure[level].keys():
                        l_new_TD_forecast_df = oLevelDfDictResultsMO[level].groupby(
                            self.mRevHierarchyOrder[level]).get_group(
                            col).copy()
                        # Get the forecast for the +1 level
                        for col1 in self.mStructure[level][col]:
                            # Replace the forecast for the current level by forecast lvl+1 * proportion
                            l_new_TD_forecast_df2 = oLevelDfDictResultsMO[level - 1].groupby(
                                self.mRevHierarchyOrder[level - 1], as_index=False).get_group(
                                col1).copy()
                            l_new_TD_forecast_df2.loc[:,
                            self.mInitialForecastCol + "_" + iPrefix] = l_new_TD_forecast_df.loc[:,
                                                                        self.mInitialForecastCol + "_" + iPrefix].values * \
                                                                        iProp[col][col1]
                            results_current_lvl.append(l_new_TD_forecast_df2)
                    oLevelDfDictResultsMO[level - 1] = pd.concat(results_current_lvl)

            # higher levels : BU approach
            for level in levels_above:
                if level > 0:
                    results_current_lvl = []
                    for col in self.mStructure[level].keys():
                        l_new_BU_forecast_df = oLevelDfDictResultsMO[level - 1].loc[
                                               oLevelDfDictResultsMO[level - 1][
                                                   self.mRevHierarchyOrder[level - 1]].isin(
                                                   self.mStructure[level][col]), :]
                        l_new_TD_forecast_df2 = oLevelDfDictResultsMO[level].groupby(
                            self.mRevHierarchyOrder[level]).get_group(
                            col).copy()
                        l_new_TD_forecast_df2.loc[:, self.mInitialForecastCol + "_" + iPrefix] = \
                            l_new_BU_forecast_df.groupby(self.mDateCol)[
                                self.mInitialForecastCol + "_" + iPrefix].sum().values
                        results_current_lvl.append(l_new_TD_forecast_df2)
                    oLevelDfDictResultsMO[level] = pd.concat(results_current_lvl)

        return oLevelDfDictResultsMO

    def computeOptimalCombination(self, iPrefix='OC'):
        """

        :return: dict of results dfs for the optimal approach
        """
        oLevelDfDictResultsOC = self.mLevelDfDict.copy()

        # unpacking all datasets to create multiplication matrix
        i = 0
        for level in self.mStructure.keys():
            for col in self.mStructure[level].keys():
                l_current_forecast_df = oLevelDfDictResultsOC[level].groupby(self.mRevHierarchyOrder[level]).get_group(
                    col).copy()
                l_current_row = l_current_forecast_df.loc[:, self.mInitialForecastCol].values.reshape(1, -1)
                if i == 0:
                    M = l_current_row
                else:
                    M = np.vstack((M, l_current_row))
                i += 1

        # Create Summing matrix and computes its pseudo inv to get optimal forecasts
        S = self.create_summing_matrix()
        # Pseudo inverse of matrix
        lPInv = np.linalg.inv(S.T.dot(S)).dot(S.T)

        lOptimalForecasts = S.dot(lPInv.dot(M))

        # Repack optimal forecasts in DF dict of results
        i = 0
        for level in self.mStructure.keys():
            results_current_lvl = []
            for col in self.mStructure[level].keys():
                l_current_forecast_df = oLevelDfDictResultsOC[level].groupby(self.mRevHierarchyOrder[level]).get_group(
                    col).copy()
                l_current_forecast_df.loc[:, self.mInitialForecastCol + "_" + iPrefix] = lOptimalForecasts[i, :]
                results_current_lvl.append(l_current_forecast_df)
                i += 1
            oLevelDfDictResultsOC[level] = pd.concat(results_current_lvl)

        return oLevelDfDictResultsOC
