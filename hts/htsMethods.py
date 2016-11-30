import numpy as np

from hierarchyHandler import cHierarchyHandler


class cHtsOptimizer(cHierarchyHandler):
    def __init__(self, iLevelDfDict, iHierarchyDf=None, iHierarchyOrder=None):
        """

        :param iHierarchyDf: a Df with the hierarchy
        :param iHierarchyOrder: a dict with names and level values : 0 is the base level (most granular)
        """
        cHierarchyHandler.__init__(self, iHierarchyDf, iHierarchyOrder)
        self.mStructure = self.create_structure()
        self.mLevelDfDict = iLevelDfDict

    def computeTopDownHistoricalProportions(self, iTsCol=None):
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
                            lYxx = self.mLevelDfDict[level - 1].get_group(col1).loc[:, iTsCol].values
                            lYt = self.mLevelDfDict[level].get_group(col).loc[:, iTsCol].values
                            oAvgHistProp[col][col1] = (lYxx[np.where(lYt > 0)] / lYt[np.where(lYt > 0)]).mean()
                            oPropHistAvg[col][col1] = lYxx.mean() / lYt.mean()

        return oAvgHistProp, oPropHistAvg

    def computeTopDownForecasts(self, iProp, iPrefix, iInitialForecastCol='Forecast'):
        """

        :param iProp: proportion used to computes TD
        :param iPrefix: prefix to add to ne forecasts
        :param iInitialForecastCol: initial forecast column
        :return: same dict of Dfs with updated forecasts cols
        """
        oForecast_DF_TD = {}
        lLevelsReversed = sorted(self.mStructure.keys(), reverse=True)

        # Highest level (less granular)
        lHighestLevel = lLevelsReversed[0]
        results_highest_lvl = []
        # Highest is unchanged (because it is TOP DOWN approach)
        for col in self.mStructure[lHighestLevel].keys():
            current_df = self.mLevelDfDict[lHighestLevel - 1].get_group(col).copy()
            current_df.loc[:, iInitialForecastCol + "_" + iPrefix] = current_df.loc[iInitialForecastCol]

        # From less to most granular
        for level in lLevelsReversed:
            for col in self.mStructure[level].keys():
                for col1 in self.mStructure[level][col]:
                    l_new_TD_forecast = oForecast_DF_TD[col + "_" + iPrefix + "_Forecast"] * iProp[col][col1]
                    oForecast_DF_TD[col1 + "_" + iPrefix + "_Forecast"] = l_new_TD_forecast

        return oForecast_DF_TD
