# Numpy and pandas
import numpy as np


class cHierarchyHandler:
    def __init__(self, iHierarchyDf=None, iHierarchyOrder=None):
        """

        :param iHierarchyDf: a Df with the hierarchy
        :param iHierarchyOrder: a dict with names and level values : 0 is the base level (most granular)
        """
        self.mHierarchy = iHierarchyDf
        self.mHierarchyOrder = iHierarchyOrder
        self.mRevHierarchyOrder = dict((v, k) for k, v in iHierarchyOrder.iteritems())

    def create_structure(self):
        """

        :return: the structure from hierarchyDf and hierarchyOrder
        """
        self.mHierarchy = self.mHierarchy.drop_duplicates()
        self.mHierarchy = self.mHierarchy.set_index(np.array(range(self.mHierarchy.shape[0])))
        self.mHierarchy = self.mHierarchy.rename(columns=self.mHierarchyOrder)

        # Create the structure of hierarchy
        oStructure = {}
        # List of levels
        lLevels = self.mHierarchy.columns.values

        # For each level, create an empty dict
        for level in sorted(lLevels):
            oStructure[level] = {}
        # Fill it !
        for row in range(self.mHierarchy.shape[0]):
            for level in lLevels:
                col = self.mHierarchy[level][row]
                if col not in oStructure[level].keys():
                    oStructure[level][col] = set()
                if level > 0:
                    col1 = self.mHierarchy[level - 1][row]
                    oStructure[level][col].add(col1)

        return oStructure

    def create_summing_matrix(self):
        """

        iHierachy is as DF with the hierarchy of data
        level 0 is the base level (most granular level)
        :return: the summing matrix
        """
        lStructure = self.create_structure()

        lNbNodes = sum([len(lStructure[level]) for level in lStructure.keys()])
        lBaseLevelCount = len(lStructure[0])

        lIndices = {}
        # Summing matrix
        oSummingMatrix = np.zeros((lNbNodes, lBaseLevelCount))
        for level in lStructure.keys():
            if level > 0:
                for col in lStructure[level].keys():
                    i = len(lIndices)
                    lIndices[col] = i
                    for col1 in lStructure[level][col]:
                        ii = lIndices[col1]
                        for j in range(lBaseLevelCount):
                            oSummingMatrix[i][j] = oSummingMatrix[ii][j] + oSummingMatrix[i][j]
            else:
                # Base level filling
                for col in lStructure[level].keys():
                    lNew_index = len(lIndices)
                    lIndices[col] = lNew_index
                    oSummingMatrix[lNew_index][lNew_index] = 1

        return oSummingMatrix, lIndices
