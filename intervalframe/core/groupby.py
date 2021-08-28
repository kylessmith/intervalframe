import pandas as pd
from utilities import concat


class IntervalFrameGroupBy(object):
    """
    Grouped IntervalFrame

    :class:`~intervalframe.IntervalFrameGroupBy` stores grouped IntervalFrame
    """

    def __init__(self, iframe, by):
        """
        Initialize IntervalFrameGroupBy

        Parameters
        ----------


        Returns
        -------


        """

        # Initalize grouping
        self.groups = iframe.groupby(by=by).groups
        self.iframe = iframe


    def __iter__(self):
        """
        Iterate over groups
        """

        # Iterate
        for group in self.groups:
            yield self.iframe.iloc[self.groups[group].values,:]


    def _intersect_generator(self, start, end):
        """
        Intersection
        """

        # Iterate
        for group in self.groups:
            yield self.iframe.iloc[self.groups[group].values,:].intersect(start, end)


    def intersect(self, start, end):
        """
        Find overlapping intervals in IntervalFrameGroupBy

        Parameters
        ----------


        Returns
        -------


        """

        # Intersect group
        intersection = concat(self._intersect_generator(start, end))

        return intersection