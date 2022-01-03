import numpy as np
import pandas as pd
from ailist import IntervalArray, LabeledIntervalArray
from ailist import Interval, LabeledInterval
from .. import IntervalFrame
from .. import IntervalSeries


class iLocator(object):
    """
    Position indexer

    :class:`~intervalframe.iLocator` indexes intervals
    """
    
    def __init__(self, iobject):
        """
        Initialize iLocator

        Parameters
        ----------
            iframe : IntervalFrame or IntervalSeries
                Intervals to index

        Returns
        -------
            None

        """
        
        # Record Interval object
        self.iobject = iobject
        
        
    def __getitem__(self, args):
        """
        Get index
        """
        
        # Check args is of length two
        if len(args) != len(self.iobject.shape):
            raise IndexError("Wrong number of indexers.")

        # Get index
        index = self.iobject.index[args[0]]
        
        # Create DataFrame or Series
        if isinstance(self.iobject, IntervalFrame.IntervalFrame):
            if self.iobject.df.shape[1] > 0:
                if np.issubdtype(type(args[0]), np.integer):
                    data = self.iobject.df.iloc[[args[0]], args[1]]
                elif np.issubdtype(type(args[1]), np.integer):
                    data = self.iobject.df.iloc[args[0], args[1]]
                else:
                    data = pd.DataFrame(self.iobject.df.values[args],
                                    columns=self.iobject.df.columns.values).astype(self.iobject.df.dtypes.to_dict(), copy=False)
            else:
                data = pd.DataFrame([], index=range(len(index)))
        else:
            if self.iobject.series.shape[0] > 0:
                data = pd.Series(self.iobject.series.values[args],
                                    name=self.iobject.series.name).astype(self.iobject.series.dtype.to_dict(), copy=False)
            else:
                data = pd.Series([], index=range(len(index)))
                          
        # Create IntervalFrame of IntervalSeries
        if isinstance(data, pd.Series):
            if isinstance(index, Interval) or isinstance(index, LabeledInterval):
                iobject = IntervalSeries.IntervalSeries(index.to_list(), data, copy_series=False, copy_intervals=False)
            else:
                iobject = IntervalSeries.IntervalSeries(index, data, copy_series=False, copy_intervals=False)
        else:
            if isinstance(index, Interval) or isinstance(index, LabeledInterval):
                iobject = IntervalFrame.IntervalFrame(index.to_list(), data, copy_df=False, copy_intervals=False)
            else:
                iobject = IntervalFrame.IntervalFrame(index, data, copy_df=False, copy_intervals=False)
        
        return iobject
    

    def __setitem__(self, index, value):
        """
        Set index
        """

        if isinstance(self.iobject, IntervalFrame.IntervalFrame):
            self.iobject.df.iloc[index] = value
        else:
            self.iobject.series.iloc[index] = value
        
        
class Locator(object):
    """
    Label indexer

    :class:`~intervalframe.Locator` indexes intervals
    """
    
    def __init__(self, iobject):
        """
        Initialize Locator

        Parameters
        ----------
            iobject : IntervalFrame of IntervalSeries
                Intervals to index

        Returns
        -------
            None

        """
        
        # Record IntervalFrame or IntervalSeries
        self.iobject = iobject
        
        
    def __getitem__(self, args):
        """
        Get index
        """
        
        # Check args is of length two
        if len(args) != len(self.iobject.shape):
            raise IndexError("Wrong number of indexers.")

        # Get index
        if isinstance(self.iobject.index, LabeledIntervalArray):
            if isinstance(args[0], slice):
                if isinstance(self.iobject, IntervalSeries.IntervalSeries):
                    intervals = self.iobject.index[args]
                else:
                    intervals = self.iobject.index[args[0]]
                    index = np.arange(self.iobject.shape[0])[args[0]]
            else:
                if isinstance(self.iobject, IntervalSeries.IntervalSeries):
                    intervals, index = self.iobject.index.get_with_index(args)
                else:
                    intervals, index = self.iobject.index.get_with_index(args[0])
        else:
            raise IndexError("No labels in index.")
        
        # Create DataFrame or Series
        if isinstance(self.iobject, IntervalFrame.IntervalFrame):
            if isinstance(args, tuple) and isinstance(args[1], str):
                if isinstance(args[0], slice):
                    data = self.iobject.df.iloc[args[0],:].loc[:,args[1]].copy()
                else:
                    data = self.iobject.df.iloc[index,:].loc[:,args[1]].copy()
            elif isinstance(args, tuple) and np.issubdtype(type(args[1]), np.integer):
                data = self.iobject.df.iloc[index,:].iloc[:,args[1]].copy()
            else:
                data = self.iobject.df.iloc[index,:].loc[:,args[1]].copy()
        else:
            if isinstance(args, slice):
                data = self.iobject.series.iloc[args].copy()
            else:
                data = self.iobject.series.iloc[index].copy()
                          
        # Create IntervalFrame or IntervalSeries
        if isinstance(data, pd.Series):
            iobject = IntervalSeries.IntervalSeries(intervals, data, copy_series=False, copy_intervals=False)
        else:
            iobject = IntervalFrame.IntervalFrame(intervals, data, copy_df=False, copy_intervals=False)
        
        return iobject

    
    def __setitem__(self, index, value):
        """
        Set index
        """

        # Check if index has keys
        if isinstance(self.iobject, IntervalFrame.IntervalFrame):
            if isinstance(index, tuple) and len(index) == 2:
                if isinstance(index[0], slice):
                    self.iobject.df.loc[index] = value
                elif isinstance(self.iobject.index, LabeledIntervalArray):
                    indices = self.iobject.index.get_index(index[0])
                    self.iobject.df.loc[:,index[1]].values[indices] = value
        
        else:
            if isinstance(index, slice):
                self.iobject.series.loc[index] = value
            elif isinstance(self.iobject.index, LabeledIntervalArray):
                indices = self.iobject.index.get_index(index)
                self.iobject.series.loc[index].values[indices] = value