from ailist import IntervalArray, LabeledIntervalArray
import pandas as pd
import numpy as np
import copy as cp
from .index import indexers
from tabulate import tabulate
from bcpseg import bcpseg
import cbseg
from collections import Counter
from . import IntervalFrame


#_mpl_repr
class IntervalSeries(object):
    """
    Annotated augmented interval list

    :class:`~intervalframe.IntervalFrame` stores a intervals
    """

    def __init__(self, intervals, series=None, labels=None, dtype=None,
                 copy_series=False, copy_intervals=False):
        """
        Initialize IntervalFrame

        Parameters
        ----------
            intervals : AIList
                Intervals to be stored
            sereis : pandas.Series
                DataFrame to annotate intervals with
            labels : array-like
                Labels for hierarchical indexing
            dtype : 
                Dtype of series
            copy_sereies : bool
                Whether to copy Series
            copy_intervals : bool
                Whether to copy AIList

        Returns
        -------
            None

        """

        # Determine if intervals need to be copied
        if copy_intervals:
            intervals = cp.copy(intervals)
        
        # Initialize Index
        if isinstance(intervals, IntervalArray) or isinstance(intervals, LabeledIntervalArray):
            self.index = intervals
        else:
            raise TypeError("Unrecognized input for intervals.")

        # Initialize Series
        if series is None:
            # Intervals given
            if intervals is None:
                if dtype is None:
                    series = pd.Series([], index=range(0))
                else:
                    series = pd.Series([], index=range(0)).astype(dtype, copy=False)
            else:
                if dtype is None:
                    series = pd.DataFrame([], index=range(len(self.index)))
                else:
                    series = pd.DataFrame([], index=range(len(self.index))).astype(dtype, copy=False)
        
        elif isinstance(series, pd.Series):
            if copy_series:
                series = series.copy(deep=True)
                
        # Set series
        self.series = series

        # Make sure index is frozen
        self.index.freeze()


    def __repr__(self):
        """
        IntervalFrame representation
        """

        # Initialized string
        repr_string = ""

        # If no columns present
        if self.series.shape[0] == 0:
            repr_string += repr(self.index)
            repr_string += repr(self.series)

        # If columns present
        else:
            # Determine dimensions
            n_rows = min(self.series.shape[0], 5)
            
            # Initialize values
            repr_list = [[] for i in range(n_rows+1)]
            
            # Determine column names
            repr_list[0] = ["interval"]
            repr_list[0] += [str(self.series.name)]
            
            i = 0 # track rows
            for interval in self.index:
                if i >= n_rows:
                    break
                #repr_list[i+1].append(repr(interval).split(",")[0] + ")")
                repr_list[i+1].append(repr(interval))
                repr_list[i+1] += [str(self.series.values[i])]
                i += 1
            
            # Create tabulate table
            repr_string = tabulate(repr_list, headers="firstrow")

        return repr_string

    
    @property
    def shape(self):
        """
        """
        
        return self.series.shape


    @property
    def iloc(self):
        """
        """

        return indexers.iLocator(self)

    
    @property
    def loc(self):
        """
        """

        return indexers.Locator(self)
        
    
    @property
    def values(self):
        """
        """
        
        return self.series.values

    
    @staticmethod
    def from_array(starts, ends, labels=None):
        """
        """

        # Add intervals
        if labels is None:
            index = IntervalArray()
            index.from_array(starts, ends)
        else:
            index = LabeledIntervalArray()
            index.from_array(starts, ends, labels)
            
        # Create IntervalSeries
        iseries = IntervalSeries(index)
        
        return iseries
                
                
    def starts(self):
        """
        """
        
        # Extract starts in intervals
        starts = self.index.extract_starts()
        
        return starts
        
    
    def ends(self):
        """
        """
        
        # Extract ends in intervals
        ends = self.index.extract_ends()
        
        return ends


    def intersect(self, start, end, label=None):
        """
        Find intersecting intervals

        Paramters
        ---------
            start : int
                Starting position
            end : int
                Ending position
            label : str
                Label to intersect with [default: None]

        Returns
        -------
            overlaps : IntervalFrame
                Overlapping intervals
            
        """

        # Intersect
        if label is None:
            overlaps, overlap_index = self.index.intersect_with_index(start, end)
        else:
            overlaps, overlap_index = self.index.intersect_with_index(start, end, label=str(label))
            
        # Create df
        if len(self.series) > 0:
            series = pd.Series(self.series.values[overlap_index])
        else:
            series = pd.Series([], index=range(len(overlaps)))

        # Create IntervalFrame
        overlaps = IntervalSeries(overlaps, series, copy_intervals=False, copy_series=False)

        return overlaps


    def subtract(self, iframe):
        """
        Subtract intersecting regions for IntervalFrame

        Parameters
        ----------
            iframe : IntervalFrame
                Intervals to remove

        Returns
        -------
            subtracted_frame : IntervalFrame
                Intervals with removed regions

        """

        pass


    def difference(self, iframe):
        """
        Remove any overlapping intervals

        Parameters
        ----------
            iframe : IntervalFrame
                Intervals to remove

        Returns
        -------
            diff_frame : IntervalFrame
                Intervals that do not overlap
        """

        pass


    def nhits(self, start, end):
        """
        Find number of intersecting intervals

        Paramters
        ---------
            start : int
                Starting position
            end : int
                Ending position

        Returns
        -------
            n : int
                Number of overlapping intervals
            
        """

        pass


    def nhits_from_array(self, starts, ends):
        """
        Find number of intersecting intervals

        Paramters
        ---------
            start : numpy.ndarray
                Starting position
            end : numpoy.ndarray
                Ending position

        Returns
        -------
            n : numpy.ndarray
                Number of overlapping intervals
            
        """

        pass


    def coverage(self):
        """
        Find number of intervals overlapping every
        position in the IntervalFrame

        Parameters
        ----------
            None

        Returns
        -------
            pandas.Series{double}
                Position on index and coverage as values
        """

        pass


    def bin_coverage(self, bin_size=100000, min_length=None, max_length=None):
        """
        Find sum of coverage within binned
        positions

        Parameters
        ----------
            bin_size : int
                Size of the bin to use
            min_length : int
                Minimum length of intervals to include [default = None]
            max_length : int
                Maximum length of intervals to include [default = None]

        Returns
        -------
            cov_iframe : IntervalFrame
                Positions of coverage values

        """

        # Determine nhits
        #ailist_cov = self.intervals.bin_coverage(bin_size=bin_size, min_length=min_length, max_length=max_length)

        # Create AIList from pd.Series index
        #positions = AIList()
        #positions.from_array(ailist_cov.index.values,
                            #ailist_cov.index.values + bin_size,
                            #np.arange(len(ailist_cov)),
                            #ailist_cov.values)

        # Construct IntervalFrame
        #cov_iframe = IntervalFrame(intervals=positions, df=ailist_cov.to_frame())

        #return cov_iframe
        pass


    def bin_nhits(self, bin_size=100000, min_length=None, max_length=None):
        """
        Find number of intervals overlapping binned
        positions

        Parameters
        ----------
            bin_size : int
                Size of the bin to use
            min_length : int
                Minimum length of intervals to include [default = None]
            max_length : int
                Maximum length of intervals to include [default = None]

        Returns
        -------
            nhits_iframe : IntervalFrame
                Position of nhits values

        """

        # Determine nhits
        nhits_bins, nhits_index = self.index.bin_nhits(bin_size, min_length, max_length)

        # Construct IntervalFrame
        nhits_df = pd.DataFrame(nhits_index, index=range(len(nhits_index)), columns=["nhits"])
        nhits_iframe = IntervalSeries(nhits_bins, nhits_df, copy_intervals=False,
                                     copy_df=False)

        return nhits_iframe


    def overlap(self, iframe, key="overlap"):
        """
        Find overlaps in one IntervalFrame with another

        Parameters
        ----------
            iframe : IntervalFrame
                Intervals used to annotate
            key : str
                Name of the column to use in resulting IntervalFrame
        
        Returns
        -------
            results_iframe : IntervalFrame
                Intervals with column indicating index of overlap

        """

        # Find overlaps
        if isinstance(iframe.index, LabeledIntervalArray) and isinstance(self.index, LabeledIntervalArray):
            query_index, ref_index = self.index.intersect_from_LabeledIntervalArray(iframe.index)
        elif isinstance(iframe.index, IntervalArray) and isinstance(self.index, IntervalArray):
            query_index, ref_index = self.index.intersect_from_IntervalArray(iframe.index)
        else:
            raise TypeError("IntervalFrames must have same type of index.")

        # Index iframes
        results_iframe = IntervalFrame.IntervalFrame(self.index[ref_index], copy_intervals=False)
        results_iframe.df["overlap"] = query_index

        #Create intervals
        #results_intervals = self.index[ref_index]

        # Create df
        #if self.df.shape[1] > 0:
            #df = pd.DataFrame(self.df.values[ref_index,:],
                              #columns=self.df.columns.values).astype(self.df.dtypes.to_dict(), copy=False)
            #df = self.df.iloc[ref_index,:].reset_index(drop=True, inplace=True)
        #else:
            #df = pd.DataFrame([], index=range(len(results_intervals)))

        # Create IntervalFrame
        #results_iframe = IntervalFrame(results_intervals, df, copy_intervals=False, copy_df=False)
        #results_iframe.df["overlap"] = query_index

        return results_iframe


    def merge(self, gap=0):
        """
        Annotate values in one IntervalFrame with another

        Parameters
        ----------
            gap : int
                Allowed gap between intervals [default: 0]
        
        Returns
        -------
            merged_iframe : IntervalFrame
                Merged intervals

        """

        # Merge intervals
        merged_index = self.index.merge(gap=gap)
        
        # Create IntervalFrame
        merged_iframe = IntervalSeries(merged_index, copy_intervals=False)

        return merged_iframe

    
    def segment(self, method="bcp_online", cutoff=0.5, hazard=100, shuffles=5000, p=0.00005):
        """
        Annotate values in one IntervalFrame with another

        Parameters
        ----------
            method : str
                Method for segmenting intervals
            cutoff : float (default = 0.5)
                Cutoff for bcp methods
            hazard : int (default = 100)
                Hazard values for bcp methods
            shuffles : int (default = 5000)
                Number of shuffles for cbs method
            p : float (default = 0.00005)
                Pvalue cutoff for cbs method
        
        Returns
        -------
            segment_iseries : IntervalSeries
                Segmented Intervals

        """

        # Segment intervals for IntervalArray
        if method == "bcp_online":
            if isinstance(self.index, IntervalArray):
                segment_intervals = bcpseg(self.series.values, cutoff=cutoff, method="online", hazard=hazard)
            else:
                segment_intervals = bcpseg(self.series.values, labels=self.index.extract_labels(), cutoff=cutoff, method="online", hazard=hazard)
        elif method == "bcp_online_both":
            if isinstance(self.index, IntervalArray):
                segment_intervals = bcpseg(self.series.values, cutoff=cutoff, method="online_both", hazard=hazard)
            else:
                segment_intervals = bcpseg(self.dseries.values, labels=self.index.extract_labels(), cutoff=cutoff, method="online_both", hazard=hazard)
        elif method == "bcp_offline":
            if isinstance(self.index, IntervalArray):
                segment_intervals = bcpseg(self.series.values, cutoff=cutoff, method="offline", hazard=hazard)
            else:
                segment_intervals = bcpseg(self.series.values, labels=self.index.extract_labels(), cutoff=cutoff, method="offline", hazard=hazard)
        elif method == "cbs":
            if isinstance(self.index, IntervalArray):
                segment_intervals = cbseg.segment(self.series.values, shuffles=shuffles, p=p)
            else:
                segment_intervals = cbseg.segment(self.series.values, labels=self.index.extract_labels(), shuffles=shuffles, p=p)
            segment_intervals = cbseg.validate(self.series.values, segment_intervals, shuffles=shuffles, p=p)
        else:
            raise NameError("method input not recognized.")

        #Re-index segments
        print(segment_intervals)
        segment_intervals.index_with_aiarray(self.index)

        # Create IntervalSeries
        segment_iseries = IntervalSeries(segment_intervals)

        return segment_iseries
        
        
    def downsample(self, proportion):
        """
        Randomly downsample intervals

        Parameters
        ----------
            proportion : float
                Proportion of intervals to keep

        Returns
        -------
            filtered_iseries : IntervalSeries
                Downsampled values
        """
        
        # Downsample
        filtered_intervals, filtered_index = self.index.downsample_with_index(proportion)
        
        # Create series
        series = pd.Series([], index=range(len(filtered_intervals)))

        # Create IntervalSeries
        filtered_iseries = IntervalSeries(filtered_intervals, series, copy_intervals=False, copy_series=False)
            
        return filtered_iseries
        
        
    def length_dist(self):
        """
        Calculate length distribution of intervals
        
        Parameters
        ----------
            None
        
        Returns
        -------
            length_distribution : numpy.ndarray
                Length distribution
        """
        
        # Calculate length distribution
        length_distribution = self.index.length_dist()
        
        return length_distribution
        
        
    def wps(self, protection=60, min_length=None, max_length=None):
        """
        Calculate Window Protection Score for each position in AIList range
        
        Parameters
        ----------
            protection : int
                Protection window to use
            min_length : int
                Minimum length of intervals to include [default = None]
            max_length : int
                Maximum length of intervals to include [default = None]
            label : str
                Label for hierarchical indexing
            
        Returns
        -------
            scores : dict of pandas.Series or pandas.Series
                Position on index and WPS as values
        
        """
        
        # Calculate WPS without label
        scores = self.index.wps(protection, min_length, max_length)
            
        return scores


    def exact_match(self, iframe):
        """
        Find exact matches between LabeledIntervalArrays

        Parameters
        ----------
            iframe : LabeledIntervalArray
                Intervals to match

        Returns
        -------
            filtered_iseries : IntervalSeries
                Matched intervals

        """

        # Determine matches
        filtered_intervals, filtered_index = self.index.filter_exact_match(iframe.index)

        # Create series
        series = pd.Series([], index=range(len(filtered_intervals)))

        # Create IntervalSeries
        filtered_iseries = IntervalSeries(filtered_intervals, series, copy_intervals=False, copy_series=False)

        return filtered_iseries
        
    
    def copy(self):
        """
        Copy IntervalSeries
        
        Parameters
        ----------
            None
        
        Returns
        -------
            copy_iseries : IntervalSeries
                Copied intervals
        
        """
        
        # Make copy of IntervalSeries
        copy_iseries = IntervalSeries(self.index, self.series,
                                    copy_intervals=True,
                                    copy_series=True)
        
        return copy_iseries
        