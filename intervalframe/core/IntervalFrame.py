from ailist import IntervalArray, LabeledIntervalArray
import pandas as pd
import numpy as np
import copy as cp
from .index.indexers import iLocator, Locator
from tabulate import tabulate
from bcpseg import bcpseg
import cbseg
from collections import Counter


#_mpl_repr
class IntervalFrame(object):
    """
    Annotated augmented interval list

    :class:`~intervalframe.IntervalFrame` stores a intervals
    """

    def __init__(self, intervals=None, df=None, labels=None, columns=None, dtypes=None,
                 copy_df=False, copy_intervals=False):
        """
        Initialize IntervalFrame

        Parameters
        ----------
            intervals : AIList
                Intervals to be stored
            df : pandas.DataFrame
                DataFrame to annotate intervals with
            labels : array-like
                Labels for hierarchical indexing
            columns :  array-like
                Columns for DataFrame
            dtypes : dict
                Dtype for DataFrame columns
            copy_df : bool
                Whether to copy DataFrame
            copy_intervals : bool
                Whether to copy AIList

        Returns
        -------
            None

        """

        # Determine if intervals need to be copied
        if copy_intervals and intervals is not None:
            intervals = intervals.copy()
        
        # Initialize Index
        if isinstance(intervals, IntervalArray) or isinstance(intervals, LabeledIntervalArray):
            self.index = intervals
        elif intervals is None and df is None:
            self.index = None
        elif intervals is None and df.shape == (0,0):
            self.index = None
        else:
            raise TypeError("Unrecognized input for intervals.")

        # Initialize DataFrame
        if df is None:
            # Intervals given
            if intervals is None:
                # Columns given
                if columns is None:
                    df = pd.DataFrame([], index=range(0))
                else:
                    if dtypes is None:
                        df = pd.DataFrame([], index=range(0), columns=columns)
                    else:
                        df = pd.DataFrame([], index=range(0), columns=columns).astype(dtypes, copy=False)
            else:
                # Columns given
                if columns is None:
                    df = pd.DataFrame([], index=range(len(self.index)))
                else:
                    if dtypes is None:
                        df = pd.DataFrame([], index=range(len(self.index)), columns=columns)
                    else:
                        df = pd.DataFrame([], index=range(len(self.index)), columns=columns).astype(dtypes, copy=False)
        
        elif isinstance(df, pd.DataFrame):
            if copy_df:
                df = df.copy(deep=True)
                
        # Set df
        self.df = df
        
        # Check shape
        if self.index is None:
            if self.df.shape[0] != 0:
                raise TypeError("DataFrame and Intervals don't match.")
        elif self.index.size != self.df.shape[0]:
            raise TypeError("DataFrame and Intervals don't match.")

        # Make sure index is frozen
        if self.index is not None:
            self.index.freeze()


    def __repr__(self):
        """
        IntervalFrame representation
        """

        # Initialized string
        repr_string = ""

        # If nothing
        if self.index is None:
            repr_string += "None\n"

        # If no columns present
        elif self.df.shape[1] == 0:
            repr_string += repr(self.index)
            repr_string += repr(self.df)

        # If columns present
        else:
            # Determine dimensions
            n_rows = min(self.df.shape[0], 5)
            n_cols = min(self.df.shape[1], 5)
            
            # Initialize values
            repr_list = [[] for i in range(n_rows+1)]
            
            # Determine column names
            repr_list[0] = ["interval"]
            repr_list[0] += list(map(str, list(self.df.columns.values[:n_cols])))
            
            i = 0 # track rows
            for interval in self.index:
                if i >= n_rows:
                    break
                #repr_list[i+1].append(repr(interval).split(",")[0] + ")")
                repr_list[i+1].append(repr(interval))
                repr_list[i+1] += list(map(str, list(self.df.iloc[i,:n_cols].values)))
                i += 1
            
            # Create tabulate table
            repr_string = tabulate(repr_list, headers="firstrow")

        return repr_string

    
    @property
    def shape(self):
        """
        """
        
        return self.df.shape


    @property
    def iloc(self):
        """
        """

        # If index is None
        if self.index is None:
            return None

        return iLocator(self)

    
    @property
    def loc(self):
        """
        """

        # If index is None
        if self.index is None:
            return None

        return Locator(self)
        
    
    @property
    def values(self):
        """
        """

        return self.df.values

    
    @property
    def columns(self):
        """
        """

        return self.df.columns

    
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
            
        # Create IntervalFrame
        iframe = IntervalFrame(index)
        
        return iframe
                
                
    def starts(self):
        """
        """

        # If index is None
        if self.index is None:
            return None
        
        # Extract starts in intervals
        starts = self.index.extract_starts()
        
        return starts
        
    
    def ends(self):
        """
        """

        # If index is None
        if self.index is None:
            return None
        
        # Extract ends in intervals
        ends = self.index.extract_ends()
        
        return ends


    def concat(self, iframe_iter):
        """
        Append IntervalFrame to current IntervalFrame

        Parameters
        ----------
            ilist : IntervalFrame

        Returns
        -------
            None

        """

        # Iterate over iframes
        concat_iframe = self.copy()
        if concat_iframe.index is not None:
            concat_iframe.index.unfreeze()
        for iframe in iframe_iter:
            concat_iframe.df = pd.concat([concat_iframe.df, iframe.df], ignore_index=True, join="outer")
            if concat_iframe.index is None:
                concat_iframe.index = iframe.index.copy()
            else:
                concat_iframe.index.append(iframe.index)
        concat_iframe.index.construct()
        concat_iframe.index.freeze()

        # Reset index
        concat_iframe.df.index = range(concat_iframe.df.shape[0])

        return concat_iframe


    def intersect(self, start, end, label=None):
        """
        Find intersecting intervals

        Paramters
        ---------
            start : int
                Starting position
            end : int
                Ending position
            label : str or list
                Labels to intersect with [default: None]

        Returns
        -------
            overlaps : IntervalFrame
                Overlapping intervals
            
        """

        # If index is None
        if self.index is None:
            raise AttributeError("LabeledIntervalArray is empty.")

        # Intersect
        if label is None:
            overlaps, overlap_index = self.index.intersect_with_index(start, end)
        else:
            overlaps, overlap_index = self.index.intersect_with_index(start, end, label=str(label))
            
        # Create df
        if self.df.shape[1] > 0:
            df = pd.DataFrame(self.df.values[overlap_index,:],
                            columns=self.df.columns.values).astype(self.df.dtypes.to_dict(), copy=False)
        else:
            df = pd.DataFrame([], index=range(len(overlaps)))

        # Create IntervalFrame
        overlaps = IntervalFrame(overlaps, df, copy_intervals=False, copy_df=False)

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

        # If index is None
        if self.index is None:
            raise AttributeError("LabeledIntervalArray is empty.")

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

        # If index is None
        if self.index is None:
            raise AttributeError("LabeledIntervalArray is empty.")

        pass


    def nhits(self, start, end, label=None):
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

        # If index is None
        if self.index is None:
            raise AttributeError("LabeledIntervalArray is empty.")

        if label is None:
            nhits = self.index.nhits(start, end)
        else:
            nhits = self.index.nhits(start, end, label)

        return nhits


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

        # If index is None
        if self.index is None:
            raise AttributeError("LabeledIntervalArray is empty.")

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

        # If index is None
        if self.index is None:
            raise AttributeError("LabeledIntervalArray is empty.")

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

        # If index is None
        if self.index is None:
            raise AttributeError("LabeledIntervalArray is empty.")

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

        # If index is None
        if self.index is None:
            raise AttributeError("LabeledIntervalArray is empty.")

        # Determine nhits
        nhits_bins, nhits_index = self.index.bin_nhits(bin_size, min_length, max_length)

        # Construct IntervalFrame
        nhits_df = pd.DataFrame(nhits_index, index=range(len(nhits_index)), columns=["nhits"])
        nhits_iframe = IntervalFrame(nhits_bins, nhits_df, copy_intervals=False,
                                     copy_df=False)

        return nhits_iframe
    

    def annotate(self, iframe, column, method="mean"):
        """
        Annotate values in one IntervalFrame with another

        Parameters
        ----------
            iframe : IntervalFrame
                Intervals used to annotate
            columns : str
                Name of the column to use
            method : str
                How to annotate ('mean','std','median','label')
        
        Returns
        -------
            None

        """

        # If index is None
        if self.index is None:
            raise AttributeError("LabeledIntervalArray is empty.")

        # Find overlaps
        if isinstance(iframe.index, LabeledIntervalArray) and isinstance(self.index, LabeledIntervalArray):
            query_index, ref_index = self.index.intersect_from_LabeledIntervalArray(iframe.index)
        elif isinstance(iframe.index, IntervalArray) and isinstance(self.index, IntervalArray):
            query_index, ref_index = self.index.intersect_from_IntervalArray(iframe.index)
        else:
            raise TypeError("IntervalFrames must have same type of index.")

        # Set function
        if method == "mean":
            func = np.mean
        elif method == "median":
            func = np.median
        elif method == "std":
            func = np.std
        elif method == "var":
            func = np.var

        # Calculate value
        if method != "label":
            values = np.zeros(self.df.shape[0])
            # Iterate over overlaps
            for i in pd.unique(ref_index):
                values[i] = func(iframe.df.loc[query_index[ref_index == i],
                                               column].values)

        elif method == "label":
            values = np.repeat("", dtype="S10")
            # Iterate over overlaps
            for i in pd.unique(ref_index):
                values[i] = Counter(iframe.df.loc[query_index[ref_index == i],
                                                  column].values).most_common(1)[0][0]

        # Append column to df
        self.df[method] = values


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

        # If index is None
        if self.index is None:
            raise AttributeError("LabeledIntervalArray is empty.")

        # Find overlaps
        if isinstance(iframe.index, LabeledIntervalArray) and isinstance(self.index, LabeledIntervalArray):
            query_index, ref_index = self.index.intersect_from_LabeledIntervalArray(iframe.index)
        elif isinstance(iframe.index, IntervalArray) and isinstance(self.index, IntervalArray):
            query_index, ref_index = self.index.intersect_from_IntervalArray(iframe.index)
        else:
            raise TypeError("IntervalFrames must have same type of index.")

        # Index iframes
        if ref_index.shape[0] > 0: 
            results_iframe = self.iloc[ref_index,:]
            results_iframe.df["overlap"] = query_index
        else:
            results_iframe = IntervalFrame()

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

        # If index is None
        if self.index is None:
            raise AttributeError("LabeledIntervalArray is empty.")

        # Merge intervals
        merged_index = self.index.merge(gap=gap)
        
        # Create IntervalFrame
        merged_iframe = IntervalFrame(merged_index)

        return merged_iframe

    
    def segment(self, column, method="bcp_online", cutoff=0.5, hazard=100, shuffles=5000, p=0.00005):
        """
        Annotate values in one IntervalFrame with another

        Parameters
        ----------
            iframe : IntervalFrame
                Intervals used to annotate
            column : str
                Name of the column to use
        
        Returns
        -------
            segment_iframe : IntervalFrame
                Segmented Intervals

        """

        # If index is None
        if self.index is None:
            raise AttributeError("LabeledIntervalArray is empty.")

        # Segment intervals for IntervalArray
        if method == "bcp_online":
            if isinstance(self.index, IntervalArray):
                segment_intervals = bcpseg(self.df[column].values, cutoff=cutoff, method="online", hazard=hazard)
            else:
                segment_intervals = bcpseg(self.df[column].values, labels=self.index.extract_labels(), cutoff=cutoff, method="online", hazard=hazard)
        elif method == "bcp_online_both":
            if isinstance(self.index, IntervalArray):
                segment_intervals = bcpseg(self.df[column].values, cutoff=cutoff, method="online_both", hazard=hazard)
            else:
                segment_intervals = bcpseg(self.df[column].values, labels=self.index.extract_labels(), cutoff=cutoff, method="online_both", hazard=hazard)
        elif method == "bcp_offline":
            if isinstance(self.index, IntervalArray):
                segment_intervals = bcpseg(self.df[column].values, cutoff=cutoff, method="offline", hazard=hazard)
            else:
                segment_intervals = bcpseg(self.df[column].values, labels=self.index.extract_labels(), cutoff=cutoff, method="offline", hazard=hazard)
        elif method == "cbs":
            if isinstance(self.index, IntervalArray):
                segment_intervals = cbseg.segment(self.df[column].values, shuffles=shuffles, p=p)
            else:
                segment_intervals = cbseg.segment(self.df[column].values, labels=self.index.extract_labels(), shuffles=shuffles, p=p)
            segment_intervals = cbseg.validate(self.df[column].values, segment_intervals, shuffles=shuffles, p=p)
        else:
            raise NameError("method input not recognized.")

        #Re-index segments
        #print(segment_intervals)
        segment_intervals.index_with_aiarray(self.index)

        # Create IntervalFrame
        segment_iframe = IntervalFrame(segment_intervals)

        return segment_iframe
        
        
    def downsample(self, proportion):
        """
        Randomly downsample intervals

        Parameters
        ----------
            proportion : float
                Proportion of intervals to keep

        Returns
        -------
            filtered_iframe : IntervalFrame
                Downsampled values
        """

        # If index is None
        if self.index is None:
            raise AttributeError("LabeledIntervalArray is empty.")
        
        # Downsample
        filtered_intervals, filtered_index = self.index.downsample_with_index(proportion)
        
        # Create df
        if self.df.shape[1] > 0:
            df = pd.DataFrame(self.df.values[filtered_index,:],
                            columns=self.df.columns.values).astype(self.df.dtypes.to_dict(), copy=False)
        else:
            df = pd.DataFrame([], index=range(len(filtered_intervals)))

        # Create IntervalFrame
        filtered_iframe = IntervalFrame(filtered_intervals, df, copy_intervals=False, copy_df=False)
            
        return filtered_iframe
        
        
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

        # If index is None
        if self.index is None:
            raise AttributeError("LabeledIntervalArray is empty.")
        
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

        # If index is None
        if self.index is None:
            raise AttributeError("LabeledIntervalArray is empty.")
        
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
            matched_iframe : LabeledIntervalArray
                Matched intervals

        """

        # If index is None
        if self.index is None:
            raise AttributeError("LabeledIntervalArray is empty.")

        # Determine matches
        filtered_intervals, filtered_index = self.index.filter_exact_match(iframe.index)

        # Create df
        if self.df.shape[1] > 0:
            df = pd.DataFrame(self.df.values[filtered_index,:],
                            columns=self.df.columns.values).astype(self.df.dtypes.to_dict(), copy=False)
        else:
            df = pd.DataFrame([], index=range(len(filtered_intervals)))

        # Create IntervalFrame
        filtered_iframe = IntervalFrame(filtered_intervals, df, copy_intervals=False, copy_df=False)

        return filtered_iframe
        
    
    def copy(self):
        """
        Copy IntervalFrame
        
        Parameters
        ----------
            None
        
        Returns
        -------
            copy_iframe : IntervalFrame
                Copied intervals
        
        """

        # If index is None
        #if self.index is None:
            #raise AttributeError("LabeledIntervalArray is empty.")
        
        # Make copy of IntervalFrame
        copy_iframe = IntervalFrame(self.index, self.df,
                                    copy_intervals=True,
                                    copy_df=True)
        
        return copy_iframe
        