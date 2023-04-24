from ailist import IntervalArray, LabeledIntervalArray
import pandas as pd
import numpy as np
import copy as cp
from .index.indexers import iLocator, Locator
from tabulate import tabulate
import linear_segment
from collections import Counter
from typing import Any, Dict, List, Optional, Hashable, Sequence
from pandas.api.types import is_sparse


#_mpl_repr
class IntervalFrame(object):
    """
    Annotated augmented interval list

    :class:`~intervalframe.IntervalFrame` stores a intervals
    """

    def __init__(self, 
                 intervals: Optional[LabeledIntervalArray] = None,
                 df: Optional[pd.DataFrame] = None,
                 labels: Optional[np.ndarray] = None,
                 columns: Optional[np.ndarray] = None,
                 dtypes: Optional[Dict] = None,
                 copy_df: bool = False,
                 copy_intervals: bool = False):
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

            # Add shape
            repr_string += "\n\n[" + str(self.shape[0]) + " rows x " + str(self.shape[1]) + " columns]\n"

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
            index.add(starts, ends)
        else:
            index = LabeledIntervalArray()
            index.add(starts, ends, labels)
            
        # Create IntervalFrame
        iframe = IntervalFrame(index)
        
        return iframe
    

    @staticmethod
    def from_dict_range(dict_range, bin_size=10000):
        """
        """

        # Add intervals
        index = LabeledIntervalArray.create_bin(dict_range, bin_size)

        # Create IntervalFrame
        iframe = IntervalFrame(index)

        return iframe


    @staticmethod
    def read_h5(h5_group):
        """
        Read IntervalFrame to h5py group

        Parameters
        ----------
            h5_group : h5py.Group
                Section of h5 file to read from

        Returns
        -------
            iframe : IntervalFrame
                IntervalFrame from h5 file

        """

        from ..read.read_h5 import read_h5_intervalframe
        
        # Write
        if len(h5_group) > 0:
            iframe = read_h5_intervalframe(h5_group)
        else:
            iframe = IntervalFrame()

        return iframe

    
    @staticmethod
    def read_bed(filename: str,
                 header: int = None,
                 skipfirst: bool = False,
                 is_url: bool = False):
        """
         Read BED file into IntervalFrame

        Parameters
        ----------
            filename : str
                Bed to read from
            header : int
                Which line to use as header
            skipfirst : bool
                Skip first line
            is_url : bool
                Whether file is url

        Returns
        -------
            iframe : IntervalFrame
                IntervalFrame from h5 file
        """

        from ..read.read_text import read_bed

        # Read bed
        iframe = read_bed(filename,
                          header=header,
                          skipfirst=skipfirst,
                          is_url=is_url)

        return iframe

    
    @staticmethod
    def read_parquet(filename: str):
        """
         Read parquet file into IntervalFrame

        Parameters
        ----------
            filename : str
                Parquet to read from

        Returns
        -------
            iframe : IntervalFrame
                IntervalFrame from parquet file
        """

        from ..read.read_parquet import read_parquet

        # Read bed
        iframe = read_parquet(filename)

        return iframe

                
    def starts(self):
        """
        """

        # If index is None
        if self.index is None:
            return None
        
        # Extract starts in intervals
        starts = self.index.starts
        
        return starts
        
    
    def ends(self):
        """
        """

        # If index is None
        if self.index is None:
            return None
        
        # Extract ends in intervals
        ends = self.index.ends
        
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
            overlaps, overlap_index = self.index.intersect(start, end,
                                                            return_intervals=True,
                                                            return_index=True)
        else:
            overlaps, overlap_index = self.index.intersect(start, end,
                                                           label=str(label),
                                                           return_intervals=True,
                                                           return_index=True)
            
        # Create df
        if self.df.shape[1] > 0:
            df = pd.DataFrame(self.df.values[overlap_index,:],
                            columns=self.df.columns.values).astype(self.df.dtypes.to_dict(),
                            copy=False)
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


    def nhits(self,
              start: int | np.ndarray,
              end: int | np.ndarray,
              label=None):
        """
        Find number of intersecting intervals

        Paramters
        ---------
            start : int | np.ndarray
                Starting position
            end : int | np.ndarray
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


    def bin_coverage(self,
                     bin_size: int = 100000,
                     min_length: int = None,
                     max_length: int = None):
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


    def bin_nhits(self,
                  bin_size: int = 100000,
                  min_length: int = None,
                  max_length: int = None):
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


    def iter_intersect(self,
                       iframe):
        """
        """

        for index in iframe.index.iter_intersect(self.index, return_intervals=False, return_index=True):
            if len(index) > 0:
                new_iframe = iframe.iloc[index,:]
            else:
                new_iframe = IntervalFrame(columns=[iframe.columns])
            
            yield new_iframe

    
    def annotate(self,
                 iframe,
                 column: str,
                 method: str = "mean",
                 column_name: str = None):
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
            column_name : str
                Name of the new column [defualt: method]
        
        Returns
        -------
            None

        """

        # If index is None
        if self.index is None:
            raise AttributeError("LabeledIntervalArray is empty.")

        # Check column_name
        if column_name is None:
            column_name = method

        # Set function
        if method == "mean":
            func = np.mean
        elif method == "median":
            func = np.median
        elif method == "std":
            func = np.std
        elif method == "var":
            func = np.var
        elif method == "sum":
            func = np.sum
        elif method == "label":
            func = lambda a : Counter(a).most_common(1)[0][0]

        if method == "label":
            values = np.zeros(self.df.shape[0]).astype(object)
        else:
            values = np.zeros(self.df.shape[0])
        # Check if sparse
        sparse_present = False
        if is_sparse(iframe.df.loc[:,column].values):
            sparse_present = True
            df_values = iframe.df.loc[:,column].to_dense().values
        for i, index in enumerate(iframe.index.iter_intersect(self.index, return_intervals=False, return_index=True)):
            if len(index) > 0:
                if sparse_present:
                    value = func(df_values[index])
                else:
                    value = iframe.df.loc[:,column].values[index]
                    values[i] = func(value)
            else:
                values[i] = np.nan

        # Append column to df
        self.df[column_name] = values


    def overlap(self,
                iframe,
                key: str = "overlap"):
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
            query_index, ref_index = self.index.intersect_from_LabeledIntervalArray(iframe.index,
                                                                                    return_intervals=False,
                                                                                    return_index=True)
 
        elif isinstance(iframe.index, IntervalArray) and isinstance(self.index, IntervalArray):
            query_index, ref_index = self.index.intersect_from_IntervalArray(iframe.index,
                                                                            return_intervals=False,
                                                                            return_index=True)
        else:
            raise TypeError("IntervalFrames must have same type of index.")

        # Index iframes
        if ref_index.shape[0] > 0:
            results_iframe = self.iloc[ref_index,:]
            results_iframe.df["overlap"] = query_index
        else:
            results_iframe = IntervalFrame()

        return results_iframe


    def merge(self,
              gap: int = 0):
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

    
    def segment(self,
                column: str,
                method: str = "online_both",
                cutoff: float = 0.5,
                hazard: int = 100,
                shuffles: int = 5000,
                p: float = 0.00005):
        """
        Annotate values in one IntervalFrame with another

        Parameters
        ----------
            iframe : IntervalFrame
                Intervals used to annotate
            column : str
                Name of the column to use
            method : str
                How to annotate ('bcp_online','bcp_online_both','cbs')
            cutoff : float
                Cutoff for bcpseg
            hazard : int
                Hazard for bcpseg
            shuffles : int
                Number of shuffles for cbs
            p : float
                P-value for cbs
        
        Returns
        -------
            segment_iframe : IntervalFrame
                Segmented Intervals

        """

        # If index is None
        if self.index is None:
            raise AttributeError("LabeledIntervalArray is empty.")

        # Segment intervals for IntervalArray
        if isinstance(self.index, IntervalArray):
            segment_intervals = linear_segment.segment(self.df[column].values,
                                                        labels = None,
                                                        cutoff = cutoff,
                                                        method = method,
                                                        hazard = hazard,
                                                        shuffles = shuffles,
                                                        p = p)
        else:
            segment_intervals = linear_segment.segment(self.df[column].values,
                                                        labels = self.index.labels,
                                                        cutoff = cutoff,
                                                        method = method,
                                                        hazard = hazard,
                                                        shuffles = shuffles,
                                                        p = p)

        # Re-index segments
        segment_intervals.index_with_aiarray(self.index)

        # Create IntervalFrame
        segment_iframe = IntervalFrame(segment_intervals)

        return segment_iframe
        
        
    def downsample(self,
                   proportion: float):
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
        filtered_intervals, filtered_index = self.index.downsample(proportion,
                                                                   return_intervals=True,
                                                                   return_index=True)
        
        # Create df
        if self.df.shape[1] > 0:
            df = pd.DataFrame(self.df.values[filtered_index,:],
                              columns=self.df.columns.values).astype(self.df.dtypes.to_dict(),
                              copy=False)
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
        
        
    def wps(self,
            protection: int = 60,
            min_length: int = None,
            max_length: int = None):
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


    def exact_match(self,
                    iframe):
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


    def combine(self,
                iframe_list: List,
                combine_method: str = "union",
                sparse_optimize: bool = False,
                fill_value: Any = np.nan):
        """
        Combine two IntervalFrames

        Parameters
        ----------
            iframe_list : List[IntervalFrame]
                IntervalFrame to combine with
            combine_method : str
                Method to combine intervals (default = "union")
            sparse_optimize : bool
                Optimize for sparse IntervalFrames (default = False)

        Returns
        -------
            combined_iframe : IntervalFrame
                Combined IntervalFrame

        See Also
        --------
            IntervalFrame.union
            IntervalFrame.intersect
            IntervalFrame.subtract

        Examples
        --------
        >>>from ailist import LabeledIntervalArray
        >>>from intervalframe import IntervalFrame
        >>>ail = LabeledIntervalArray()
        >>>ail.add(0, 10, "a")
        >>>ail.add(20, 30, "a")
        >>>ail.add(40, 50, "b")
        >>>ail.add(60, 70, "b")
        >>>ail2 = LabeledIntervalArray()
        >>>ail2.add(0, 10, "a")
        >>>ail2.add(20, 30, "b")
        >>>ail2.add(20, 30, "a")
        >>>is_match1, is_match2 = ail.is_exact_match(ail2)
        >>>iframe1 = IntervalFrame(intervals=ail)
        >>>iframe1.df.loc[:,"test"] = [1,2,3,4]
        >>>iframe2 = IntervalFrame(intervals=ail2)
        >>>iframe2.df.loc[:,"test"] = [8,9,10]
        >>>iframe2.df.loc[:,"test2"] = ['z','b','d']
        >>>iframe3 = iframe1.combine([iframe2])
        >>>iframe3
        interval              test  test2
        ------------------  ------  -------
        Interval(0-10, a)        1  nan
        Interval(20-30, a)       2  nan
        Interval(40-50, b)       3  nan
        Interval(60-70, b)       4  nan
        Interval(20-30, b)       9  b

        [5 rows x 2 columns]
        """

        # If index is None
        if self.index is None:
            raise AttributeError("LabeledIntervalArray is empty.")

        # Combine
        if combine_method == "union":
            for i, iframe in enumerate(iframe_list):
                if i == 0:
                    has_match1, has_match2 = self.index.is_exact_match(iframe.index)
                    intervals = self.index[has_match1]
                    intervals.append(self.index[~has_match1])
                    intervals.append(iframe.index[~has_match2])

                    # Format DataFrames
                    #common_columns = self.df.columns.intersection(iframe.columns).values
                    #other_columns1 = self.df.columns.difference(iframe.columns).values
                    other_columns2 = iframe.df.columns.difference(self.columns).values
                    ind1, ind2 = self.index.which_exact_match(iframe.index)
                    common_df = self.df.iloc[ind1,:].copy()
                    for col in other_columns2:
                        common_df[col] = np.nan
                    common_df[other_columns2] = iframe.df.loc[:,other_columns2].values[ind2,:]
                    # Construct new df
                    df = pd.concat([common_df, self.df.loc[~has_match1,:], iframe.df.loc[~has_match2]], axis=0)
                else:
                    has_match1, has_match2 = intervals.is_exact_match(iframe.index)
                    intervals.append(iframe.index[~has_match2])

                    # Format DataFrames
                    #common_columns = df.columns.intersection(iframe.columns).values
                    #other_columns1 = df.columns.difference(iframe.columns).values
                    other_columns2 = iframe.df.columns.difference(df.columns).values
                    ind1, ind2 = intervals.which_exact_match(iframe.index)
                    common_df = df.iloc[ind1,:].copy()
                    for col in other_columns2:
                        common_df[col] = np.nan
                    common_df[other_columns2] = iframe.df.loc[:,other_columns2].values[ind2,:]
                    # Construct new df
                    df = pd.concat([common_df, df.loc[~has_match1,:], iframe.df.loc[~has_match2]], axis=0)

                if sparse_optimize:
                    for col in df.columns.values:
                        if np.sum(df.loc[:,col].values == fill_value) > 0.5 * df.shape[0]:
                            self.df[col] = pd.arrays.SparseArry(self.df.loc[:,col].values, fill_value=fill_value)
        else:
            raise NameError("combine_method not recognized.")

        # Create IntervalFrame
        combined_iframe = IntervalFrame(intervals=intervals,
                                        df=df)

        return combined_iframe


    def sparse_optimize(self,
                        fill_value: Any = np.nan):
        """
        Optimize for sparse IntervalFrames
        
        Parameters
        ----------
            None : None
                None

        Returns
        -------
            None : None
                None

        See Also
        --------
            IntervalFrame.annotate
            IntervalFrame.combine

        Examples
        --------
        >>>from ailist import LabeledIntervalArray
        """

        # If index is None
        if self.index is None:
            raise AttributeError("LabeledIntervalArray is empty.")

        # Optimize
        for col in self.columns:
            if np.sum(self.df.loc[:,col].values == fill_value) > 0.5 * self.df.shape[0]:
                self.df[col] = pd.arrays.SparseArry(self.df.loc[:,col].values, fill_value=fill_value)

        return None


    def to_h5(self, 
              h5_group,
              compression_opts = 4):
        """
        Write IntervalFrame to h5py group

        Parameters
        ----------
            h5_group : h5py.Group
                Section of h5 file to write to
            compression_opt : int
                Gzip level of compression (default = 4)

        Returns
        -------
            None : None
                None

        """

        from ..write.write_h5 import write_h5_intervalframe
        
        # Write
        write_h5_intervalframe(self, h5_group, compression_opts=compression_opts)


    def to_parquet(self,
                   filename,
                   **kwargs):
        """
        Write IntervalFrame to parquet file

        Parameters
        ----------
            filename : str
                Name of parquet file
            kwargs : dict
                Keyword arguments for pyarrow.parquet.write_table

        Returns
        -------
            None : None
                None

        """

        from ..write.write_parquet import write_parquet
        
        # Write
        write_parquet(self, filename, **kwargs)


    def drop_columns(self,
                     columns: Hashable | Sequence[Hashable] | pd.Index):
        """
        Drop columns from IntervalFrame

        Parameters
        ----------
            columns : Hashable | Sequence[Hashable] | pd.Index
                Column(s) to drop

        Returns
        -------
            None : None
        """

        self.df = self.df.drop(columns, axis=1)
        
    
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
        
        # Make copy of IntervalFrame
        copy_iframe = IntervalFrame(self.index, self.df,
                                    copy_intervals=True,
                                    copy_df=True)
        
        return copy_iframe
        