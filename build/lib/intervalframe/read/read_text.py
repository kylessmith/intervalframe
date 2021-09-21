import os
import glob
import numpy as np
import pandas as pd
from ailist import IntervalArray, LabeledIntervalArray
from intervalframe import IntervalFrame
import h5py


def read_bed(bed_file):
    """
    Read BED formated file
    """

    # Initialize intervals
    intervals = LabeledIntervalArray()

    # Determine number of fields
    o = open(bed_file, "r")
    n_fields = len(o.readline().strip().split("\n"))
    o.close()

    # Determine if df present
    read_df = False
    if n_fields > 3:
        read_df = True
        cols = np.arange(n_fields - 3) + 3
    
    # Read intervals
    o = open(bed_file, "r")
    header = o.readline()
    for line in o:
        fields = line.strip().split("\t")
        intervals.add(int(fields[1]), int(fields[2]), fields[0])
    o.close()
    
    # Read df
    df = None
    if read_df:
        df = pd.read_csv(bed_file, header=0, index_col=None, sep="\t", usecols=cols)

    # Create IntervalFrame
    iframe = IntervalFrame(intervals, df)

    return iframe