import os
import glob
import gzip
import numpy as np
import pandas as pd
from ailist import IntervalArray, LabeledIntervalArray
from intervalframe import IntervalFrame
import h5py


def read_bed(bed_file, header=None):
    """
    Read BED formated file
    """

    # Initialize intervals
    intervals = LabeledIntervalArray()
    
    # Determine if gzipped
    gzipped = False
    if bed_file.endswith(".gz"):
        gzipped = True
    
    # Determine number of fields
    if gzipped:
        o = gzip.open(bed_file,'r')
        n_fields = len(o.readline().strip().split(b"\t"))
    else:
        o = open(bed_file, "r")
        n_fields = len(o.readline().strip().split("\t"))
    o.close()

    # Determine if df present
    read_df = False
    if n_fields > 3:
        read_df = True
        cols = np.arange(n_fields - 3) + 3
    
    # Open file
    if gzipped:
        o = gzip.open(bed_file,'r')
    else:
        o = open(bed_file, "r")
    # Read intervals
    if header is not None:
        header_line = o.readline()
    for line in o:
        if gzipped:
            line = line.decode()
        fields = line.strip().split("\t")
        intervals.add(int(fields[1]), int(fields[2]), fields[0])
    o.close()
    
    # Read df
    df = None
    if read_df:
        df = pd.read_csv(bed_file, header=header, index_col=None, sep="\t", usecols=cols)

    # Create IntervalFrame
    iframe = IntervalFrame(intervals, df)

    return iframe