from intervalframe import IntervalFrame
from ailist import AIList
import numpy as np
import pandas as pd
import pyranges as pr
# load example datasets
exons, cpg = pr.data.exons(), pr.data.cpg()

i = IntervalFrame.from_array(exons.df.loc[:,"Start"].values.astype(int),
                             exons.df.loc[:,"End"].values.astype(int),
                             labels = exons.df.loc[:,"Chromosome"].values)

i2 = IntervalFrame.from_array(cpg.df.loc[:,"Start"].values.astype(int),
                              cpg.df.loc[:,"End"].values.astype(int),
                              labels = cpg.df.loc[:,"Chromosome"].values)

#o = i.overlap(i2)

i.df.loc[:,"values"] = np.random.random(1000)
i.df.loc[:,"values"].values[100:200] = i.df.loc[:,"values"].values[100:200] + 2
s = i.segment("values")

i2.annotate(i, "values", "mean")
bh = i.bin_nhits()

pd_mi2 = pd.MultiIndex.from_arrays([cpg.df.loc[:,"Chromosome"].values, pd.IntervalIndex.from_arrays(cpg.df.loc[:,"Start"].values, cpg.df.loc[:,"End"].values)], names=["chrom", "interval"])
pd_i2 = pd.DataFrame(i2.df.values, index=pd_mi2)

%timeit pd_i2.loc["chrX"].loc[pd_i2.loc["chrX"].index.overlaps(pd.Interval(64182,164184)),:]
# 1.35 ms ± 17.5 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
%timeit i2.intersect(64182, 164184, label="chrX")
# 153 µs ± 1.05 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)


%timeit exons.intersect(cpg)
%timeit i.overlap(i2)

gr = pr.from_dict({"Chromosome": ["chr1"] * 3, "Start": [1, 4, 10], "End": [3, 9, 11], "ID": ["a", "b", "c"]})
gr2 = pr.from_dict({"Chromosome": ["chr1"] * 3, "Start": [2, 2, 9], "End": [3, 9, 10]})

i = IntervalFrame()
i.from_array(gr.df.loc[:,"Start"].values.astype(int),
             gr.df.loc[:,"End"].values.astype(int),
             labels = gr.df.loc[:,"Chromosome"].values)

i2 = IntervalFrame()
i2.from_array(gr2.df.loc[:,"Start"].values.astype(int),
             gr2.df.loc[:,"End"].values.astype(int),
             labels = gr2.df.loc[:,"Chromosome"].values)

o = i.overlap(i2)
o2 = gr.intersect(gr2)