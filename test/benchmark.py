import numpy as np
import pandas as pd
from intervalframe import IntervalFrame
import pyranges as pr
from collections import Counter

# Set seed
np.random.seed(100)


# First values
starts1 = np.random.randint(0, 100000, 100000)
ends1 = starts1 + np.random.randint(1, 10000, 100000)
ids1 = np.arange(len(starts1))
values1 = np.ones(len(starts1))
df1 = pd.DataFrame(values1)
labels1 = np.repeat("a", len(starts1))
labels1[np.random.random(100000) > 0.75] = "b"
labels1[np.random.random(100000) > 0.75] = "c"

# Second values
starts2 = np.random.randint(0, 100000, 100000)
ends2 = starts2 + np.random.randint(1, 10000, 100000)
ids2 = np.arange(len(starts2))
values2 = np.ones(len(starts2))
labels2 = np.repeat("a", len(starts2))
labels2[np.random.random(100000) > 0.75] = "b"
labels2[np.random.random(100000) > 0.75] = "c"

####################
# IntervalFrame ####
####################

%time iframe1 = IntervalFrame.from_array(starts1, ends1, labels=labels1)
# CPU times: user 37.7 ms, sys: 1.16 ms, total: 38.8 ms
# Wall time: 39.3 ms
%time iframe1.df.loc[:,"values"] = values1
# CPU times: user 1.34 ms, sys: 355 µs, total: 1.69 ms
# Wall time: 1.33 ms

%time iframe2 = IntervalFrame.from_array(starts2, ends2, labels=labels2)
# CPU times: user 43.1 ms, sys: 1.08 ms, total: 44.2 ms
# Wall time: 44.1 ms
%time iframe2.df.loc[:,"values"] = values2
# CPU times: user 1.38 ms, sys: 680 µs, total: 2.06 ms
# Wall time: 1.45 ms

%timeit iframe1.intersect(64182, 164184, label="a")
# 3.18 ms ± 183 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
%timeit o = iframe1.overlap(iframe2)
# 1min 54s ± 6.25 s per loop (mean ± std. dev. of 7 runs, 1 loop each)


#####################
# pyranges ##########
#####################

%time gr1 = pr.from_dict({"Chromosome": labels1, "Start": starts1, "End": ends1, "ID": ids1, "values":values1})
# CPU times: user 38.5 ms, sys: 3.57 ms, total: 42.1 ms
# Wall time: 41.1 ms
%time gr2 = pr.from_dict({"Chromosome": labels2, "Start": starts2, "End": ends2, "ID": ids2, "values":values2})
# CPU times: user 39.4 ms, sys: 3.65 ms, total: 43.1 ms
# Wall time: 42 ms

%time gro = gr1.intersect(gr2)
# CPU times: user 57.8 s, sys: 55.9 s, total: 1min 53s
# Wall time: 2min 10s


#####################
# pandas ############
#####################

%time pd_mi1 = pd.MultiIndex.from_arrays([labels1, pd.IntervalIndex.from_arrays(starts1, ends1)], names=["label", "interval"])
# CPU times: user 1.16 s, sys: 15.3 ms, total: 1.17 s
# Wall time: 1.18 s
%time pd_i1 = pd.DataFrame(values1, index=pd_mi1)
# CPU times: user 193 µs, sys: 0 ns, total: 193 µs
# Wall time: 196 µs

%timeit pd_i1.loc["a"].loc[pd_i1.loc["a"].index.overlaps(pd.Interval(64182, 164184)),:]
# 7.4 ms ± 405 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)