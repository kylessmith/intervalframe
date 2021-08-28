import numpy as np
import pandas as pd
from intervalframe import IntervalFrame
import pyranges as pr
from collections import Counter

# Set seed
np.random.seed(100)


# First values
starts1 = np.random.randint(0, 10000, 10000)
ends1 = starts1 + np.random.randint(1, 1000, 10000)
ids1 = np.arange(len(starts1))
values1 = np.ones(len(starts1))
df1 = pd.DataFrame(values1)
labels1 = np.repeat("a", len(starts1))
labels1[np.random.random(10000) > 0.75] = "b"
labels1[np.random.random(10000) > 0.75] = "c"
print(Counter(labels1))

# Second values
starts2 = np.random.randint(0, 10000, 10000)
ends2 = starts2 + np.random.randint(1, 1000, 10000)
ids2 = np.arange(len(starts2))
values2 = np.ones(len(starts2))
labels2 = np.repeat("a", len(starts2))
labels2[np.random.random(10000) > 0.75] = "b"
labels2[np.random.random(10000) > 0.75] = "c"
print(Counter(labels2))

####################
# IntervalFrame ####
####################

iframe1 = IntervalFrame.from_array(starts1, ends1, labels=labels1)
iframe1.df.loc[:,"values"] = values1

iframe2 = IntervalFrame.from_array(starts2, ends2, labels=labels2)
iframe2.df.loc[:,"values"] = values2

#o = iframe1.intersect(6418, 16418, label="a")
# 1.31 ms ± 7.03 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
#o = iframe1.overlap(iframe2)
oi = iframe1.index.intersect_from_LabeledIntervalArray(iframe2.index)
a = iframe1.index[oi[0]]
df = iframe1.df.iloc[oi[0],:]
a.construct()
#o = IntervalFrame(a, df=df)
#%time o = iframe1.overlap(iframe2)
# CPU times: user 18.9 s, sys: 14.3 s, total: 33.2 s
# Wall time: 40.4 s
# Max ~14GB, after ~7.5GB
#o.index.construct()
# CPU times: user 45.3 s, sys: 17.9 s, total: 1min 3s
# Wall time: 1min 11s
# Max ~15GB, after ~15GB

#####################
# pyranges ##########
#####################

#gr1 = pr.from_dict({"Chromosome": labels1, "Start": starts1, "End": ends1, "ID": ids1, "values":values1})
#gr2 = pr.from_dict({"Chromosome": labels2, "Start": starts2, "End": ends2, "ID": ids2, "values":values2})

#gro = gr1.intersect(gr2)
# CPU times: user 56 s, sys: 50.2 s, total: 1min 46s
# Wall time: 1min 55s
# Max ~25GB, after ~12.5GB


#####################
# pandas ############
#####################

#pd_mi1 = pd.MultiIndex.from_arrays([labels1, pd.IntervalIndex.from_arrays(starts1, ends1)], names=["label", "interval"])
#pd_i1 = pd.DataFrame(values1, index=pd_mi1)

#o = pd_i1.loc["a"].loc[pd_i1.loc["a"].index.overlaps(pd.Interval(6418,16418)),:]
#%timeit pd_i1.loc["a"].loc[pd_i1.loc["a"].index.overlaps(pd.Interval(6418,16418)),:]
# 7.47 ms ± 217 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
