Benchmarking
============

.. code-block:: python

	import numpy as np
	import pandas as pd
	# version: 1.0.0
	from intervalframe import IntervalFrame
	# version: 0.0.120
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
	# CPU times: user 34.4 ms, sys: 1.26 ms, total: 35.6 ms
	# Wall time: 37.9 ms
	%time iframe1.df.loc[:,"values"] = values1
	# CPU times: user 1.42 ms, sys: 1.01 ms, total: 2.42 ms
	# Wall time: 3.25 ms

	%time iframe2 = IntervalFrame.from_array(starts2, ends2, labels=labels2)
	# CPU times: user 36.6 ms, sys: 1.35 ms, total: 37.9 ms
	# Wall time: 37.9 ms
	%time iframe2.df.loc[:,"values"] = values2
	# CPU times: user 1.29 ms, sys: 1.2 ms, total: 2.49 ms
	# Wall time: 1.51 ms

	%timeit iframe1.intersect(64182, 164184, label="a")
	# 1.8 ms ± 7.58 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
	%time o = iframe1.overlap(iframe2)
	# CPU times: user 32 s, sys: 15.1 s, total: 47.1 s
	# Wall time: 54.9 s


	#####################
	# pyranges ##########
	#####################

	%time gr1 = pr.from_dict({"Chromosome": labels1, "Start": starts1, "End": ends1, "ID": ids1, "values":values1})
	# CPU times: user 26 ms, sys: 6.48 ms, total: 32.5 ms
	# Wall time: 37.6 ms
	%time gr2 = pr.from_dict({"Chromosome": labels2, "Start": starts2, "End": ends2, "ID": ids2, "values":values2})
	# CPU times: user 25.4 ms, sys: 4.84 ms, total: 30.2 ms
	# Wall time: 29.2 ms

	%time gro = gr1.intersect(gr2)
	# CPU times: user 33.6 s, sys: 27.1 s, total: 1min
	# Wall time: 1min 8s


	#####################
	# pandas ############
	#####################

	%time pd_mi1 = pd.MultiIndex.from_arrays([labels1, pd.IntervalIndex.from_arrays(starts1, ends1)], names=["label", "interval"])
	# CPU times: user 513 ms, sys: 10.9 ms, total: 524 ms
	# Wall time: 527 ms
	%time pd_i1 = pd.DataFrame(values1, index=pd_mi1)
	# CPU times: user 193 µs, sys: 7 µs, total: 200 µs
	# Wall time: 204 µs

	%timeit pd_i1.loc["a"].loc[pd_i1.loc["a"].index.overlaps(pd.Interval(64182, 164184)),:]
	# 5.11 ms ± 74.6 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)