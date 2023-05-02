Tutorial
=========

.. code-block:: python

	import numpy as np
	import pandas as pd
	# version: 1.0.0
	from intervalframe import IntervalFrame
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

	iframe1 = IntervalFrame.from_array(starts1, ends1, labels=labels1)
	iframe1.df.loc[:,"values"] = values1

	iframe2 = IntervalFrame.from_array(starts2, ends2, labels=labels2)
	iframe2.df.loc[:,"values"] = values2

	o = iframe1.intersect(64182, 164184, label="a")
	o = iframe1.overlap(iframe2)

