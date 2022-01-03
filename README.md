# IntervalFrame for interval manipulation

[![Build Status](https://travis-ci.org/kylessmith/intervalframe.svg?branch=master)](https://travis-ci.org/kylessmith/intervalframe) [![PyPI version](https://badge.fury.io/py/intervalframe.svg)](https://badge.fury.io/py/intervalframe)
[![Coffee](https://img.shields.io/badge/-buy_me_a%C2%A0coffee-gray?logo=buy-me-a-coffee&color=ff69b4)](https://www.buymeacoffee.com/kylessmith)

This is a Python package for easy and efficient manipulation
of Next Generation Sequencing reads.


## Install

If you dont already have numpy and scipy installed, it is best to download
`Anaconda`, a python distribution that has them included.  
```
    https://continuum.io/downloads
```

Dependencies can be installed by:

```
    pip install -r requirements.txt
```

PyPI install, presuming you have all its requirements installed:
```
	pip install intervalframe
```

## Usage

```python
from intervalframe import IntervalFrame
import numpy as np

# Create data
iframe = IntervalFrame()
```
