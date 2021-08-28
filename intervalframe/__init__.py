from __future__ import absolute_import
from .core.IntervalFrame import IntervalFrame
from .core.IntervalSeries import IntervalSeries
from .write import write_h5
from .read import read_h5

# This is extracted automatically by the top-level setup.py.
__version__ = '0.1.0'
__author__ = "Kyle S. Smith"


__doc__ = """\
API
======

Basic class
-----------

.. autosummary::
   :toctree: .
   
   IntervalFrame
   IntervalSeries
    
"""