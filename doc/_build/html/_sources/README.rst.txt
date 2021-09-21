|Stars| |PyPIDownloads| |PyPI| |Build Status| |Coffee|

.. |Stars| image:: https://img.shields.io/github/stars/kylessmith/intervalframe?logo=GitHub&color=yellow
   :target: https://github.com/kylessmith/intervalframe/stargazers
.. |PyPIDownloads| image:: https://pepy.tech/badge/intervalframe
   :target: https://pepy.tech/project/intervalframe
.. |PyPI| image:: https://img.shields.io/pypi/v/intervalframe.svg
   :target: https://pypi.org/project/intervalframe
.. |Build Status| image:: https://travis-ci.org/kylessmith/intervalframe.svg?branch=master
   :target: https://travis-ci.org/kylessmith/intervalframe
.. |Coffee| image:: https://img.shields.io/badge/-buy_me_a%C2%A0coffee-gray?logo=buy-me-a-coffee&color=ff69b4
   :target: https://www.buymeacoffee.com/kylessmith

intervalframe – DataFrame with intervals as the index
=====================================================

The Python-based implementation efficiently deals with many intervals.

Benchmark
~~~~~~~~~

Test numpy random integers, see `benchmarking <benchmarking.html>`__

+--------------------+----------------+-----------+
| Library            | Function       | Time (ms) |
+====================+================+===========+
| intervalframe      | single overlap |      3.18 |
+--------------------+----------------+-----------+
| pandas             | single overlap |       7.4 |
+--------------------+----------------+-----------+

As of conducting these benchmarks, only ncls and ailist have bulk query functions.

+-----------+--------------+----------+-----------------+
| Library   | Function     | Time (s) | Max Memory (GB) |
+===========+==============+==========+=================+
| ncls      | bulk overlap | ~151     | >50             |
+-----------+--------------+----------+-----------------+
| ailist    | bulk overlap | ~17.9    | ~9              |
+-----------+--------------+----------+-----------------+

Querying intervals is much faster and more efficient with ailist