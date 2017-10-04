## DETAILED DESCRIPTIONS OF DATA FILES

Here are brief descriptions of the data.

- u.data.csv

The full u data set, 100000 ratings by 943 users on 1682 items.

Transformed from ``u.base`` file, Users and items are numbered consecutively from ``0``. This is a comma separated list of  ``user_id, movie_id, month_group, rating``. Considering the rating data starts from September 1997 and ends in April 1998,the ``month_group`` ranges from ``0`` to ``7``.

This data file can be generated from ``u.data`` by  ``MovieLens_data_make.py`` .



- u1.base.csv
- u1.test.csv
- u2.base.csv
- u2.test.csv
- u3.base.csv
- u3.test.csv
- u4.base.csv
- u4.test.csv
- u5.base.csv
- u5.test.csv

 The data sets ``u1.base.csv`` and ``u1.test.csv`` through ``u5.base.csv`` and ``u5.test.csv``  are 80%/20% splits of the u data into training and test data. These files are transformed from ``u1.base`` and ``u1.test`` through ``u5.base`` and ``u5.test``  by   ``MovieLens_data_make.py`` .



- MovieLens_data_make.py

A Python script to transform  ``*.data``  file, ``*.base`` files and  ``*.test`` files to ``*.csv`` files.

