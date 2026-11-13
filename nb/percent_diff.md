---
jupyter:
  jupytext:
    formats: ipynb,md
    split_at_heading: true
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.6
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region -->
# Title

Here's a [question from the Reddit statistics forum](https://www.reddit.com/r/AskStatistics/comments/1czl93x/percentage_significance_formula/).

> I can't remember which formula to use. I'm not doing this for homework, I'm just a researcher with brain fog.
>
> I'm comparing percentages and I can't remember which formula to use. Google was inconsistent.

> I am comparing percentages. ie one group of people got answers right 50% of the time and this other group got answers 40% of the time. I'm trying to find the best way to crunch the numbers to see if it's significant or nah.


Intro

Update the link...
<!-- #endregion -->

<!-- #region tags=["remove-print"] -->
[Click here to run this notebook on Colab](https://colab.research.google.com/github/AllenDowney/DataQnA/blob/main/nb/XXX.ipynb).
<!-- #endregion -->

<!-- #region tags=["remove-print"] -->
I'll download a utilities module with some of my frequently-used functions, and then import the usual libraries.
<!-- #endregion -->

```python tags=["remove-print", "hide-cell"]
from os.path import basename, exists

def download(url):
    filename = basename(url)
    if not exists(filename):
        from urllib.request import urlretrieve

        local, _ = urlretrieve(url, filename)
        print("Downloaded " + str(local))
    return filename

download('https://github.com/AllenDowney/DataQnA/raw/main/nb/utils.py')

import numpy as np
np.set_printoptions(legacy='1.25')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import decorate
```

```python tags=["hide-cell", "remove-print"]
# install the empiricaldist library, if necessary

try:
    import empiricaldist
except ImportError:
    !pip install empiricaldist
```

## Answer

```python
n = 100
k1 = 40
k2 = 50
```

```python
def percent_diff(k1, k2, n):
    p1 = k1 / n * 100
    p2 = k2 / n * 100
    return p2 - p1
```

```python
actual_diff = percent_diff(k1, k2, n)
actual_diff
```

```python
p = (k1 + k2) / (n + n)
p
```

```python
from scipy.stats import binom

def simulate_percent_diff(n, p):
    k1 = binom.rvs(n, p)
    k2 = binom.rvs(n, p)
    return percent_diff(k1, k2, n)
```

```python
simulate_percent_diff(n, p)
```

```python
sample = [simulate_percent_diff(n, p) for i in range(1001)]
```

```python
sns.kdeplot(sample)
```

```python
p_value = np.mean(np.abs(sample) > actual_diff)
p_value
```

```python
ks1 = binom.rvs(n, p, size=1000001)
ks2 = binom.rvs(n, p, size=1000001)
sample = percent_diff(ks1, ks2, n)
np.mean(np.abs(sample) > actual_diff)
```

```python
import statsmodels.api as sm

ks = [40, 50]
ns = [100, 100]

stat, pval = sm.stats.proportions_ztest(ks, ns)
pval
```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

## Discussion


[*Data Q&A: Answering the real questions with Python*](https://allendowney.github.io/DataQnA/index.html)

Copyright 2024 [Allen B. Downey](https://allendowney.com)

License: [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/)
