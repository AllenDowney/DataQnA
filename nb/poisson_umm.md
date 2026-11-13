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

# Title

Here's a [question from the Reddit statistics forum](https://www.reddit.com/r/statistics/comments/1cqg0d6/q_youtube_video_where_the_creator_attended_a/).

>  A while ago I watched a YouTube video where the creator told the story that he went to a science conference and he was bored so he started measuring the number of times and the intervals between when the speakers said “ehhh” or “emmm”. He discovered the mean was equal to the variance, and spent the latter part of the video explaining why he thought this was a Poisson process and what can be learnt from it.

> I can’t find it anywhere, I don’t remember the title or the name of the channel. Does anyone know? 

[It's here]()

Update the link...

<!-- #region tags=["remove-print"] -->
[Click here to run this notebook on Colab](https://colab.research.google.com/github/AllenDowney/DataQnA/blob/main/nb/poisson_umm.ipynb).
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

download('https://github.com/AllenDowney/DataQnA/raw/main/nb/utils.py');
```

```python tags=["hide-cell", "remove-print"]
# install the empiricaldist library, if necessary

try:
    import empiricaldist
except ImportError:
    !pip install empiricaldist
```

```python tags=["remove-print", "hide-cell"]
import numpy as np
np.set_printoptions(legacy='1.25')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import decorate
```

## Answer

```python
download('https://raw.githubusercontent.com/notDavidsGit/uhmmmsData/main/uhmms_publicVersion.csv')
```

```python
df = pd.read_csv('uhmms_publicVersion.csv')
professional = df.loc[0]
df = df.drop(index=0).replace(0, np.nan)
df
```

```python
summary = df.describe()
summary
```

```python
ms = summary.loc['mean']
ss = summary.loc['std']

high = ms.max()
plt.plot([0, high], [0, high], color='gray')
plt.plot(ms, ss, '.')
decorate()
```

```python
from empiricaldist import Pmf

def make_surv(seq):
    """Make a non-standard survival function, P(X>=x)"""
    pmf = Pmf.from_seq(seq)
    surv = pmf.make_surv() + pmf

    # correct for numerical error
    surv.iloc[0] = 1
    return Surv(surv)
```

```python
for col in df.columns:
    sf = make_surv(df[col])
    sf.plot(label='')
    
decorate(yscale='log')
```

```python
from scipy.stats import binom

def sf_error_bounds(dist, n, qs, con_level=0.95):
    """Find the bounds on a normal CDF analytically.

    dist: scipy.stats distribution object
    n: sample size
    qs: quantities
    alpha: fraction excluded from the CI

    returns: tuple of arrays (low, high)
    """
    # find the correct probabilities
    ps = dist.sf(qs)

    # find the upper and lower percentiles of
    # a binomial distribution
    p_low = (1 - con_level) / 2
    p_high = 1 - p_low

    low = binom.ppf(p_low, n, ps) / n
    low[ps == 1] = 1
    high = binom.ppf(p_high, n, ps) / n
    return low, high
```

```python
ts = df['1']
ts.describe()
```

```python
n = ts.count()
lam = ts.mean()
n, lam
```

```python
from scipy.stats import expon

dist = expon(scale=lam)
dist.mean()
```

```python
qs = np.linspace(0, ts.max())
low, high = sf_error_bounds(dist, n, qs)
plt.fill_between(qs, low, high, color='gray', alpha=0.3, lw=0)

sf = make_surv(ts)
sf.plot(label='')
    
decorate(yscale='log')
```

```python
def poisson_plot(ts):
    n = ts.count()
    lam = ts.mean()
    dist = expon(scale=lam)

    qs = np.linspace(0, ts.max())
    low, high = sf_error_bounds(dist, n, qs)
    plt.fill_between(qs, low, high, color='gray', alpha=0.3, lw=0)

    sf = make_surv(ts)
    sf.plot(label='')

    decorate(yscale='log')
```

```python
for col in df.columns:
    plt.figure()
    poisson_plot(df[col])
```

## Discussion


[*Data Q&A: Answering the real questions with Python*](https://allendowney.github.io/DataQnA/index.html)

Copyright 2024–2026 [Allen B. Downey](https://allendowney.com)

License: [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/)
