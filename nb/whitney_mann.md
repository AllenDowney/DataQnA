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

Here's a [question from the Reddit statistics forum](https://www.reddit.com/r/AskStatistics/comments/1ddgpqr/comment/l860puz/?context=3).

> 

Hey,

I am currently trying to calculate some independent t-tests for my thesis and could use some help testing the assumption of the data being normally distributed.

My initial plan was to check the distribution visually and run a Shapiro-wilk test (I am using spss if that makes a difference).

So far so good, however the results don’t show a clear picture (to me) and I am not experienced enough to know what to make of it.

After visual inspection I would have judged most of my data to not be normally distributed. I have attached some examples. However, for all of these examples pictured, the Shapiro-wilk test did not turn out significant. I was unsure whether that might be due to missing power (my sample sizes range from n= 16 to n = 36). Since I really am no expert and don’t really trust my judgment, I then used R to calculate qqplots with confidence intervals for those cases. That absolute majority of my data points lie within the confidence intervals, with very few exceptions directly on the boarder or outside (but very close) to it (e.g. one or two out of 30 data points lie outside but very close to the interval) So now I am thinking that my visual judgment might be of?

Just out of interest I calculated one t-test and one Whitney-Mann test for one of my research questions to compare the results. They went into the same direction, however they did differ a bit (p = .29 vs p = .14).

Now I really do not know how to proceed. I am grateful for any advice on how to go on and which test to choose 🙏


Intro

Update the link...
<!-- #endregion -->

<!-- #region tags=["remove-print"] -->
[Click here to run this notebook on Colab](https://colab.research.google.com/github/AllenDowney/DataQnA/blob/main/nb/whitney_mann.ipynb).
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
from scipy.stats import lognorm

def make_lognormal(m, s, loc=0):
    scale = np.exp(m)
    dist = lognorm(s=s, loc=loc, scale=scale)
    return dist
```

```python
dist1 = make_lognormal(0.5, 0.5)
dist1.mean(), dist1.std()
```

```python
dist2 = make_lognormal(0.75, 0.5)
dist2.mean(), dist2.std()
```

```python
np.random.seed(17)

n = 30
group1 = dist1.rvs(n)
group2 = dist2.rvs(n)
```

```python
from scipy.stats import norm
from empiricaldist import Cdf

def plot_normal(data, low, high, **options):
    mean = np.mean(data)
    std = np.std(data)

    xs = np.linspace(low, high)
    ps = norm(mean, std).cdf(xs)

    plt.plot(xs, ps, color='gray', alpha=0.5)
    Cdf.from_seq(data).plot(**options)
    decorate(xlabel='x', ylabel='Density')
```

```python
plot_normal(group1, -3, 8, label='group1')
```

```python
plot_normal(group2, -3, 8, label='group2')
```

```python
actual_diff = group2.mean() - group1.mean()
actual_diff
```

```python
import statsmodels.stats.weightstats as smw

t_statistic, p_value, degrees_of_freedom = smw.ttest_ind(group1, group2)
p_value
```

```python
from scipy.stats import mannwhitneyu

u_statistic, p_value = mannwhitneyu(group1, group2)
p_value
```

```python
pooled = np.concatenate([group1, group2])
n, m = len(group1), len(group2)
```

```python
def simulate_difference_mean():
    np.random.shuffle(pooled)
    shuffled1 = pooled[:n]
    shuffled2 = pooled[n:]
    diff = np.mean(shuffled1) - np.mean(shuffled2)
    return diff
```

```python
sample = [simulate_difference_mean() for i in range(1001)]
```

```python
sns.kdeplot(sample)
decorate()
```

```python
p_value = (np.abs(sample) > actual_diff).mean()
p_value
```

## When is it a problem?

```python
m1 = 2
m2 = 2.03
s = 0.3

np.random.seed(1)
group1 = norm(m1, s).rvs(1000)
group2 = norm(m2, s).rvs(1000)

plot_normal(group1, 1, 3, label='group1')
plot_normal(group2, 1, 3, label='group2')

_, p_value_ttest, _ = smw.ttest_ind(group1, group2)
_, p_value_mwu = mannwhitneyu(group1, group2)
print(p_value_ttest, p_value_mwu)
```

```python
s = 0.3

np.random.seed(1)
group1 = norm(2, s).rvs(n)
group2 = norm(2.1, s).rvs(n)

plot_normal(group1, 1, 3, label='group1')
plot_normal(group2, 1, 3, label='group2')

_, p_value_ttest, _ = smw.ttest_ind(group1, group2)
_, p_value_mwu = mannwhitneyu(group1, group2)
print(p_value_ttest, p_value_mwu)
```

```python
m1 = 0.5
m2 = 0.54
s = 0.1
seed = 1

np.random.seed(seed)
group1 = make_lognormal(0.5, s).rvs(n)
group2 = make_lognormal(0.54, s).rvs(n)
```

```python
plot_normal(group1, 1, 2.5, label='group1')
plot_normal(group2, 1, 2.5, label='group2')
```

```python
_, p_value_ttest, _ = smw.ttest_ind(group1, group2)
_, p_value_mwu = mannwhitneyu(group1, group2)
p_value_ttest, p_value_mwu
```

```python
def error_func(m2, s, seed=1, m1=0.5, target=0.05):
    np.random.seed(seed)
    group1 = make_lognormal(0.5, s).rvs(n)
    group2 = make_lognormal(m2, s).rvs(n)

    _, p_value_mwu = mannwhitneyu(group1, group2)
    return p_value_mwu - target
```

When `m2` is the same as `m1`, the p-value is large.

```python
error_func(0.5, s)
```

When `m2` is substantially larger than `m1`, the p-value is small.

```python
s = 0.1
error_func(2, s, seed)
```

For a given value of `s`, we can use a root-finder to search for the value of `m2` that makes the p-value 0.5.

```python
from scipy.optimize import brentq

s = 0.1
m2 = brentq(error_func, 0.5, 2, args=(s, seed))
m2, error_func(m2, s)
```

As we vary `s`, 

```python
def run_loop(seed=1):
    res = []
    m1 = 0.5

    for s in np.linspace(0.01, 2):
        m2 = brentq(error_func, 0.5, 3, args=(s, seed))

        np.random.seed(seed)
        group1 = make_lognormal(m1, s).rvs(n)
        group2 = make_lognormal(m2, s).rvs(n)

        _, p_value_ttest, _ = smw.ttest_ind(group1, group2)
        _, p_value_mwu = mannwhitneyu(group1, group2)
        res.append((s, m2, p_value_mwu, p_value_ttest))

    columns = ['s', 'm2', 'p_value_mwu', 'p_value_ttest']
    return pd.DataFrame(res, columns=columns)
```

```python
res_df = run_loop(2)
```

```python
plt.plot(res_df['s'], res_df['p_value_mwu'], color='gray', alpha=0.5, label='U-test')
plt.plot(res_df['s'], res_df['p_value_ttest'], label='t-test')
decorate(xlabel='s')
decorate(ylabel='p-value')
```

```python
def p_value_diff(group1, group2):
    _, p_value_ttest, _ = smw.ttest_ind(group1, group2)
    _, p_value_mwu = mannwhitneyu(group1, group2)
    
    return (p_value_ttest - p_value_mwu) / p_value_mwu
```

```python
p_value_diff(group1, group2)
```

```python
for s in np.linspace(0.001, 1):
    dist1 = make_lognormal(0.5, s)
    group1 = dist1.rvs(n)

    dist2 = make_lognormal(1, s)
    group2 = dist1.rvs(n)
    
    diff = p_value_diff(group1, group2)
    print(s, diff)
```

## Discussion


[*Data Q&A: Answering the real questions with Python*](https://allendowney.github.io/DataQnA/index.html)

Copyright 2024–2026 [Allen B. Downey](https://allendowney.com)

License: [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/)
