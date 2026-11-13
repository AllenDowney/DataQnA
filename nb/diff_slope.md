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

# Testing a Difference in Slopes

Here's a [question from the Reddit statistics forum](https://www.reddit.com/r/statistics/comments/1cxxsyb/comment/l56k35f/?context=3).

> How do I determine if the difference in change-speed is significant?
> 
> Hi! I'm doing a project for uni and I'm in need of a statistical test but I'm not well read enough on statistics to figure out which.
>
> I have 7 years worth of anual data (average plant diversity, using Shannon Wiener) from two separate locations and I need to figure out if there is a significant difference between the two locations in the speed at which the diversity changes.
>
> Let me know if any additional information is needed to figure this out!

We can do this by resampling, but we'll need to think about how to model the data-generating process and the null hypothesis.
One of the best ways to model the data-generating process is to generate some random data.

<!-- #region tags=["remove-print"] -->
[Click here to run this notebook on Colab](https://colab.research.google.com/github/AllenDowney/DataQnA/blob/main/nb/diff_slope.ipynb).
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

I'll start with a range of 7 years.

```python
n = 7
xs = np.arange(n)
```

And generate fake datasets with different intercepts and slopes, but the same standard deviation of the error terms, `sigma`.

```python
np.random.seed(1)

sigma = 1
ys1 = 10 - 1 * xs + np.random.normal(0, sigma, size=n)
ys2 = 8 - 1.5 * xs + np.random.normal(0, sigma, size=n)
```

Here's what the data look like.

```python
plt.plot(xs, ys1, 'o')
plt.plot(xs, ys2, 'o')

decorate(xlabel='x', ylabel='y')
```

And here are the estimated slopes for the two datasets separately.

```python
from scipy.stats import linregress

res1 = linregress(xs, ys1)
res1.slope
```

```python
res2 = linregress(xs, ys2)
res2.slope
```

Here's the observed difference in the slopes.

```python
diff_actual = res1.slope - res2.slope
diff_actual
```

## The null hypothesis

Now, if the question is whether this difference might be due to chance, we have to define what we mean by "due to chance".
In this example, the null hypothesis is that the slope is the same, so the first question we have to answer is, "If the slope is the same, what is it?"

To answer that, we can use multiple regression with both datasets, constrained so they have the same slope, but allowing them to have different intercepts.
First I'll combine the datasets in a single `DataFrame`, with a column indicating which group each row is from.

```python
df1 = pd.DataFrame(dict(x=xs, y=ys1))
df1['group'] = 1
df1
```

```python
df2 = pd.DataFrame(dict(x=xs, y=ys2))
df2['group'] = 2
df2
```

```python
df = pd.concat([df1, df2]).reset_index(drop=True)
df
```

Now we can run the regression.

```python
import warnings

warnings.filterwarnings('ignore', message='kurtosistest*')
```

```python
import statsmodels.formula.api as smf

formula = 'y ~ x + C(group)'
result = smf.ols(formula, data=df).fit()
result.summary()
```

Under the assumption that the slope is the same in both groups, the estimated slope is about -1.2, and the intercepts for the groups are different.

We can use `predict` to generate fitted values for each group.

```python
fit1 = result.predict(df1)
fit2 = result.predict(df2)
```

And here's what they look like compared to the data.

```python
plt.plot(xs, ys1, 'o')
plt.plot(xs, fit1, color='C0')
plt.plot(xs, ys2, 'o')
plt.plot(xs, fit2, color='C1')

decorate(xlabel='x', ylabel='y')
```

## Simulating the null hypothesis

To compute a p-value, we have to compute the test statistic -- the difference in slopes -- for datasets generated under the null hypothesis.
We can do that by choosing random residuals and adding them to the fitted lines.

Here's what the residuals look like.

```python
resid1 = ys1 - fit1
resid2 = ys2 - fit2

plt.plot(xs, resid1, 'o')
plt.plot(xs, resid2, 'o')
decorate(xlabel='x', ylabel='Residual')
```

```python
resid = np.concatenate([resid1, resid2])
```

The following function generates random datasets under the null hypothesis and computes the difference in their slopes.

```python
def resample_slope():
    resampled1 = fit1 + np.random.choice(resid, size=n, replace=True)
    resampled2 = fit2 + np.random.choice(resid, size=n, replace=True)

    res1 = linregress(xs, resampled1)
    res2 = linregress(xs, resampled2)
    diff = res1.slope - res2.slope
    return diff
```

If we call this function once, we get a single instance of the test statistic under the null hypothesis -- this is the answer to the question, "If there were actually no difference in the slopes and we ran the experiment again, how big a difference might we see by chance?"

```python
np.random.seed(17)
resample_slope()
```

If we run it many times, we get a sample of the test statistic under the null hypothesis.

```python
np.random.seed(17)
diffs = [resample_slope() for i in range(1001)]
```

Here's what the distribution of the test statistic looks like.

```python
sns.kdeplot(diffs)
plt.axvline(diff_actual, color='0.7', ls=':')
decorate(xlabel='Difference in slopes', ylabel='Density')
```

The mean is near 0, as we expect if there were actually no difference in the slopes.
But the tails extend past the observed difference, which is near 0.5.
That means that a difference as big as 0.5 could plausibly happen by chance.

To quantify how plausible, we can compute the p-value, which is the fraction of the sample that exceeds the observed difference in magnitude.

```python
p_value = np.mean(np.abs(diffs) >= np.abs(diff_actual))
p_value
```

For the fake dataset I generated, the p-value is about 14%, which means that the observed difference could plausibly happen by chance, even if the slopes are actually the same in the two groups.

## The parametric version

The resampling method draws residuals from the ones we observed, so it makes no assumptions about their distribution.
Alternatively, we can assume that the residuals follow a normal distribution, and draw them from a normal distribution with the same mean and standard deviation as the observed residuals.

```python
m, s = np.mean(resid), np.std(resid)
```

Here's a version of `resample_slope` that uses the normal model.

```python
def resample_slope2():
    resampled1 = fit1 + np.random.normal(m, s, size=n)
    resampled2 = fit2 + np.random.normal(m, s, size=n)

    res1 = linregress(xs, resampled1)
    res2 = linregress(xs, resampled2)
    diff = res1.slope - res2.slope
    return diff
```

If we run it many times, we get another sample of the test statistic under the null hypothesis.

```python
np.random.seed(18)
diffs2 = [resample_slope2() for i in range(1001)]
```

Here's how the two distributions compare.

```python
sns.kdeplot(diffs, label='Resampled residuals')
sns.kdeplot(diffs2, label='Normal residuals')
plt.axvline(diff_actual, color='0.7', ls=':')
decorate(xlabel='Difference in slopes', ylabel='Density')
```

They are similar, so the p-values are similar, too.

```python
p_value2 = np.mean(np.abs(diffs2) >= np.abs(diff_actual))
p_value2
```

Finally, the same hypothesis can be tested with a regression model that includes an interaction term, which allows the slope to be different in the two groups.
The p-value for the interaction term tests whether the difference in slopes is statistically significant.

```python
result_inter = smf.ols('y ~ x * C(group)', data=df).fit()
result_inter.pvalues['x:C(group)[T.2]']
```

This p-value is bigger, about 21%, because the regression model is more conservative than the simulations.
The simulations use the observed residuals, which are a little smaller than the errors in the data-generating process, because the fitted lines are chosen to minimize them.
The regression model corrects for that, and it also accounts for the uncertainty in estimating the size of the errors from only 14 data points.
With such a small dataset, those corrections make a noticeable difference.
But the conclusion is the same: a difference in slopes this big could plausibly be due to chance.

## Discussion

To answer the question: if you have a time series of annual measurements from two locations, you can test whether the rate of change is different by estimating the slope at each location and asking whether the difference could plausibly be due to chance.
The simplest way to do that is a regression model with an interaction term, as in the last step.
The resampling methods ask the same question, but they make it explicit how we model the data-generating process and the null hypothesis.

With only 7 years of data, though, the test has low power.
In the fake data I generated, the slopes really are different -- by 0.5 per year -- but the difference we see in the data is not statistically significant.

So if the test with real data is not significant, that doesn't mean the rates of change are the same; it might mean there isn't enough data to tell.
It might be more informative to report the estimated difference in slopes with a confidence interval, which shows how big the difference could be.

Also, annual measurements from the same location might not be independent -- a good year might be followed by another good year -- which these tests don't take into account.


[*Data Q&A: Answering the real questions with Python*](https://allendowney.github.io/DataQnA/index.html)

Copyright 2024–2026 [Allen B. Downey](https://allendowney.com)

License: [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/)
