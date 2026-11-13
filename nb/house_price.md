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

# Heteroskedasticity

Here's a [question from the Reddit statistics forum](https://www.reddit.com/r/statistics/comments/1ckl864/q_how_does_conditional_heteroskedasticity/).

> How does conditional heteroskedasticity underestimate standard errors?
>
> Shouldn't it depend on the data set? What if, by increasing the independent variable X, the variance in my residuals is actually increasing? Would that not mean the standard error increases, thereby REDUCING the t-stat and increasing the risk of type 2 error instead of type 1 error?

Heteroskedasticity does not always cause a linear regression model to underestimate standard errors -- it is also possible to overestimate.
I'll construct one example of each type, so we can see how.

And then we can address two broader questions:

1. When do we have to worry about heteroskedasticity,

2. And what can we do about it?

<!-- #region tags=["remove-print"] -->
[Click here to run this notebook on Colab](https://colab.research.google.com/github/AllenDowney/DataQnA/blob/main/nb/heterosked.ipynb).
<!-- #endregion -->

<!-- #region tags=["remove-print"] -->
I'll download a utilities module with some of my frequently-used functions, and then import the usual libraries.
<!-- #endregion -->

```python tags=["remove-print"]
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

```python tags=["hide-cell"]
# install the empiricaldist library, if necessary

try:
    import empiricaldist
except ImportError:
    !pip install empiricaldist
```

## What is heteroskedasticity?

Linear regression is based on a model of the data-generating process where the independent variable $y$ is the sum of

* A linear function of $x$ with an unknown slope and intercept, and

* Random values drawn from a Gaussian distribution with mean $\mu=0$ and unknown standard deviation, $\sigma$, that does not depend on $x$.

So what happens if we violate these assumptions?

* If $y$ is a non-linear function of $x$, regression finds the best linear fit to the data, which might not be very good.

* If $\mu$ is not zero and it does not depend on $x$, the estimate of the intercept will be biased.

* If $\mu$ is not zero and it *does* depend on $x$, the estimate of the slope will be biased, and probably the intercept, too.

* If $\sigma$ depends on $x$, the data-generating process is heteroskedastic. In general, heteroskedasticity doesn't bias the slope or the intercept, but it might bias the standard error of the estimated parameters.

To see why, let's generate some data.
First, I'll draw `xs` from a normal distribution.

```python
np.random.seed(17)

xs = np.random.normal(10, 1, size=500)
xs.sort()
```

If `sigma` is constant, the model is homoskedastic.

```python
sigma = 1.3
ys = 0.5 * xs + np.random.normal(0, sigma, size=len(xs))

homosked = pd.DataFrame(dict(x=xs, y=ys))
homosked.corr()
```

Now, to generate heteroskedastic data, I'll make use `interpolate` to construct a function where `sigma` depends on `x`.

```python
from scipy.interpolate import interp1d

def interpolate(xs, sigma_seq):
    return interp1d([6, 10, 14], sigma_seq)(xs)
```

For the first example, `sigma` increases linearly with `x`.

```python
sigmas = interpolate(xs, [0.0, 3.0, 6.0])
#sigmas = interpolate(xs, [0.2, 1.8, 0.2])
np.mean(sigmas)
```

Here's what that function looks like.

```python
plt.plot(xs, sigmas, '.')

decorate(xlabel='x',
         ylabel='sigma')
```

Now we can generate `ys` with variable values of `sigma`.

```python
ys = 0.5 * xs + np.random.normal(0, sigmas)
```

If we make a scatter plot of the data, we see the cone shape that is indicative of heteroskedasticity.

```python
plt.plot(xs, ys, '.')

decorate(xlabel='x', ylabel='y')
```

The variability on the right is wider than on the left.
If we compute a linear regression, the estimated parameters are close to the true values, `slope = 0.5` and `intercept = 0`.
And the standard error seems plausible.

```python
from scipy.stats import linregress

res = linregress(xs, ys)
res._asdict()
```

Here's the data again with the fitted line.

```python
fys = res.intercept + res.slope * xs

plt.plot(xs, ys, '.')
plt.plot(xs, fys)

decorate(xlabel='x', ylabel='y')
```

If we plot the absolute values of the residuals, we can see the heteroskedasticity more clearly.

```python
resid = ys - fys
plt.plot(fys, np.abs(resid), '.')

decorate(xlabel='x', ylabel='absolute residual')
```

Now, to see if the standard error we got from `linregress` is correct, we'll compute it again by bootstrapping.

```python
heterosked = pd.DataFrame(dict(x=xs, y=ys))
heterosked.corr()
```

The following function takes a `DataFrame` containing `x` and `y`, resamples the rows, and compute the slope of the regression line.

```python
def resample_slope(df):
    resampled = df.sample(frac=1, replace=True)
    res = linregress(resampled['x'], resampled['y'])
    return res.slope
```

If we call this function many times, the result is a sample from the sampling distribution of the slopes.

```python
sample = [resample_slope(heterosked) for i in range(1001)]
```

Here's what it looks like.

```python
sns.kdeplot(sample)
decorate(xlabel='Slope',
         ylabel='Density',
         title='Sampling distribution of the slope')
```

The standard deviation of the sampling distribution is the standard error of the estimate.

```python
stderr = np.std(sample)
stderr
```

The result is only slightly bigger than what we got from `linregress`.

```python
res.stderr
```

```python
import statsmodels.api as sm

# Fit a robust linear model
huber_model = sm.RLM(ys, sm.add_constant(xs), M=sm.robust.norms.HuberT())
huber_results = huber_model.fit()

# Print the summary of the model
print(huber_results.summary())
```

```python
import numpy as np
import statsmodels.api as sm

# Generate some example data
np.random.seed(0)
X = np.random.randn(100)
y = 2 * X + np.random.normal(scale=0.5, size=100)

# Add heteroskedasticity to the residuals
heteroskedasticity = np.abs(X) * np.random.normal(scale=0.1, size=100)
y += heteroskedasticity

plt.plot(X, y, '.')
```

```python
linregress(X, y)._asdict()
```

```python
# Fit a robust linear model
huber_model = sm.RLM(y, sm.add_constant(X), M=sm.robust.norms.HuberT())
huber_results = huber_model.fit()

# Print the summary of the model
print(huber_results.summary())
```

## Plotting

```python
def resample_fit(df, xs):
    resampled = df.sample(frac=1, replace=True)
    res = linregress(resampled.x, resampled.y)
    fys = res.intercept + res.slope * xs
    return fys
```

```python
def plot_percentile_fits(df, xs):
    fits = [resample_fit(heterosked, xs) for i in range(101)]
    low, high = np.percentile(np.array(fits), [5, 95], axis=0)
    plt.fill_between(xs, low, high, alpha=0.2)
```

```python
plot_percentile_fits(heterosked, xs)
plot_percentile_fits(homosked, xs)
```

```python

```

```python

```

```python

```

[House Sales in King County, USA](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction/data)

> This dataset contains house sale prices for King County, which includes Seattle. It includes homes sold between May 2014 and May 2015.

```python
df = pd.read_csv('kc_house_data.csv')
df.shape
```

```python
df.head()
```

```python
df['price'].describe()
```

```python
df['sqft_living'].describe()
```

```python
df['x'] = df['sqft_living']
df['y'] = df['price']
```

```python
plt.plot(df['x'], df['y'], '.', ms=3, alpha=0.3)
```

```python
linregress(xs, ys)._asdict()
```

```python
np.random.seed(17)
sample = [resample_slope(df) for i in range(1001)]
```

```python
sns.kdeplot(sample)
decorate(xlabel='Slope',
         ylabel='Density',
         title='Sampling distribution of the slope')
```

```python
stderr = np.std(sample)
stderr
```

```python
df['x'] = np.log10(df['sqft_living'])
df['y'] = np.log10(df['price'])
```

```python
plt.plot(df['x'], df['y'], '.', ms=1, alpha=0.1)
```

```python
linregress(df['x'], df['y'])._asdict()
```

```python
np.random.seed(17)
sample = [resample_slope(df) for i in range(1001)]
```

```python
sns.kdeplot(sample)
decorate(xlabel='Slope',
         ylabel='Density',
         title='Sampling distribution of the slope')
```

```python
stderr = np.std(sample)
stderr
```

```python

```

```python

```

```python

```

```python

```

[*Data Q&A: Answering the real questions with Python*](https://allendowney.github.io/DataQnA/index.html)

Copyright 2024 [Allen B. Downey](https://allendowney.com)

License: [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/)

```python

```

```python

```

```python

```
