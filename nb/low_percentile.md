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

# Bootstrapping percentiles

Here's a [question from the Reddit statistics forum](https://www.reddit.com/r/statistics/comments/1cal0fl/q_determine_confidence_interval_based_on_observed).

> I'm trying to figure out how to determine the confidence interval for the .2 percentile temperature for specific set of observed temperatures (all hourly temperatures during January, February, and December since 2000). I have recordings for 53128 of the 53424 possible hourly recordings.
>
> How would I go about saying that I am X% sure that the actual .2 percentile value is between two numbers? Could anyone provide any insight on how to accomplish this. Thank you.

OP provided a link to the data, so this is a question we can answer!
For computing confidence intervals, my first choice is bootstrap resampling, but as it turns out, it does not work well for this problem.
I'll show what goes wrong and how to fix it.
Then we'll answer a follow-up question about quantifying the effect of missing data.

<!-- #region tags=["remove-print"] -->
[Click here to run this notebook on Colab](https://colab.research.google.com/github/AllenDowney/DataQnA/blob/main/nb/low_percentile.ipynb).
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
```

```python tags=["hide-cell"]
# install the empiricaldist library, if necessary

try:
    import empiricaldist
except ImportError:
    !pip install empiricaldist
```

```python tags=["remove-print"]
import numpy as np
np.set_printoptions(legacy='1.25')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import decorate
```

## Data

I downloaded the data as a CSV file, which we can read into a Pandas `DataFrame`.

```python tags=["hide-cell"]
# download the data

download('https://github.com/AllenDowney/DataQnA/raw/main/data/temperature_data_ama.csv')
```

```python
col = 'Date/Time'
df = pd.read_csv('temperature_data_ama.csv', parse_dates=[col], index_col=col)
df.head()
```

There are 53424 measurements, of which 306 are missing.

```python
len(df)
```

```python
missing = df['tmpf'].isna()
missing.sum()
```

The range of temperatures is from -9.9 degF to 89 degF.

```python
data_clean = df['tmpf'].dropna()
data_clean.describe()
```

And the 0.2 percentile is 1 degF.

```python
np.percentile(data_clean, 0.2)
```

## Basic bootstrap

The following function takes the cleaned data, resamples it, and computes the 0.2 percentile of the bootstrapped sample.

```python
def bootstrap_percentile(data):
    resampled = np.random.choice(data, size=len(data), replace=True)
    return np.percentile(resampled, 0.2)
```

If we call this function 1001 times, we get a sample from the sampling distribution of the percentile.

```python
np.random.seed(17)
sample = [bootstrap_percentile(data_clean) for i in range(1001)]
```

Here's what that sample looks like.

```python
sns.histplot(sample)
decorate(xlabel='')
```

Immediately we can see that something has gone wrong.
The resampling process produces only 8 unique values. 

```python
np.unique(sample)
```

If we try to compute a CI by pulling percentiles from the sample, the results are not credible.

```python
np.percentile(sample, [5, 95])
```

This example demonstrates a limitation of bootstrap resampling -- it does not work well when there are a small number of unique values.

However, because the data are temperature measurements, they are actually continuous quantities.
So one option is to replace bootstrapping with a model that generates continuous quantities.
We'll try that with a normal model, see that it does not work, and they try again with KDE.

## Resampling from a normal model

If we look at the CDF of the data, it resembles the characteristic sigma of the normal distribution.

```python
from empiricaldist import Cdf

cdf_data = Cdf.from_seq(data_clean)
cdf_data.plot(label='data')

decorate(xlabel='Temperature (degF)',
         ylabel='CDF')
```

So let's see how it compares to a normal model.
I'll estimate the parameters by computing the mean and standard deviation of the data.

```python
from scipy.stats import norm

mu = np.mean(data_clean)
sigma = np.std(data_clean)
dist = norm(mu, sigma)
```

And compute the normal CDF within 4 standard deviations of the mean.

```python
low, high = mu - 4*sigma, mu + 4*sigma
xs = np.linspace(low, high, 201)
ys = dist.cdf(xs)
```

Here's what the model looks like compared to the data.

```python
plt.plot(xs, ys, color='gray', label='Normal model')
cdf_data.plot(label='data')

decorate(xlabel='Temperature (degF)',
         ylabel='CDF')
```

It looks pretty good, but there are places where the data clearly deviate from the model.
That's enough to make me worry, but let's proceed and see how it goes.

The following function takes the cleaned data, generates a random sample from the normal model, and returns the 0.2 percentile of the sample.

```python
def resample_percentile_norm(data):
    resampled = dist.rvs(len(data))
    return np.percentile(resampled, 0.2)
```

If we call it 1001 times, we hope the result is a sample from the sampling distribution of the percentile.

```python
np.random.seed(17)
sample2 = [resample_percentile_norm(data_clean) for i in range(1001)]
```

And at first glance it looks good.

```python
sns.kdeplot(sample2, label='Sampling distribution')
decorate(xlabel='Temperature (degF)',
         ylabel='Density')
```

But notice that range of the sampling distribution does not include the 0.2 percentile of the data, which is 1.
We can compute a 90% CI, but again, it is not credible.

```python
ci90 = np.percentile(sample2, [5, 95])
ci90
```

To see what went wrong, let's look at the normal model and the data again, this time with the `y` axis on a log scale.
The log scale is like a microscope that lets us see more clearly what is happening in the tail of the distribution.

```python
plt.plot(xs, ys, color='gray', label='Normal model')
cdf_data.plot(label='data')

decorate(xlabel='Temperature (degF)',
         ylabel='CDF',
         yscale='log')
```

On a linear scale, it seemed like the normal model might be good enough; on a log scale, it is clear that the data deviate from the model in the left tail.

In retrospect, it is not a surprise if a simple two-parameter model fails to capture every detail of the distribution -- the world is a complicated place.
So let's try a nonparametric approach.

## Resampling with KDE

We can use kernel density estimation (KDE) to model the distribution of the data, then use the model to resample.
Here's how we estimate the distribution.

```python
from scipy.stats import gaussian_kde

kde = gaussian_kde(data_clean)
```

To see what the result looks like, we can approximate the density of the model with a discrete PMF.

```python
from empiricaldist import Pmf

pmf_kde = Pmf(kde.pdf(xs), xs)
pmf_kde.normalize()
```

And then compare the CDF of the model with the CDF of the data.

```python
pmf_kde.make_cdf().plot(color='gray', label='KDE model')
cdf_data.plot(label='data')
decorate(xlabel='Temperature (degF)',
         ylabel='CDF')
```

The result shows that KDE is doing what it is meant to do -- fitting a continuous distribution to the data with minimal assumptions.
And if we put the `y` axis on a log scale, we can see that the KDE model fits the data well in the range we care about, near 1 degF.

```python
pmf_kde.make_cdf().plot(color='gray', label='KDE model')
cdf_data.plot(label='data')
decorate(xlabel='Temperature (degF)',
         ylabel='CDF',
         yscale='log')
```

The following function takes the cleaned data, uses the KDE model to generate a random sample, and returns the 0.2 percentile of the sample.

```python
def resample_percentile_kde(data):
    resampled = kde.resample(len(data))
    return np.percentile(resampled, 0.2)
```

If we call it 1001 times, we hope once again that the result is a sample from the sampling distribution of the percentile.

```python
np.random.seed(17)
sample3 = [resample_percentile_kde(data_clean) for i in range(1001)]
```

And this time we get a better result. The sampling distribution looks good, and it contains the actual percentile of the data.

```python
sns.kdeplot(sample3, label='Sampling distribution')
decorate(xlabel='Temperature (degF)',
         ylabel='Density')
```

And the width of the 90% CI is plausible.

```python
np.percentile(sample3, [5, 95])
```

So with a couple of false starts, we have answered the original question.
But it turns out there's more.

## Fill missing values

In a follow-up message, OP wrote:

> Just in case it helps any, here's what I'm ultimately trying to accomplish with this endeavor... I am trying to come up with a plausible way of demonstrating that the .2 percentile value (1 degF) that is derived from this data set is sufficiently representative of what the value would be if there were no missing data points (hourly readings) from the dataset.

OK, that's a different question!
However, the resampling framework can be extended naturally to estimate the effect of missing data.
Here's a function that takes the original data -- including NaNs -- and fills the missing values with a random selection of valid values.
For historical reasons, this way of filling missing values is called "hot deck imputation".

```python
data_nan = df['tmpf']
valid = data_nan.dropna()
missing = data_nan.isna()

def fill_missing(data):
    filled = data.copy()
    filled[missing] = np.random.choice(valid, size=missing.sum(), replace=True)
    return filled
```

To test it, we can check that the result has no NaNs.

```python
filled = fill_missing(data_nan)
filled.isna().sum()
```

Now we can include `fill_missing` as part of the resampling pipeline.
The following function takes the original data, fills missing values, generates a sample from a KDE model, and returns the 0.2 percentile of the sample.

```python
def resample_percentile_kde_fill(data):
    filled = fill_missing(data)
    kde = gaussian_kde(filled)
    resampled = kde.resample(len(data))
    return np.percentile(resampled, 0.2)
```

If we call it many times, we get a sample from a distribution that represents the uncertainty of the estimate due to a combination of missing data and random sampling.

```python
np.random.seed(17)
sample4 = [resample_percentile_kde_fill(data_nan) for i in range(1001)]
```

Here's what the result looks like, compared to the sampling distribution from the previous section, which represents only uncertainty due to random sampling.

```python
sns.kdeplot(sample3, label='Sampling distribution')
sns.kdeplot(sample4, label='Sampling distribution with fill')
decorate(xlabel='Temperature (degF)',
         ylabel='Density')
```

The difference does not seem substantial, and the CIs are similar.

```python
np.percentile(sample3, [5, 95])
```

```python
np.percentile(sample4, [5, 95])
```

We can conclude that missing data does not have much effect on the CI.

To estimate the effect more precisely, we could run this again with a sample size of 10,001 rather than 1001.
But I won't bother because with only 306 missing values out of 53,424, I did not expect the missing data to affect the results by much, and this result confirms it.
Rather than estimate the CI more precisely, I would conclude that missing data is not a problem, and drop it.

```python
filled = data_nan.iloc[8:].interpolate()
filled.plot()
```

```python
import warnings
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tsa.arima.model import ARIMA

# Fit ARIMA model
order = (1, 1, 1)  # Example order, choose based on AIC, etc.

# The index has gaps between seasons, so it has no frequency
with warnings.catch_warnings():
    warnings.simplefilter('ignore', ValueWarning)
    model = ARIMA(filled, order=order)
    results = model.fit()
```

```python
results.summary()
```

```python
n_simulations = 100
n_steps = len(filled)

simulated = results.simulate(n_steps)

```

```python
simulated.plot()
```

```python

```

```python

```

## Discussion

Normally I am quick to recommend bootstrap resampling because "it just works".
It makes almost no assumptions about the distribution of the data, and it is easy to extend to almost any statistic.
But as this example shows, it is not infallible -- the kryptonite of bootstrapping is lack of diversity in the data.

To diagnose this problem, it is a good idea to explore the sampling distribution.
If bootstrapping goes well, the sampling distribution should have many unique values, and the range should contain the estimate computed directly from the data, usually close to the middle of the CI.

If the results from bootstrapping fail these tests, think about other ways to model the data-generating process.
If a parametric model fits the data well, you can use the data to estimate parameters and then use the model to generate simulated samples.
Otherwise, consider a non-parametric approach like KDE.

For filling missing data, hot deck imputation ignores serial correlation and other statistical structure in the data, so the imputed values are likely to be unrealistic.
But in this case that's probably a feature, because the results overestimate the effect of missing data.
As a result, we can make the argument, "Even if we assume that the missing data is highly variable, it has no substantial effect on the estimated percentile or the computed CI."

If it were necessary to fill missing data with more realistic values, we could use a time series method like ARIMA or a Gaussian process.


*Data Q&A: Answering the real questions with Python*

Copyright 2024–2026 [Allen B. Downey](https://allendowney.com)

License: [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/)
