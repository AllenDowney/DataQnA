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

# Standardization and Normalization

Here's a [question from the Reddit statistics forum](https://www.reddit.com/r/statistics/comments/1drvlyt/q_what_do_to_when_a_sw_and_ks_test_fail_on_a_pre/).

> I want to write a research article that has regression analysis in it. I normalized my independent variables and want to include in the article the results of a statistical test showing that all variables are normal. I normalized using the scale function in R and some custom normalization functions I found online but whatever I do the new data fails the Shapiro Wilkinson and KS test on some columns? What to do? 

There might be a few points of confusion here:

* One is the idea that the independent variables in a regression model have to follow a normal distribution. This is a common misconception. Ordinary least squares regression assumes that the *residuals* follow a normal distribution -- the dependent and independent variables don't have to.

* Another is the idea that statistical tests are useful for checking whether a variable follows a normal distribution. As I've explained in other articles, these tests don't do anything useful.

* The question might also reveal confusion about what "normalization" means. In the context of regression models, it usually means scaling a variable so its range is between 0 and 1. Its sounds like OP might be using it to mean transforming a variable so it follows a normal distribution.

In general, transforming an independent variable to normal is not a good idea. It has no benefit, and it changes the estimated parameters for no justified reason. Let me explain.

<!-- #region tags=["remove-print"] -->
[Click here to run this notebook on Colab](https://colab.research.google.com/github/AllenDowney/DataQnA/blob/main/nb/standardize.ipynb).
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

## Happiness

As an example, I'll use data from the [World Happiness Report](https://worldhappiness.report/), which uses survey data from 153 countries to explore the relationship between self-reported happiness and six potentially predictive factors:

* Income as represented by per capita GDP.

* Social support

* Healthy life expectancy at birth

* Freedom to make life choices

* Generosity

* Perceptions of corruption

The dependent variable, happiness, is the national average of responses to the "Cantril ladder question" used by the [Gallup World Poll](https://news.gallup.com/poll/122453/understanding-gallup-uses-cantril-scale.aspx):

> Please imagine a ladder with steps numbered from zero at the bottom to 10 at the top.
The top of the ladder represents the best possible life for you and the bottom of the ladder represents the worst possible life for you.
On which step of the ladder would you say you personally feel you stand at this time?  

The data they used is [available here](https://happiness-report.s3.amazonaws.com/2020/WHR20_DataForFigure2.1.xls).

```python
download('https://happiness-report.s3.amazonaws.com/2020/WHR20_DataForFigure2.1.xls')
```

We can read the data into a Pandas `DataFrame`.

```python
df = pd.read_excel('WHR20_DataForFigure2.1.xls')
df.head()
```

I'll select the columns we're interested in and give them shorter names.

```python
data = pd.DataFrame()

data['ladder'] = df['Ladder score']
data['log_gdp'] = df['Logged GDP per capita']
data['social'] = df['Social support']
data['life_exp'] = df['Healthy life expectancy']
data['freedom'] = df['Freedom to make life choices']
data['generosity'] = df['Generosity']
data['corruption'] = df['Perceptions of corruption']
```

Notice that GDP is already on a log scale, which is generally a good idea.
The other variables are in a variety of units and different scales.

Now, contrary to the advice OP was given, let's run a regression model without even looking at the data.

```python
import statsmodels.formula.api as smf

formula = 'ladder ~ log_gdp + social + life_exp + freedom + generosity + corruption'

results_raw = smf.ols(formula, data=data).fit()
results_raw.params
```

Because the independent variables are on different scales, the estimated parameters are in different units.
For example, `life_exp` is healthy life expectancy at birth in units of years, so the estimated parameter, 0.035, indicates that a difference of one year of life expectancy between countries is associated with an increase of 0.035 units on the Cantril ladder.

Now let's check retroactively whether this dataset is consistent with the assumption that the residuals follow a normal distribution.
I'll use the following functions to compare the CDF of the residuals to a normal model with the same mean and standard deviation. 

```python
from scipy.stats import norm
from empiricaldist import Cdf

def make_normal_model(sample):
    """Make a Cdf of a normal distribution.
    
    sample: sequence of numbers
    
    returns: Cdf object
    """
    m, s = sample.mean(), sample.std()
    qs = np.linspace(m - 4 * s, m + 4 * s, 101)
    ps = norm.cdf(qs, m, s)
    return Cdf(ps, qs)
```

```python
def plot_normal_model(sample, **options):
    """Plot the Cdf of a sample and a normal model.
    
    sample: sequence of numbers
    """
    cdf_model = make_normal_model(sample)
    cdf_data = Cdf.from_seq(sample)

    cdf_model.plot(color='gray')
    cdf_data.plot(**options)
```

```python
plot_normal_model(results_raw.resid, label='residuals')
decorate(xlabel='residuals', ylabel='CDF')
```

The normal model fits the residuals well enough that the deviations are not a concern.
This regression model is just fine with no transformations required.

But suppose we follow the advice OP received and check whether the dependent and independent variables follow normal distributions.

## Looking for trouble

Here are the distributions of the variables compared to a normal model.
Sometimes the model fits pretty well, sometimes not.

```python
column = 'ladder'
plot_normal_model(data[column], label=column)
decorate(xlabel=column, ylabel='CDF')
```

```python
column = 'log_gdp'
plot_normal_model(data[column], label=column)
decorate(xlabel=column, ylabel='CDF')
```

```python
column = 'social'
plot_normal_model(data[column], label=column)
decorate(xlabel=column, ylabel='CDF')
```

```python
column = 'life_exp'
plot_normal_model(data[column], label=column)
decorate(xlabel=column, ylabel='CDF')
```

```python
column = 'freedom'
plot_normal_model(data[column], label=column)
decorate(xlabel=column, ylabel='CDF')
```

```python
column = 'generosity'
plot_normal_model(data[column], label=column)
decorate(xlabel=column, ylabel='CDF')
```

```python
column = 'corruption'
plot_normal_model(data[column], label=column)
decorate(xlabel=column, ylabel='CDF')
```

If we ran a statistical test, most of these would fail. But we don't care -- regression models don't require these variables to follow normal distributions. And when we make this kind of comparison across countries, there no reason to expect them to.

## Standardize

Nevertheless, there are reasons we might want to transform the independent variables before running a regression model -- one is to quantify the "importance" of the different factors. We can do that by standardizing the variables, which means transforming them to have mean 0 and standard deviation 1.

In the untransformed variables, the means vary in magnitude from about 0.1 to 64.

```python
data.mean().sort_values()
```

And the standard deviations vary from about 0.1 to 7.

```python
data.std().sort_values()
```

And, as I've already noted, they are expressed in different units.
There is no meaningful way to compare the parameter of `life_exp`, which is steps of the Cantril ladder per year of life expectancy, with the parameter of `corruption`, which is in steps of the ladder per percentage point.
But we can make these comparisons meaningful by standardizing the variables -- that is, subtracting off the mean and dividing by the standard deviation.

```python
standardized = (data - data.mean()) / data.std()
```

After transformation, the means are all close to 0.

```python
standardized.mean()
```

And the standard deviations are all close to 1.

```python
standardized.std()
```

Here's the regression with the transformed variables.

```python
results_standardized = smf.ols(formula, data=standardized).fit()
results_standardized.params
```

The intercept is close to 0 -- that's a consequence of how the math works out.
And now the magnitudes of the parameters indicate how much a change in each independent variable, expressed as a multiple of its standard deviation, is associated with a change in the dependent variables, also as a multiple of its standard deviation.

By this metric, `social` is the most important factor, followed closely by `log_gdp` and `life_exp`. `generosity` is the least important factor, at least as its quantified by this variable.

## Normalizing

An alternative to standardization is normalization, which transforms a variable so its range is between 0 and 1.
To be honest, I'm not sure what the point of normalization is -- it makes the results a little harder to interpret, and it doesn't have any advantages I can think of.

But just for completeness, here's how it's done.


```python
normalized = (data - data.min()) / (data.max() - data.min())
```

As intended, the normalized variables have the same range.

```python
normalized.min()
```

```python
normalized.max()
```

But as a result they have somewhat different means and standard deviations.

```python
normalized.mean()
```

```python
normalized.std()
```

Here's the regression with normalized variables.

```python
results_normalized = smf.ols(formula, data=normalized).fit()
results_normalized.params
```

We'll look more closely at those results soon, but first let's look at one more preprocessing option, transforming values to follow a normal distribution.

## Transform to normal

Here's how we transform values to a normal distribution. 

```python
n = len(data)
ps = np.arange(1, n+1) / (n + 1)
zs = norm.ppf(ps)

transformed = data.copy()
for column in transformed.columns:
    transformed = transformed.sort_values(column)
    transformed[column] = zs
```

The `ps` are the cumulative probabilities; the `zs` are the corresponding quantiles in a standard normal distribution (`ppf` computes the "percent point function" which is an unnecessary name for the inverse CDF).
Inside the loop, we sort `transformed` by one of the columns and then assign the `zs` in that order.

The transformed means are all close to 0.

```python
transformed.mean()
```

And the standard deviations are close to 1 (but not exact because the tails of the distribution are cut off).

```python
transformed.std()
```

As intended, the transformed variables follow a normal distribution.

```python
plot_normal_model(transformed['corruption'], label='corruption')
decorate(ylabel='CDF')
```

Here's the regression model with the transformed values.

```python
results_transformed = smf.ols(formula, data=transformed).fit()
results_transformed.params
```

Now let's compare the estimated parameters with the three transformations.

```python
results_standardized.params.plot(style='o', label='standardized')
results_normalized.params.plot(style='x', label='normalized')
results_transformed.params.plot(style='+', label='transformed')

decorate(ylabel='Estimated parameter')
```

The results with standardized variables are the easiest to interpret, so in that sense they are "right". The other transformations have no advantages over standardization, and they produce somewhat different parameters.

## Discussion

Regression does not require the dependent or independent variables to follow a normal distribution.
It only assumes that the distribution of residuals is approximately normal, and even that requirement is not strict -- regression is quite robust.

You do not need to check whether any of these variables follows a normal distribution, and you definitely should not use a statistical test.

As part of your usual exploratory data analysis, you should look at the distribution of all variables, but that's primarily to detect any anomalies, outliers, or really extreme distributions -- not to check whether they fit normal distributions.

If you want to use the parameters of the model to quantify the importance of the independent variables, use standardization. There's no reason to use normalization, and definitely no reason to transform to a normal distribution.


[*Data Q&A: Answering the real questions with Python*](https://allendowney.github.io/DataQnA/index.html)

Copyright 2024 [Allen B. Downey](https://allendowney.com)

License: [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/)
