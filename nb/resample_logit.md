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

# Resampling and Logistic Regression

Here's a [question from Reddit's statistics forum](https://www.reddit.com/r/statistics/comments/10ayfm5/q_parametric_bootstrap_for_logistic_regression/):

> I am trying to do parametric bootstrap for logistic regression but I don't know what the random error terms should be. I know how to do it with linear regression since the error terms follow a normal distribution. Really appreciate any pointers to resources about this topic.

One of the responses recommends [this paper](https://www.scirp.org/journal/paperinformation.aspx?paperid=70962): "An Application of Bootstrapping in Logistic Regression Model", by Isaac Akpor Adjei and Rezaul Karim.


The paper suggests two ways to compute the sampling distribution of the parameters of a logistic regression by bootstrap sampling.
They characterize one as parametric and the other as non-parametric, and advise:

> The non-parametric bootstrap [relies] on weaker assumptions (or no assumptions at all) about the underlying distribution of the population. Statistical practitioners should use non-parametric procedures only in so far as the assumptions about the underlying distribution are seriously doubtful in their validity. ... However, when assumptions are not violated, non-parametric procedures will usually have greater variance (in point estimation), less power (in hypothesis testing), wider intervals (in confidence interval estimation), lower probability of correct selection (in ranking and selection) and higher risk (in decision theory) when compared to a corresponding parametric procedure (Efron and Tibshirani, 1994 [1] ).

The premise of this advice is that the parametric and non-parametric methods are answering the same question in different ways, and that one might be preferred over the other because the results are preferable in one way or another. It is also based on the assumption that the sampling distribution of the parameters -- and the statistics derived from it -- are uniquely and objectively defined.


I disagree with these premises. The parametric and non-parametric methods they present are based on different modeling decisions, so they compute answers to different questions.
And the models they are based on are only two of many possible models.

To explain what I mean, I will implement the methods they propose and explain the assumptions each is based on. Then I will propose a third method that is a hybrid of the two. I'll show that all three methods yield different results, and suggest criteria for when one might be preferred over the others.

<!-- #region tags=["remove-print"] -->
[Click here to run this notebook on Colab](https://colab.research.google.com/github/AllenDowney/DataQnA/blob/main/nb/resample_logit.ipynb).
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

```python tags=["remove-print"]
import numpy as np
np.set_printoptions(legacy='1.25')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import decorate
```

## The Data

As an example, I'll use data from the General Social Survey (GSS).
I'll download an HDF file from the *Elements of Data Science* repository, which contains a subset of the GSS data that has been resampled to correct for stratified sampling.

```python
download('https://github.com/AllenDowney/ElementsOfDataScience/raw/master/data/gss_eda.hdf')
```

```python
gss = pd.read_hdf('gss_eda.hdf', 'gss')
gss.shape
```

It includes 64,814 respondents and 8 variables for each respondent.

```python
gss.head()
```

To demonstrate logistic regression, I'll use [one of the questions](https://gssdataexplorer.norc.org/variables/285/vshow) in the General Social Survey, which asks "Do you think the use of marijuana should be made legal or not?"
The responses are in a column called `GRASS` -- here are the values.  

```python
gss['GRASS'].value_counts()
```

I'll use StatsModels, which provides a function that does logistic regression.
First we have to recode the dependent variable so `1` means "yes" and `0` means "no".  We can do that by replacing `2` with `0`.

```python
gss['GRASS'] = gss['GRASS'].replace(2, 0)
```

To model quadratic relationships, I'll add columns that contain the values of `AGE` and `EDUC` squared.

```python
gss['AGE2'] = gss['AGE']**2
gss['EDUC2'] = gss['EDUC']**2
```

And I'll drop the rows that have missing values for the variables we'll need.

```python
data = gss.dropna(subset=['AGE', 'EDUC', 'SEX', 'GUNLAW', 'GRASS'])
```

## Logistic Regression

Here are the results of logistic regression with these variables.

```python
import statsmodels.formula.api as smf

formula = 'GRASS ~ AGE + AGE2 + EDUC + EDUC2 + C(SEX)'
result_hat = smf.logit(formula, data=data).fit()
result_hat.summary()
```

To get a sense of what the results look like, we can plot the predicted probability of saying  "yes" as a function of age, for male and female respondents.

```python
df = pd.DataFrame()
df['AGE'] = np.linspace(18, 89)
df['EDUC'] = 16

df['AGE2'] = df['AGE']**2
df['EDUC2'] = df['EDUC']**2

df['SEX'] = 1
pred1 = result_hat.predict(df)
pred1.index = df['AGE']

df['SEX'] = 2
pred2 = result_hat.predict(df)
pred2.index = df['AGE']

pred1.plot(label='Male', alpha=0.6)
pred2.plot(label='Female', alpha=0.6)

plt.xlabel('Age')
plt.ylabel('Fraction')
plt.title('Support for legal marijuana')
plt.legend();
```

Men are more likely to support legalization than women, and younger people more likely than older people.

## Bootstrap sampling

Now let's implement the method characterized as non-parametric, which is based on bootstrap resampling.
The following function samples the rows of `data` with replacement and runs the regression model on the resampled data.

```python
options = dict(disp=False, start_params=result_hat.params)

def bootstrap(i):
    rng = np.random.default_rng(i)
    bootstrapped = data.sample(n=len(data), replace=True, random_state=rng)
    results = smf.logit(formula, data=bootstrapped).fit(**options)
    return results.params
```

The options sent to `fit` make it run faster, but don't affect the results by much.
Each call makes its own random number generator, seeded with `i`, so the results are reproducible.

Each time we run this process, the result represents a single draw from the sampling distribution of the parameters.

```python
bootstrap(0)
```

By running it many times, we generate a sample from the sampling distribution.

```python
estimates = [bootstrap(i) for i in range(101)]
sampling_dist = pd.DataFrame(estimates)
sampling_dist.head()
```

Here's what the sampling distribution looks like for one of the parameters.

```python
ci90 = sampling_dist['C(SEX)[T.2]'].quantile([0.05, 0.95])
sns.kdeplot(sampling_dist['C(SEX)[T.2]'])
[plt.axvline(x, ls=':') for x in ci90]
plt.title('Sampling distribution of a parameter');
```

The mean of the sampling distribution should be close to the parameters we estimated with the original dataset, and it is.

```python
pd.DataFrame({"Sampling mean": sampling_dist.mean(), 
              "Estimates": result_hat.params})
```

The standard deviations of the sampling distributions should be close to the standard errors computed by StatsModels, and they are.
Most of them are close enough that the difference probably doesn't matter in practice.

```python
def standard_errors(sampling_dist, result_hat):
    df = pd.DataFrame({"Sampling std": sampling_dist.std(), 
                       "Standard error": result_hat.bse})
    num, den = df.values.T
    df['Percent diff'] = (num / den - 1) * 100
    return df
```

```python
standard_errors(sampling_dist, result_hat)
```

Of course, there is nothing magic about the standard errors computed by StatsModels -- they are approximations based on a model of the sampling process, just like the results from resampling.

## The Resampling Framework

The purpose of the sampling distribution is to quantify the variability in an estimate due to random sampling. It is the answer to the question, "If we ran this sampling process many times, how much would we expect the results to vary?"

To answer that question, we need a model of the sampling process. We use the model to simulate the sampling process and generate counterfactual datasets that represent other samples the process could have generated.

The following figure represents this framework.


<img src="https://github.com/AllenDowney/ElementsOfDataScience/raw/master/figs/resampling.png" width="500">


In bootstrap resampling, we treat the rows of the original dataset as if they are the population and draw a random sample from it.
We use this simulated data to compute the sample statistic; in this example, it's the parameters of the logistic regression model.
We collect the results to form a sample from the sampling distribution, which we can use to compute confidence intervals and standard errors.

This way of simulating the sampling process asks, in effect, what would have happened if we had surveyed different people.
But that's not the only possible model of the sampling process.
An alternative is to ask what would happen if we surveyed the same people, but they gave different answers.
That's the model that underlies the parametric method.

## Parametric bootstrap

So, how do we simulate a sampling process where we survey the same people, but they give different answers?
One way is to assume, for the purposes of the simulation, that the parameters we estimated from the original data are correct.
If so, we can use the regression model to compute the predicted probability for each respondent, like this:

```python
pi_hat = result_hat.predict(data)
pi_hat.describe()
```

Now we can use these probabilities to generate a biased coin toss for each respondent.

```python
from scipy.stats import bernoulli

rng = np.random.default_rng(0)
simulated = bernoulli.rvs(pi_hat.values, random_state=rng)
simulated.mean()
```

Then we can run the regression model with these simulated values as the dependent variable.
The following function shows how.

```python
def bootstrap2(i):
    rng = np.random.default_rng(i)
    flipped = data.assign(GRASS=bernoulli.rvs(pi_hat.values, random_state=rng))
    results = smf.logit(formula, data=flipped).fit(**options)
    return results.params
```

Again, the result from a single simulation is a random value from the sampling distribution.

```python
bootstrap2(0)
```

If we run it many times, we get a sample from the sampling distribution.

```python
estimates = [bootstrap2(i) for i in range(101)]
```

```python
sampling_dist2 = pd.DataFrame(estimates)
sampling_dist2.head()
```

Here's the sampling distribution for a single parameter.

```python
ci90 = sampling_dist2['C(SEX)[T.2]'].quantile([0.05, 0.95])
sns.kdeplot(sampling_dist2['C(SEX)[T.2]'])
[plt.axvline(x, ls=':') for x in ci90]
plt.title('Sampling distribution of a parameter');
```

Again, the standard deviations should be close to the standard errors, and they are.

```python
standard_errors(sampling_dist2, result_hat)
```

So we have two models of the sampling process that yield different sampling distributions. But these are not the only models.

If the first method asks, "What if we surveyed different people?" and the second asks "What if we surveyed the same people and they gave different answers?", let's consider a third method that asks "¿Por qué no los dos?".

## The Hybrid Model

The following function uses bootstrap sampling to simulate surveying different people; then it uses the parametric method to simulate their responses.

```python
def bootstrap3(i):
    rng = np.random.default_rng(i)
    bootstrapped = data.sample(n=len(data), replace=True, random_state=rng)
    pi_hat = result_hat.predict(bootstrapped)
    flipped = bootstrapped.assign(GRASS=bernoulli.rvs(pi_hat.values, random_state=rng))
    results = smf.logit(formula, data=flipped).fit(**options)
    return results.params
```

```python
bootstrap3(0)
```

```python
estimates = [bootstrap3(i) for i in range(101)]
```

```python
sampling_dist3 = pd.DataFrame(estimates)
sampling_dist3.head()
```

Here's the sampling distribution for one of the parameters.

```python
ci90 = sampling_dist3['C(SEX)[T.2]'].quantile([0.05, 0.95])
sns.kdeplot(sampling_dist3['C(SEX)[T.2]'])
[plt.axvline(x, ls=':') for x in ci90]
plt.title('Sampling distribution of a parameter');
```

Again, the standard errors are comparable to the ones computed by StatsModels.

```python
standard_errors(sampling_dist3, result_hat)
```

Now we have four ways to compute sampling distributions, confidence intervals, and standard errors -- and they yield different results. So you might wonder...

## Which One Is Right?

None of them are right.
They are based on different models of the sampling process, so they quantify different sources of uncertainty.

In some cases we might find that different methods converge on the same results, eventually, under certain assumptions.
But eventually doesn't really matter -- with finite data sets, the results are not generally the same.
So the important question is whether the differences are big enough to matter.

In this example, it is easy to implement multiple models and compare the results.
If they were substantially different, we would need to think more carefully about the modeling assumptions they are based on and choose the one we think is the best description of the data-generating process.

But in this example, the differences are small enough that they probably don't matter in practice.
So we are free to choose whichever is easiest to implement, or fastest to compute, or convenient in some other way.

It is a common error to presume that the result of an analytic method is uniquely correct, and that results from computational methods like resampling are approximations to it.
Analytic methods are often fast to compute, but they are always based on modeling assumptions and usually based on approximations, so they are no more correct than computational methods.

## Discussion

To answer the original question: in a parametric bootstrap for linear regression, we simulate a dataset by adding random errors, drawn from a normal distribution, to the predicted values.
Logistic regression has no additive error term.
The model predicts a probability for each respondent, and the randomness is in the outcome: each response is a Bernoulli trial with that probability.
So to simulate a dataset, we compute the predicted probabilities and flip a biased coin for each row, as in `bootstrap2`.

But the more important point is that the parametric bootstrap is not a better or worse version of the non-parametric one.
Each method is based on a different model of the sampling process, so they answer different questions: what if we surveyed different people, what if the same people gave different answers, or both.
When the results differ, the question is not which method is right, but which model best describes the data-generating process -- and whether the differences are big enough to matter.


[*Data Q&A: Answering the real questions with Python*](https://allendowney.github.io/DataQnA/index.html)

Copyright 2024–2026 [Allen B. Downey](https://allendowney.com)

License: [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/)
