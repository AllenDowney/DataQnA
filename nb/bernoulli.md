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

# Bernoulli's Sample Size

Here's a [question from the Reddit statistics forum](https://www.reddit.com/r/statistics/comments/1g9r82f/q_required_sample_size_to_infer_which_of_two/).

> I'm looking at a Bernoulli process which may have either of two possible values of its trial probability parameter, let's call them `q1` and `q2`. I know these values beforehand, but not which of them is the right one.
> 
> I'm interested in finding out which of the two it is, but sampling from the process is quite costly (~10 real time minutes per sample), so before committing to this I would like to make a rough estimation for how many samples it will likely take to tell with reasonable confidence (let's say 95%) that the one that looks more likely in the running sample is indeed the right one. I'm aware that this required sample size will very sensitively depend on how close to each other `q1` and `q2` are.
> 
> So I suppose I'm looking for an approximation formula that relates sample size n, w.l.o.g. true probability `q1`, false probability `q2`, and required confidence level `c` (and, if that's not too much of a complication, Bayesian prior belief `b1` that `q1` is indeed true) to each other.
> 
> For actually conducting the experiment, I was just going to apply the law of total probability, using the binomial distribution's probability mass function with observed `k` and either value of `p` for the conditional probability update, sampling until either of the two model's posterior probability exceeds the required confidence level `c`. Is this a valid way to go about it or am I introducing any kind of bias this way?

There are two questions here: the first is whether we can say ahead of time what sample size we'll need to get to the desired level of confidence, `c`; the second is whether it's OK to sample until you reach `c`.
To answer the second question first, yes.

Now let's see why that's true.

<!-- #region tags=["remove-print"] -->
[Click here to run this notebook on Colab](https://colab.research.google.com/github/AllenDowney/DataQnA/blob/main/nb/bernoulli.ipynb).
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

## Bernoulli Trials

To put the question in a context, suppose you manufacture mechanical coin flippers that are supposed to be fair -- that is, they should flip heads or tails with equal probability.
And most of them do, but suppose 20% of the flippers have a defect that causes them to flip heads 70% of the time.
As part of the quality control process, you want to test the flippers and detect the defective ones.
How many flips does it take to be confident you have identified good and defective flippers correctly?

Here are the probabilities of flipping heads.

```python
q1 = 0.5
q2 = 0.7
```

And here are the prior probabilities that a random flipper is good or defective.

```python
b1 = 0.8
b2 = 0.2
```

I'll put those values in a `Pmf` object.

```python
from empiricaldist import Pmf

bs = [b1, b2]
qs = [q1, q2]
prior = Pmf(bs, qs)
prior
```

The following function simulates flipping `n` coins with probability `p` and returns the number of heads.

```python
from scipy.stats import binom

def generate(n, p):
    return n, binom(n, p).rvs()
```

Here's an example with `n=10` and probability `0.5`.

```python
np.random.seed(1)

n = 10
n, k = generate(n, 0.5)
k
```

Now suppose we don't know `q`, but we flip 10 times and get 5 heads. That's more likely if `q` is 0.5 than if `q` is 0.65.
To figure out how much more likely, we can use the binomial PMF, as OP suggested.

```python
likelihood = binom.pmf(k, n, [q1, q2])
likelihood
```

To get the posterior probabilities for `q1` and `q2`, we multiply the priors by the likelihoods and renormalize the PMF.

```python
posterior = prior * likelihood
posterior.normalize()
posterior
```

Getting 5 heads out of 10 flips is evidence in favor of `q1`, but not strong enough to reach a posterior probability of 0.95.

The following function encapsulates the steps of the update we just computed.

```python
def update(prior, data):
    n, k = data
    likelihood = binom.pmf(k, n, [q1, q2])
    posterior = prior * likelihood
    posterior.normalize()
    return posterior
```

We can use it to simulate the experiment, generating `n` flips and computing the posterior probabilities.

```python
n = 10
update(prior, generate(n, q1))
```

Now, let's find the sample size that's big enough to distinguish good and defective flippers with 95% confidence.
We'll use one of my favorite methods for finding critical values, a biased random walk.

The following function simulates a series of experiments with different values of `n`.
After each experiment, it computes the posterior probability of the correct value of `q`. If that probability is less than 95%, that means the sample size was not big enough, so it increases `n` for the next experiment. If the probability exceeds 95%, that means the sample size was bigger than necessary, so it decreases `n` for the next experiment. 

```python
def simulate(n, q, threshold=0.95, iters=3001):
    ns = []
    for i in range(iters):
        ns.append(n)
        posterior = update(prior, generate(n, q))
        if posterior[q] < threshold:
            n += 1
        else:
            n -= 1
    return ns
```

Here's how we use this function, starting with an initial guess about `n`.

```python
n0 = 18
ns = simulate(n0, q1)
mean = np.mean(ns)
mean
```

This method should move toward a value of `n` that has a 50% chance of reaching a final confidence of 95%, and then vary randomly around that value.
So the average is a good estimate of the critical value of `n`.
For a good device, `n` is around 18.

```python
plt.plot(ns)
plt.axhline(mean, ls=':', color='gray')
decorate(xlabel='Time step', ylabel='n')
```

For a bad device, it takes longer to accumulate enough evidence to overcome the lower prior probability.

```python
n0 = 52
ns = simulate(n0, q2)
mean = np.mean(ns)
mean
```

The critical value of `n` is around 52.

```python
plt.plot(ns)
plt.axhline(mean, ls=':', color='gray')
decorate(xlabel='Time step', ylabel='n')
```

So that's an answer to the first question. It's not a formula, as OP requested, but if we're given different values of `q1`, `q2`, and the prior probabilities, we can run these simulations to estimate the critical values of `n`.

## Run Until Confident

Now here's a different version of the experiment that runs until one of the posterior probabilities exceeds the threshold.

```python

def run_until_threshold(prior, actual, n=1, threshold=0.95):
    """
    """
    posterior = prior
    for i in range(1, 1001):
        data = generate(n, actual)
        posterior = update(posterior, data)
        for q, p in posterior.items():
            if p >= threshold:
                return i, q

    return -1, np.nan
```

The return values are the number of time steps, and the conclusion of the experiment, which is the value of `q` whose posterior probability exceeded the threshold.

```python
actual = prior.sample(1)[0]
i, final = run_until_threshold(prior, actual)
actual, i, final
```

The following loop runs this simulated experiment many times and put the results in a `DataFrame`.

```python
res = []
for i in range(1001):
    actual = prior.sample(1)[0]
    i, final = run_until_threshold(prior, actual)
    res.append((actual, i, final))

df = pd.DataFrame(res, columns=['actual', 'steps', 'final'])
df.head()
```

The average number of steps to reach the given level of confidence is about 25.

```python
df['steps'].mean()
```

And the conclusion is correct about 95% of the time, as promised.

```python
def is_correct(df):
    return (df['actual'] == df['final']).mean()
```

```python
is_correct(df)
```

And that's true whether the conclusion is `q1` or `q2`.

```python
is_correct(df.loc[df['final'] == q1])
```

```python
is_correct(df.loc[df['final'] == q2])
```

But the number of steps is shorter if the actual value is `q1`, and longer if it's `q2`.

```python
steps1 = df.loc[df['actual'] == q1, 'steps']
steps1.mean()
```

```python
steps2 = df.loc[df['actual'] == q2, 'steps']
steps2.mean()
```

Here's what the distribution of step sizes looks like.

```python
from empiricaldist import Cdf

Cdf.from_seq(steps1).plot(label=f'actual = {q1}')
Cdf.from_seq(steps2).plot(label=f'actual = {q2}')
decorate(ylabel='CDF')
```

Usually it doesn't take more than 50 steps to reach the threshold, but if you're unlucky, it can take more than 100. 

## Discussion

In classical hypothesis testing, it is a no-no to run an experiment until it reaches a desired level of significance. In fact, under the null hypothesis, [the probability of generating a false positive approaches 100%](https://allendowney.blogspot.com/2011/10/repeated-tests-how-bad-can-it-be.html).

But with this kind of Bayesian update, it's not a problem. The results depend only on the data, not the stopping condition. And the results in this notebook show that the posterior probabilities are calibrated -- that is, if you are 95% sure you have identified a good or defective flipper, you are right 95% of the time.

Although we used Bayes's Theorem to compute the posterior probabilities, I would argue that this is not really an example of Bayesian statistics. Bayes's Theorem is a mathematical law -- you can use it with no obligation to adopt the Bayesian interpretation of probability. In this example, if we choose a flipper at random, the unknown parameter, `q`, is a random quantity with an objectively known distribution. So the analysis we did is valid under even the strictest frequentist interpretation of probability.


[*Data Q&A: Answering the real questions with Python*](https://allendowney.github.io/DataQnA/index.html)

Copyright 2024–2026 [Allen B. Downey](https://allendowney.com)

License: [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/)
