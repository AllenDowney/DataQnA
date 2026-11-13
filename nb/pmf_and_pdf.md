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

# PMFs and PDFs

Here's a [question from the Reddit statistics forum](https://www.reddit.com/r/statistics/comments/1cibmgg/q_comparison_between_pmf_and_pdf_on_a_plot/).

> I met a basic problem in pdf and pmf. I perform a grid approximation on bayesian problem. After normalizing the vector, I got a discretized pmf. Then I want to draw pmf and pdf on a plot to check if they are similar distribution. However, the pmf doesn’t resemble the pdf at all. The instruction told me that I need to sample from my pmf then draw a histogram with density for comparison. It really works, but why can't I directly compare them?

In *Think Bayes* I used this kind of discrete approximation in almost every chapter, so this issue came up a lot! There are at least two good solutions:

* Normalize the PDF and PMF so they are on the same scale, or

* Use CDFs.

I'll demonstrate both.

<!-- #region tags=["remove-print"] -->
[Click here to run this notebook on Colab](https://colab.research.google.com/github/AllenDowney/DataQnA/blob/main/nb/pmf_and_pdf.ipynb).
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

<!-- #region -->
## Baseball


As an example, I'll solve an exercise from Chapter 4 of [*Think Bayes*](https://allendowney.github.io/ThinkBayes2/chap04.html#exercises):

> In Major League Baseball, most players have a batting average between .230 and .300, which means that their probability of getting a hit is between 0.23 and 0.30.
>
> Suppose a player appearing in their first game gets 3 hits out of 3 attempts. What is the posterior distribution for their probability of getting a hit?

To represent the prior distribution, I'll use a beta distribution with parameters I chose to be consistent with the expected range of batting averages.
<!-- #endregion -->

```python
from scipy.stats import beta

a, b = 40, 110
beta_prior = beta(a, b)
beta_prior.mean(), beta_prior.ppf([0.15, 0.85])
```

Here's what the PDF of this distribution looks like.

```python
qs = np.linspace(0, 1, 201)
ps = beta_prior.pdf(qs)
```

```python
plt.plot(qs, ps, label='prior')
decorate(xlabel='Batting average', ylabel='PDF')
```

Unlike probability masses, probability densities can exceed 1.
But the area under the curve should be 1, which we can check using `trapezoid`, which is a NumPy function that computes numerical integrals using the trapezoid rule.

```python
area = np.trapezoid(ps, qs)
area
```

Within floating-point error, the area under the PDF is 1.
To do the Bayesian update, I'll put these values in a `Pmf` object and normalize it so the sum of the probability masses is 1.

```python
from empiricaldist import Pmf

pmf_prior = Pmf(ps, qs, copy=True)
pmf_prior.normalize()
```

Now we can use `binom` to compute the probability of the data for each possible batting average in `qs`.

```python
from scipy.stats import binom

k = 3
n = 3
likelihood = binom.pmf(k, n, qs)
```

To compute the posterior distribution, we multiply the prior by the likelihood and normalize again so the sum of the posterior `Pmf` is 1.

```python
pmf_posterior = pmf_prior * likelihood
pmf_posterior.normalize()
```

The result is a discrete approximation of the actual posterior distribution -- so let's see how close it is.

Because the beta distribution is the conjugate prior of the binomial likelihood function, the posterior is also a beta distribution, with parameters updated to reflect the data: 3 successes and 0 failures.

```python
beta_posterior = beta(a+3, b)
```

Here's what this theoretical posterior distribution looks like compared to our numerical approximation.

```python
ps = beta_posterior.pdf(qs)
plt.plot(qs, ps, color='gray', label='beta')
pmf_posterior.plot(label='posterior')

decorate(xlabel='Batting average', ylabel='PDF/PMF')
```

Oops! Something has gone wrong. I assume this is what OP meant by "the pmf doesn't resemble the pdf at all".

The problem is that the PMF is normalized so the total of the probability masses is 1, and the PDF is normalized so the area under the curve is 1.
They are not on the same scale, so the y-axis in this figure is different for the two curves.

To fix the problem, we can find the area under the PMF.

```python
area = np.trapezoid(pmf_posterior.ps, pmf_posterior.qs)
area
```

And divide through to create a discrete approximation of the posterior PDF.

```python
pdf_posterior = pmf_posterior / area
```

Now we can compare density to density.

```python
ps = beta_posterior.pdf(qs)
plt.plot(qs, ps, color='gray', label='beta')
pdf_posterior.plot(label='posterior')

decorate(xlabel='Batting average', ylabel='PDF')
```

The curves are visually indistinguishable, and the numerical differences are small. 

```python
diff = ps - pdf_posterior.ps
np.max(np.abs(diff))
```

As an aside, note that the posterior and prior distributions are not very different.
The prior mean is 0.267 and the posterior mean is 0.281.
If a rookie goes 3 for 3 in their first game, that's a good start, but it doesn't mean they are the next Ted Williams.

```python
pmf_prior.mean(), pmf_posterior.mean()
```

As an alternative to comparing PDFs, we could convert the PMF to a CDF, which contains the cumulative sum of the probability masses.

```python
cdf_posterior = pmf_posterior.make_cdf()
```

And compare to the mathematical CDF of the beta distribution.

```python
ps = beta_posterior.cdf(qs)
plt.plot(qs, ps, color='gray', label='beta')

cdf_posterior.step(label='posterior')

decorate(xlabel='Batting average', ylabel='CDF')
```

In this example, I plotted the discrete approximation of the CDF as a step function -- which is technically what it is -- to highlight how it overlaps with the beta distribution.

## Discussion

Probability density is hard to define -- the best I can do is usually, "It's something that you can integrate to get a probability mass".
And when you work with probability densities computationally, you often have to discretize them, which can add another layer of confusion.

One way to distinguish PMFs and PDFs is to remember that PMFs are normalized so sum of the probability masses is 1, and PDFs are normalized so the area under the curve is 1.
In general, you can't plot PMFs and PDFs on the same axes because they are not in the same units.


[*Data Q&A: Answering the real questions with Python*](https://allendowney.github.io/DataQnA/index.html)

Copyright 2024–2026 [Allen B. Downey](https://allendowney.com)

License: [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/)
