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

# Combining Risks

Here's a [question from the Reddit statistics forum](https://www.reddit.com/r/AskStatistics/comments/1cufwos/can_someone_please_help_me_calculate_combined/).

> Bit of a weird one but I'm hoping you're the community to help. I work in children's residential care and I'm trying to find a way of better matching potential young people together.
>
> The way we calculate individual risk for a child is 
>
> `risk = likelihood + impact` (`R=L+I`), so
>
> `L4 + I5 = R9`
>
> That works well for individuals but I need to work out a good way of calculating a combined risk to place children [in] the home together. I'm currently using the [arithmetic] average but I don't feel that it works properly as the average is always lower then the highest risk.
>
> I'll use a fairly light risk as an example, running away from the home. (We call this MFC missing from care) It's fairly common that one of the kids will run away from the home at some point or another either out of boredom or frustration. If young person A has a risk of 9 and young person B has a risk of 12 the the average risk of MFC in the home would be 10.5
>
> HOWEVER more often then not having two young people that go MFC will often result in more episodes as they will run off together, so having a lower risk rating doesn't really make sense. Adding the two together to 21 doesn't really work either though as the likelihood is the thing that increases not necessarily the impact.
>
> I'm a lot better at chasing after run away kids then I am mathematics so please help 😂.

Here's one way to think about this question: based on background knowledge and experience, OP has qualitative ideas about what happens when we put children at different risks together, and he is looking for a statistical summary that is consistent with these ideas.

The arithmetic mean probably makes sense as a starting point, but it clashes with the observation that if you put two children together who are high risk, they interact in ways that increase the risk.
For example, if we put together children with risks 9 and 12, the average is 10.5, and OP's domain knowledge says that's too low -- it should be more than 12.

At the other extreme, I'll guess that putting together two low risk children might be beneficial to both -- so the combined risk might be lower than either.

And that implies that there is a neutral point somewhere in the middle, where the combined risk is equal to the individual risks.

To construct a summary statistic like that, I suggest a weighted sum of the arithmetic and geometric means.
That might sound strange, but I'll show that it has the properties we want.
And it might not be as strange as it sounds -- there's a reason it might make sense.

<!-- #region tags=["remove-print"] -->
[Click here to run this notebook on Colab](https://colab.research.google.com/github/AllenDowney/DataQnA/blob/main/nb/combine_risk.ipynb).
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

## Weighted sum of means

The following function computes the arithmetic mean of a sequence of values, which is the sum divided by `n`.

```python
def amean(xs):
    n = len(xs)
    return np.sum(xs) / n
```

The following function computes the geometric mean of a sequence, which is the product raised to the power `1/n`.

```python
def gmean(xs):
    n = len(xs)
    return np.prod(xs) ** (1/n)
```

And the following function computes the weighted sum of the arithmetic and geometric means.
The constant `k` determines how much weight we give the geometric mean.

```python
def mean_plus_gmean(*xs, k=1):
    return amean(xs) + k * (gmean(xs) - 4)
```

The value `4` determines the neutral point. So if we put together two people with risk `4`, the combined average is `4`.

```python
mean_plus_gmean(4, 4)
```

Above the neutral point, there is a penalty if we put together two children with higher risks.

```python
mean_plus_gmean(5, 5)
```

In that case, the combined risk is higher than the individual risks.
Below the neutral point, there is a bonus if we put together two children with low risks.

```python
mean_plus_gmean(3, 3)
```

In that case, the combined risk is less than the individual risks.

If we combine low and high risks, the discrepancy brings the average down a little.

```python
mean_plus_gmean(3, 5)
```

In the example OP presented, where we put together two people with high risk, the penalty is substantial.

```python
mean_plus_gmean(9, 12)
```

If that penalty seems too high, we can adjust the weight, `k`, and the neutral point accordingly.

This behavior extends to more than two people.
If everyone is neutral, the result is neutral.

```python
mean_plus_gmean(4, 4, 4)
```

If you add one person with higher risk, there's a moderate penalty, compared to the arithmetic mean.

```python
mean_plus_gmean(4, 4, 5), amean([4, 4, 5])
```

With two higher risk people, the penalty is bigger.

```python
mean_plus_gmean(4, 5, 5), amean([4, 5, 5])
```

And with three it is even bigger.

```python
mean_plus_gmean(5, 5, 5), amean([5, 5, 5])
```

## Does this make any sense?

The idea behind this suggestion is logistic regression with an interaction term.
Let me explain where that comes from.
OP explained:

> The way we calculate individual risk for a child is 
>
> `risk = likelihood + impact` (`R=L+I`), so
>
> `L4 + I5 = R9`

At first I thought it was strange to add a likelihood and an impact score, 
Thinking about expected value, I thought it should be the product of a probability and a cost.
But if both are on a log scale, adding these logarithms is like multiplying probability by impact on a linear scale, so that makes more sense.

And if the scores are consistent with something like a log-odds scale, we can see a connection with logistic regression.
If `r1` and `r2` are risk scores, we can imagine a regression equation that looks like this, where `p` is the probability of an outcome like "missing from care":

`logit(p) = a r1 + b r2 + c r1 r2`

In this equation, `logit(p)` is the combined risk score, `a`, `b`, and `c` are unknown parameters, and the product `r1 r2` is an interaction term that captures the tendency of high risks to amplify each other.

With enough data, we could estimate the unknown parameters.
Without data, the best we can do is chose values that make the results consistent with expectations.

Since `r1` and `r2` are interchangeable, they have to have the same parameter.
And since the whole risk scale has an unspecified zero point, we can set it `a` and `b` to `1/2`.
Which means there is only one parameter left, the weight of the interaction term.

`logit(p) = (r_1 + r2) / 2 + k r1 r2`

Now we can see that the first term is the arithmetic mean and the second term is close to the geometric mean, but without the square root.

So the function I suggested -- the weighted sum of arithmetic and weighted means -- is not identical to the logistic model, but it is motivated by it.

With this rationale in mind, we might consider a revision: rather than add the likelihood and impact scores, and then compute the weighted sum of means, it might make more sense to separate likelihood and impact, compute the weighted sum of the means of the likelihoods, and then add back the impact.

## Computing by hand

In case the Python code makes it hard to see what's going on, let's work an example by hand.
Suppose `r1` is 9 and `r2` is 12.

```python
r1 = 9
r2 = 12
```

Here's the arithmetic mean.

```python
m1 = (9 + 12) / 2
m1
```

Here's the geometric mean.

```python
m2 = np.sqrt(9 * 12)
m2
```

And here's how we combine them.

```python
k = 1
combined_risk = m1 + k * (m2 - 4)
combined_risk
```

## Discussion

This question got my attention because OP is working on a challenging and important problem -- and they provided useful context.
It's an intriguing idea to define something that is intuitively like an average, but is not always bounded between the minimum and maximum of the data.

If we think strictly about generalized means, that's not possible. But if we think in terms of logarithms, regression, and interaction terms, we find a way.



[*Data Q&A: Answering the real questions with Python*](https://allendowney.github.io/DataQnA/index.html)

Copyright 2024–2026 [Allen B. Downey](https://allendowney.com)

License: [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/)
