---
title: "Least Squares Fitting: From Mathematical Foundations to Machine Learning"
date: 2026-09-12
permalink: /posts/2026/09/least-squares-fitting/
categories:
  - Mathematics
  - Machine Learning
tags:
  - least squares
  - linear regression
  - numerical methods
  - machine learning
toc: true
toc_label: "Table of contents"
---

Suppose we measure a quantity at several points. The observations do not lie
perfectly on a line: experiments contain noise, models are imperfect, and the
world is more complicated than our equations. We therefore need a principled
way to choose the line—or, more generally, the model—that best represents the
data.

Least squares provides that principle. It is simple enough to derive with
calculus, rich enough to reveal the geometry of data fitting, and important
enough to sit near the foundation of statistics, numerical analysis, and
machine learning.

This article develops the method from first principles, derives its analytical
solution, implements it in Python, and explains why its basic pattern appears
throughout modern machine learning.

## From observations to a mathematical model

Assume that we have $n$ observations,

$$
(x_1,y_1), (x_2,y_2), \ldots, (x_n,y_n),
$$

and we want to describe their relationship with a straight line,

$$
\hat{y}_i = \beta_0 + \beta_1 x_i.
$$

Here, $\hat{y}_i$ is the value predicted by the line, $\beta_0$ is its
intercept, and $\beta_1$ is its slope. The difference between an observation
and its prediction is the **residual**,

$$
r_i = y_i - \hat{y}_i.
$$

Unless every point happens to lie on one line, no choice of $\beta_0$ and
$\beta_1$ makes all residuals zero. The fitting problem is therefore an
optimization problem: which parameters make the residuals collectively as
small as possible?

## Why minimize squared residuals?

The least-squares objective is

$$
S(\beta_0,\beta_1)
= \sum_{i=1}^{n}\left[y_i-(\beta_0+\beta_1x_i)\right]^2.
$$

Squaring is not the only possible choice, but it has several useful
properties:

1. **Positive and negative errors cannot cancel.** Every squared residual is
   non-negative.
2. **Large errors receive more weight.** Doubling a residual multiplies its
   contribution by four.
3. **The objective is smooth.** It can be differentiated, which gives an exact
   solution for linear models and efficient optimization algorithms for more
   complicated ones.
4. **It has a statistical interpretation.** If measurement errors are
   independent Gaussian random variables with constant variance, minimizing
   the sum of squared residuals is equivalent to maximizing the likelihood of
   the observations.
5. **It has a geometric interpretation.** The fitted values are the orthogonal
   projection of the observation vector onto the space spanned by the model's
   features.

These properties make least squares mathematically convenient and practically
useful. They also reveal its limitations: because large residuals are squared,
an outlier can strongly influence the fit.

## Analytical solution for a straight line

To find the best line, differentiate $S$ with respect to both unknown
parameters and set the derivatives to zero:

$$
\frac{\partial S}{\partial \beta_0}
= -2\sum_{i=1}^{n}(y_i-\beta_0-\beta_1x_i)=0,
$$

$$
\frac{\partial S}{\partial \beta_1}
= -2\sum_{i=1}^{n}x_i(y_i-\beta_0-\beta_1x_i)=0.
$$

These are the **normal equations** for simple linear regression. Solving them
gives

$$
\hat{\beta}_1 =
\frac{\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})}
{\sum_{i=1}^{n}(x_i-\bar{x})^2},
$$

and

$$
\hat{\beta}_0 = \bar{y}-\hat{\beta}_1\bar{x},
$$

where $\bar{x}$ and $\bar{y}$ are the sample means. The slope is the
co-variation of $x$ and $y$, normalized by the variation in $x$. The intercept
then places the fitted line through the point $(\bar{x},\bar{y})$.

The hats distinguish estimated parameters from the unknown parameters of an
ideal data-generating process.

## The matrix formulation

The real power of least squares becomes clearer when a model has several input
variables. Collect the observations into a vector $\mathbf{y}$ and the model
parameters into a vector $\boldsymbol{\beta}$. Put the input features into a
design matrix $\mathbf{X}$:

$$
\mathbf{y} =
\begin{bmatrix}
y_1 \\ y_2 \\ \vdots \\ y_n
\end{bmatrix},
\qquad
\boldsymbol{\beta} =
\begin{bmatrix}
\beta_0 \\ \beta_1 \\ \vdots \\ \beta_p
\end{bmatrix},
$$

$$
\mathbf{X} =
\begin{bmatrix}
1 & x_{11} & \cdots & x_{1p} \\
1 & x_{21} & \cdots & x_{2p} \\
\vdots & \vdots & \ddots & \vdots \\
1 & x_{n1} & \cdots & x_{np}
\end{bmatrix}.
$$

The first column of ones represents the intercept. Predictions and residuals
are now

$$
\hat{\mathbf{y}}=\mathbf{X}\boldsymbol{\beta},
\qquad
\mathbf{r}=\mathbf{y}-\mathbf{X}\boldsymbol{\beta}.
$$

We minimize the squared Euclidean length of the residual vector:

$$
J(\boldsymbol{\beta})
=\frac{1}{2}\left\|\mathbf{y}-\mathbf{X}\boldsymbol{\beta}\right\|_2^2.
$$

The factor $1/2$ does not change the minimizer; it only cancels the factor $2$
that appears during differentiation. The gradient is

$$
\nabla_{\boldsymbol{\beta}}J
=\mathbf{X}^{\mathsf T}
(\mathbf{X}\boldsymbol{\beta}-\mathbf{y}).
$$

At the minimum the gradient is zero, so

$$
\mathbf{X}^{\mathsf T}\mathbf{X}\hat{\boldsymbol{\beta}}
=\mathbf{X}^{\mathsf T}\mathbf{y}.
$$

If $\mathbf{X}^{\mathsf T}\mathbf{X}$ is invertible, the analytical solution
is

$$
\boxed{
\hat{\boldsymbol{\beta}}
=(\mathbf{X}^{\mathsf T}\mathbf{X})^{-1}
\mathbf{X}^{\mathsf T}\mathbf{y}
}.
$$

If the columns of $\mathbf{X}$ are linearly dependent, the inverse does not
exist and the parameters are not uniquely identifiable. The Moore–Penrose
pseudoinverse gives the minimum-norm least-squares solution,

$$
\hat{\boldsymbol{\beta}}=\mathbf{X}^{+}\mathbf{y}.
$$

In numerical code, we should not explicitly calculate
$(\mathbf{X}^{\mathsf T}\mathbf{X})^{-1}$. Forming the product can worsen the
conditioning of the problem, and computing an inverse does unnecessary work.
Stable solvers instead use QR factorization or singular value decomposition.

## The geometry: fitting is a projection

Every possible prediction $\mathbf{X}\boldsymbol{\beta}$ lies in the column
space of $\mathbf{X}$. Usually the observation vector $\mathbf{y}$ does not lie
in this space, so it cannot be reproduced exactly.

Least squares chooses the point in the column space closest to $\mathbf{y}$.
At that point the residual is perpendicular to every column of $\mathbf{X}$:

$$
\mathbf{X}^{\mathsf T}\mathbf{r}=0.
$$

Substituting $\mathbf{r}=\mathbf{y}-\mathbf{X}\hat{\boldsymbol{\beta}}$
produces the normal equations again. The calculus and geometry are therefore
two views of the same fact: there is no direction inside the model space that
can further reduce the error.

## A complete Python example

The following example fits a line to six noisy observations. NumPy's
`lstsq` routine uses a stable linear-algebra solver and is preferable to
explicitly forming the inverse in the boxed equation.

```python
import numpy as np
import matplotlib.pyplot as plt

# Observed data
x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([1.1, 2.9, 5.2, 6.8, 9.1, 11.0])

# Design matrix: the first column fits the intercept.
X = np.column_stack([np.ones_like(x), x])

# Solve min ||X beta - y||_2 without explicitly computing an inverse.
beta, residual_sum, rank, singular_values = np.linalg.lstsq(
    X, y, rcond=None
)
intercept, slope = beta

# Predictions and diagnostics
y_hat = X @ beta
residuals = y - y_hat
mse = np.mean(residuals**2)
r_squared = 1.0 - np.sum(residuals**2) / np.sum((y - y.mean())**2)

print(f"intercept = {intercept:.4f}")
print(f"slope     = {slope:.4f}")
print(f"MSE       = {mse:.4f}")
print(f"R^2       = {r_squared:.4f}")

# Plot the observations and fitted line.
x_plot = np.linspace(x.min(), x.max(), 200)
y_plot = intercept + slope * x_plot

plt.scatter(x, y, color="tab:blue", label="observations")
plt.plot(x_plot, y_plot, color="tab:red", label="least-squares fit")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.show()
```

The numerical output is

```text
intercept = 1.0381
slope     = 1.9914
MSE       = 0.0178
R^2       = 0.9985
```

Thus the fitted relationship is approximately

$$
\hat{y}=1.0381+1.9914x.
$$

The mean squared error (MSE) is the average squared residual. The coefficient
of determination $R^2$ compares the residual variation with the variation of
the observations around their mean. An $R^2$ near one indicates a close fit to
this dataset, but it does not prove causation or guarantee good predictions on
new data.

## From an analytical solution to gradient descent

For linear regression with a moderate number of features, a direct
least-squares solver is excellent. Machine-learning problems, however, may
contain millions of observations or models for which no closed-form solution
exists. The same objective can then be minimized iteratively.

For the mean squared error

$$
J(\boldsymbol{\beta})
=\frac{1}{2n}\left\|\mathbf{X}\boldsymbol{\beta}-\mathbf{y}\right\|_2^2,
$$

the gradient is

$$
\nabla J
=\frac{1}{n}\mathbf{X}^{\mathsf T}
(\mathbf{X}\boldsymbol{\beta}-\mathbf{y}).
$$

Gradient descent repeatedly moves the parameters in the opposite direction:

$$
\boldsymbol{\beta}^{(k+1)}
=\boldsymbol{\beta}^{(k)}-\eta\nabla J,
$$

where $\eta$ is the learning rate. A minimal implementation is:

```python
beta_gd = np.zeros(X.shape[1])
learning_rate = 0.05

for _ in range(5_000):
    error = X @ beta_gd - y
    gradient = X.T @ error / len(y)
    beta_gd -= learning_rate * gradient

print(beta_gd)  # approximately [1.0381, 1.9914]
```

The direct solution and gradient descent reach the same minimum because the
least-squares objective is a convex quadratic. In more complex models, the
loss surface may no longer be convex, but the pattern remains: define a model,
measure its error, compute a gradient, and update its parameters.

## Why least squares is foundational to machine learning

Least squares contains the main ingredients of supervised machine learning in
their clearest form:

- **Data:** rows of $\mathbf{X}$ are examples and columns are features.
- **A parameterized model:** $\mathbf{X}\boldsymbol{\beta}$ maps inputs to
  predictions.
- **A loss function:** squared error measures disagreement between predictions
  and targets.
- **Training:** optimization selects parameters that minimize the loss on the
  observed examples.
- **Evaluation:** residuals, MSE, and performance on unseen data tell us how
  well the learned relationship generalizes.

The same structure extends immediately beyond fitting a straight line. We can
add polynomial terms such as $x^2$ and $x^3$, interaction terms such as
$x_1x_2$, or transformed features. The model may look nonlinear as a function
of the inputs while remaining linear in its fitted parameters, so ordinary
least squares still applies.

Least squares also introduces ideas that reappear across machine learning:

- **Regularization:** ridge regression adds a penalty
  $\lambda\|\boldsymbol{\beta}\|_2^2$ to discourage excessively large
  parameters. Its solution is

  $$
  \hat{\boldsymbol{\beta}}_{\text{ridge}}
  =(\mathbf{X}^{\mathsf T}\mathbf{X}+\lambda\mathbf{I})^{-1}
  \mathbf{X}^{\mathsf T}\mathbf{y}.
  $$

- **Bias–variance trade-off:** a model that is too simple underfits; a model
  with too many flexible features may fit the training noise and fail on new
  data.
- **Feature engineering:** the columns chosen for $\mathbf{X}$ determine what
  relationships the model can represent.
- **Optimization:** gradient descent, mini-batches, and learning rates are
  central to training neural networks.
- **Probabilistic modeling:** choosing a loss function often corresponds to an
  assumption about the distribution of the errors.

Deep neural networks are not generally solved by the normal equations, and
classification models often use losses other than squared error. Nevertheless,
their training follows the same intellectual template made visible by least
squares: choose a family of functions, quantify error, optimize parameters,
and test whether the learned pattern generalizes.

## When ordinary least squares is not enough

Least squares is a starting point, not a universal answer. Its assumptions and
failure modes should guide the next choice:

- **Outliers:** squared errors give extreme observations great influence.
  Robust losses such as the absolute-error or Huber loss may be better.
- **Multicollinearity:** strongly correlated features make parameter estimates
  unstable. Regularization or dimensionality reduction can help.
- **Non-constant noise:** if the error variance changes across observations,
  weighted least squares may be more appropriate.
- **Nonlinear structure:** new features or a nonlinear model may be necessary.
- **Overfitting:** training error alone is not enough; validation or test data
  are needed to estimate generalization.
- **Causality:** a fitted association does not establish that changing one
  variable will cause the other to change.

The residuals are therefore not merely leftovers. Plotting and studying them
can reveal curvature, changing variance, outliers, dependence, or missing
features—evidence that the model should be reconsidered.

## Final perspective

Least squares turns an imprecise request—"find the best fit"—into a precise
mathematical problem. Calculus gives the normal equations, linear algebra gives
the projection viewpoint, probability explains when the objective is
statistically natural, and numerical methods show how to compute the solution
reliably.

Its deeper importance is methodological. Machine learning is not magic hidden
inside an algorithm. It begins with the same sequence visible in least squares:
represent the data, state the model, define what error means, optimize, and
check the result on observations the model has not seen.

That is why a line fitted through a handful of noisy points is more than a
classroom exercise. It is a compact introduction to how machines learn from
data.
