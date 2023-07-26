---
layout: single
title: "Discrete error and convergence"
header:

categories:
    - Research
tags:
    - Research
    - Publish
    - Article writing
toc: true
toc_label: "Table of contents"
toc_icon: "heart"
---

I was asked a question today about a typical way to present the error between a numerical scheme and an exact solution, and the convergence of the method. I will demonstrate one method that is typically used based on the $L_{2}$-norm of the norm.

Consider the ODE
$$
\frac{dx}{dt} = x(t) \qquad x(0) = 1
$$
which has the analytic solution