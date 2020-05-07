---
layout: post
title:  "Testing Mathjax"
date:   "2020-05-07 13:26:51"
categories: jekyll mathjax
author: noukoudashisoup
---
This is our very first post for our testing purpose. 

The formula $$E=mc^2$$ is well known. 
But another famous formula (in our unit) is the following

$$ 
f(x) = \langle k(x,\cdot), f\rangle. 
$$

This equation is known as the reproducing property of an reproducing kernel $$k$$. 

Let $k_x(y) = k(x, y)$. Then, the kernel function is evaluated as 

$$
\begin{align}
k(x, y) &= \langle k_x(\cdot), k(y, \cdot)\rangle\\
    &= \langle k(x, \cdot), k(y, \cdot) \rangle. 
\end{align}
$$