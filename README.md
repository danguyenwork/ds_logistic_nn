# Logistic classifier neural network

This is a neural network implementation using a logistic cost function.

## Cost function

The cost function for logistic regression is:

$$l = ylog(1-a) + (1-y)log(1-a)$$

where a = $\sigma(z)$ and $z = wx + b$, where

- $a$ is the predicted value (0, 1)
- $w$ is the set of parameters for the set of features x
- $b$ is the intercept
- $\sigma$ is the sigmoid function that maps z to 0 or 1

This cost function works because it produces a convex shape that has one global minima.

For $y = 1$, a is expected to be very large to minimize it, which means a needs to be equal to 1.

The same apply for when $y=0$. a is expected to be very small to minimize the function, which means a needs to be equal to 0.

So when our cost function is 0, $a = y$, which is the goal of a classifier.

The cost function of all training examples $m$ is:

$$J(w,b) = \frac{1}{m}\sum_{i+1}^m l(a_i, y_i)$$

## Partial Derivative

We have two sets of functions:

$$l = ylog(1-a) + (1-y)log(1-a)$$
$$a = \sigma(z) = \frac{1}{1+e^{-z}}$$

Using calculus, we know that:

$$\frac{\delta l}{\delta z} =  \frac{\delta l}{\delta a}\frac{\delta a}{\delta z}$$

Performing these partial derivatives gives:

$$\frac{\delta l}{\delta a} = \frac{1-y}{1-a} -\frac{y}{a}$$

$$\frac{\delta a}{\delta z} = -(1+e^{-z})^{-2} * -e^{-z} = \frac{1}{1+e^{-z}} * \frac{e^{-z}}{1+e^{-z}} = a(1-a) $$

Therefore:

$$\frac{\delta l}{\delta z} = (\frac{1-y}{1-a} -\frac{y}{a}) * a(1-a) = a - y$$

## Gradient descent

Gradient descent is a first-order iterative optimization algorithm for finding the minimum of a function. To find a local minimum of a function using gradient descent, one takes steps proportional to the negative of the gradient (or of the approximate gradient) of the function at the current point.

In this case, we will find the gradient (or derivative) of each parameter with respect to our training examples, i.e. $\delta w_1, \delta w_2$, and subtract an amount of that $ * $ the learning rate $\alpha$ to iteratively arrive at the optimal values of $w_i$.

Shape:

```python
X.shape: n x m (features x size)
w.shape: n x 1 (features x 1)
Y.shape: 1 x m (1 x size)
b: scalar

z.shape: w.T * X = (1 x n) x (n x m) = 1 x m
a.shape: sigmoid(z) = 1 x m

dz.shape: A - Y = 1 x m
dw.shape: X * dz.T = (n x m) x (m x 1) = n x 1

db: scalar
```

Here is the pseudo-code:


```python
for _ in max_iter: # loop to iterate max_iter times
  dw, db = 0, 0, 0, 0 # initialize the cumulative partial derivatives for this iteration.

  Z = w.T * x + b # calculate z using current parameter set w
  A = 1 / (1 + e^-z) # calculate a
  dz = A - Y # calculate the partial derivative of cost function to z
  dw = 1/m * (X * dz.T) # calculate the cumulative derivatives each parameter
  db = mean(dz) # update the cumulative derivatives for intercept

  # update new values of w_1, w_2 and b by the averaged derivative multipled by a learning rate
  w := w - alpha * dw
  b := b - alpha * db
```
