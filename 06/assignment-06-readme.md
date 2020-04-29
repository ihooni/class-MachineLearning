# Logistic regression for a binary classification with a non-linear classification boundary

#### 1. Training Data

- load the training data file ('data-nonlinear.txt')
- each row $`\{ (x^{(i)}, y^{(i)}, l^{(i)}) \}`$ of the data consists of a 2-dimensional point $`(x, y)`$ with its label $`l`$
- $`x, y \in \mathbb{R}`$ and $`l \in \{0, 1\}`$

#### 2. Logistic regression

- $`\hat{h} = \sigma(z)`$ 
- $`z = g(x, y; \theta)`$, where $`g`$ is a high dimensional function and $`\theta \in \mathbb{R}^{k}`$
- $`\theta = (\theta_{0}, \theta_{1}, \cdots, \theta_{k-1})`$
- $`g(x, y ; \theta) = \theta_{0} f_{0}(x, y) + \theta_{1} f_{1}(x, y) + \cdots + \theta_{k-1} f_{k-1}(x, y)`$
- $`f_{k}(x, y)`$ be any high dimensional function of $`x`$ and $`y`$
- $`\sigma(z) = \frac{1}{1 + \exp(-z)}`$
- $`\sigma^{\prime}(z) = \sigma(z) (1 - \sigma(z))`$
- the dimension $`k`$ of $`\theta`$ can be 16, but it can be less than that. you can choose $`k`$ for the best performance

#### 3. Objective Function

- $`J(\theta) = \frac{1}{m} \sum_{i=1}^m ( - l^{(i)} \log(\sigma(g(x^{(i)}, y^{(i)}; \theta))) - (1 - l^{(i)}) \log(1 - \sigma(g(x^{(i)}, y^{(i)}; \theta)))  )`$

#### 4. Gradient Descent

- $`\theta_{k}^{(t+1)} \coloneqq \theta_0^{(t)} - \alpha \frac{1}{m} \sum_{i=1}^{m} (\sigma(g(x^{(i)}, y^{(i)}; \theta)) - l^{(i)}) \frac{\partial g(x^{(i)}, y^{(i)}; \theta^{(t)})}{\partial \theta_{k}}`$, for all $`k`$
- you should choose a learning rate $`\alpha`$ in such a way that the convergence is achieved
- you can use any initial conditions $`\theta_k^{(0)}`$ for all $`k`$
 
#### 5. Training

- find optimal parameters $`\theta`$ using the training data

#### 6. Compute the training accuracy

- the accuracy is computed by $`\frac{\textrm{number of correct predictions}}{\textrm{total number of predictions}}`$

## Code

- load the training data from the file and plot the points

``` python
import numpy as np
import matplotlib.pyplot as plt

data    = np.genfromtxt("data-nonlinear.txt", delimiter=',')

pointX  = data[:, 0]
pointY  = data[:, 1]
label   = data[:, 2]

pointX0 = pointX[label == 0]
pointY0 = pointY[label == 0]

pointX1 = pointX[label == 1]
pointY1 = pointY[label == 1]

plt.figure()
plt.scatter(pointX0, pointY0, c='b')
plt.scatter(pointX1, pointY1, c='r')
plt.tight_layout()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
```

<img src="data-nonlinear.png"  width="500">

## Submission

### 1. Codes, Comments and Results

_PDF file that is exported from Notebook including codes, comments, and results for the above problem using Jupyter Notebook or Colab_

#### [Plotting the results]

##### 1. Plot the training data [1pt]
- plot the training data points $`(x, y)`$ with their labels $`l`$ in colors (blue for label 0 and red for label 1)

##### 2. Write down the high dimensional function $`g(x, y; \theta)`$ [2pt]
- write down the equation for the non-linear function $`g(x, y; \theta)`$ used for the classifier in LaTeX format

##### 3. Plot the training error [3pt]
- plot the training error $`J(\theta)`$ at every iteration of gradient descent until convergence (in blue color)

##### 4. Plot the training accuracy [3pt]
- plot the training accuracy at every iteration of gradient descent until convergence (in red color)
- the score will be given depending on the accuracy

##### 5. Write down the final training accuracy [5pt]
- present the final training accuracy in number (%) at convergence
- the score will be given depending on the accuracy

##### 6. Plot the optimal classifier superimposed on the training data [5pt]
- plot the boundary of the optimal classifier at convergence (in green color)
- the boundary of the classifier is defined by $`\{ (x, y) \mid \sigma(g(x, y ; \theta)) = 0.5 \} = \{ (x, y) \mid g(x, y ; \theta) = 0 \}`$
- plot the training data points $`(x, y)`$ with their labels $`l`$ in colors superimposed on the illustration of the classifier (blue for label 0 and red for label 1)
- you can use `contour` function in python3
- the score will be given depending on the accuracy

### 2. Commit History

_PDF file that is exported from the commit history at github_

#### [Apply `commit` at the end of the implementations at least for the following steps]

You should `git commit` at least 4 times with meaningful and proper commit messages in such way that you can demonstrate the progress of your programming in an effective way [1pt]