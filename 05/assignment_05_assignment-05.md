# Logistic regression for a binary classification

#### 1. Training Data

- load the training data file ('data.txt')
- each row $`\{ (x^{(i)}, y^{(i)}, l^{(i)}) \}`$ of the data consists of a 2-dimensional point $`(x, y)`$ with its label $`l`$
- $`x, y \in \mathbb{R}`$ and $`l \in \{0, 1\}`$

#### 2. Logistic regression

- $`\hat{h} = \sigma(z)`$ 
- $`z = \theta_0 + \theta_1 x + \theta_2 y`$, where $`\theta_0, \theta_1, \theta_2 \in \mathbb{R}`$
- $`\sigma(z) = \frac{1}{1 + \exp(-z)}`$
- $`\sigma^{\prime}(z) = \sigma(z) (1 - \sigma(z))`$

#### 3. Objective Function

- $`J(\theta_0, \theta_1, \theta_2) = \frac{1}{m} \sum_{i=1}^m ( - l^{(i)} \log(\sigma(z^{(i)})) - (1 - l^{(i)}) \log(1 - \sigma(z^{(i)}))  )`$

#### 4. Gradient Descent

- $`\theta_0^{(t+1)} \coloneqq \theta_0^{(t)} - \alpha \frac{1}{m} \sum_{i=1}^{m} (\sigma(z^{(i)}) - l^{(i)})`$
- $`\theta_1^{(t+1)} \coloneqq \theta_1^{(t)} - \alpha \frac{1}{m} \sum_{i=1}^{m} (\sigma(z^{(i)}) - l^{(i)}) x^{(i)}`$
- $`\theta_2^{(t+1)} \coloneqq \theta_2^{(t)} - \alpha \frac{1}{m} \sum_{i=1}^{m} (\sigma(z^{(i)}) - l^{(i)}) y^{(i)}`$
- you should choose a learning rate $`\alpha`$ in such a way that the convergence is achieved
- you can use any initial conditions $`(\theta_0^{(0)}, \theta_1^{(0)}, \theta_2^{(0)})`$
 
#### 5. Training

- find optimal parameters $`(\theta_0, \theta_1, \theta_2)`$ using the training data

#### 6. Visualize Classifier

- visualize the obtained classifier with varying $`x`$ and $`y`$ values that range from the minimum to the maximum of the training data

## Code

- load the training data from the file and plot the points

``` python
import numpy as np
import matplotlib.pyplot as plt

data    = np.genfromtxt("data.txt", delimiter=',')

x       = data[:, 0]
y       = data[:, 1]
label   = data[:, 2]

x_label0    = x[label == 0]
x_label1    = x[label == 1]

y_label0    = y[label == 0]
y_label1    = y[label == 1]

plt.figure(figsize=(8, 8))
plt.scatter(x_label0, y_label0, alpha=0.3, c='b')
plt.scatter(x_label1, y_label1, alpha=0.3, c='r')
plt.show()
```
 
## Submission

### 1. Codes, Comments and Results

_PDF file that is exported from Notebook including codes, comments, and results for the above problem using Jupyter Notebook or Colab_

#### [Plotting the results]

##### 1. Plot the training data [2pt]
- plot the training data points $`(x, y)`$ with their labels $`l`$ in colors (blue for label 0 and red for label 1)

##### 2. Plot the estimated parameters [3pt]
- plot the estimated parameters $`(\theta_0, \theta_1, \theta_2)`$ at every iteration of gradient descent until convergence
- the colors for the parameters $`(\theta_0, \theta_1, \theta_2)`$ should be red, green, blue, respectively

##### 3. Plot the training error [3pt]
- plot the training error $`J(\theta_0, \theta_1, \theta_2)`$ at every iteration of gradient descent until convergence (in blue color)

##### 4. Plot the obtained classifier [4pt]
- plot the classifier $`\sigma(z)`$ where $`z = \theta_0 + \theta_1 x + \theta_2 y`$ with $`x = [30 : 0.5 : 100]`$ and $`y = [30 : 0.5 : 100]`$
- $`[a : t : b]`$ denotes a range of values from $`a`$ to $`b`$ with a stepsize $`t`$
- use a colormap where blue is used for 0, red is used for 1, their weighted combination for a value between 0 and 1
- plot the training data points $`(x, y)`$ with their labels $`l`$ in colors (blue for label 0 and red for label 1) superimposed on the classifier

### 2. Commit History

_PDF file that is exported from the commit history at github_

#### [Apply `commit` at the end of the implementations at least for the following steps]

You can `commit` as many as you want and the order of commits does not matter, but you have to make meaningful and proper comments for commit messages

1. Plot the training data [1pt]
2. Plot the estimated parameters [1pt]
3. Plot the training error [1pt]
4. Plot the obtained classifier [1pt]