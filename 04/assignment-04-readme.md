# Linear regression with multiple variables

#### 1. Data

- load a set of data points $`\{ (x^{(i)}, y^{(i)}, z^{(i)}, h^{(i)}) \}`$ from the given CSV file ('data_train.csv') for training
- load a set of data points $`\{ (x^{(i)}, y^{(i)}, z^{(i)}, h^{(i)}) \}`$ from the given CSV file ('data_test.csv') for testing

#### 2. Linear Model

- $`f_\theta(x, y, z) = \theta_0 + \theta_1 x + \theta_2 y + \theta_3 z`$, $`\quad`$ where $`\theta = (\theta_0, \theta_1, \theta_2, \theta_3)`$ and $`\theta_0, \theta_1, \theta_2, \theta_3 \in \mathbb{R}`$

#### 3. Objective Function

- $`J(\theta_0, \theta_1, \theta_2, \theta_3) = \frac{1}{2 m} \sum_{i=1}^m ( \theta_0 + \theta_1 x^{(i)} + \theta_2 y^{(i)} + \theta_3 z^{(i)} - h^{(i)} )^2`$

#### 4. Gradient Descent
 
- $`\theta_0^{(t+1)} \coloneqq \theta_0^{(t)} - \alpha \frac{1}{m} \sum_{i=1}^{m} (f_\theta(x^{(i)}, y^{(i)}, z^{(i)}) - h^{(i)})`$
- $`\theta_1^{(t+1)} \coloneqq \theta_1^{(t)} - \alpha \frac{1}{m} \sum_{i=1}^{m} (f_\theta(x^{(i)}, y^{(i)}, z^{(i)}) - h^{(i)}) x^{(i)}`$
- $`\theta_2^{(t+1)} \coloneqq \theta_2^{(t)} - \alpha \frac{1}{m} \sum_{i=1}^{m} (f_\theta(x^{(i)}, y^{(i)}, z^{(i)}) - h^{(i)}) y^{(i)}`$
- $`\theta_3^{(t+1)} \coloneqq \theta_3^{(t)} - \alpha \frac{1}{m} \sum_{i=1}^{m} (f_\theta(x^{(i)}, y^{(i)}, z^{(i)}) - h^{(i)}) z^{(i)}`$
- you should choose a learning rate $`\alpha`$ in such a way that the convergence is achieved
- you can use any initial conditions $`(\theta_0^{(0)}, \theta_1^{(0)}, \theta_2^{(0)}, \theta_3^{(0)})`$
 
#### 5. Training

- find optimal parameters $`(\theta_0, \theta_1, \theta_2, \theta_3)`$ using the training dataset ('data_train.csv')

#### 6. Testing

- evaluate the inference using the testing dataset ('data_test.csv')
- use the objective function $`J(\theta_0, \theta_1, \theta_2, \theta_3)`$ for measuring the dissimilarity between the expected value and the inference using the testing data

## Code

- load CSV file

``` python
import csv

with open('data_train.csv', newline='') as myfile:
    reader  = csv.reader(myfile, delimiter=',')
    ct = 1 
    for i in reader:
        print('[', ct, ']', 'x =', i[0], ', y = ', i[1], ', z = ', i[2], ', h = ', i[3])
        ct += 1

with open('data_test.csv', newline='') as myfile:
    reader  = csv.reader(myfile, delimiter=',')
    ct = 1 
    for i in reader:
        print('[', ct, ']', 'x =', i[0], ', y = ', i[1], ', z = ', i[2], ', h = ', i[3])
        ct += 1
```
 
## Submission

### 1. Codes, Comments and Results

_PDF file that is exported from Notebook including codes, comments, and results for the above problem using Jupyter Notebook or Colab_

#### [Plotting the results]

##### 1. Plot the estimated parameters using the training dataset [3pt]
- plot the estimated parameters $`\{ (\theta_0, \theta_1, \theta_2, \theta_3) \}`$ at every iteration of gradient descent until convergence
- the colors for the parameters $`\{ (\theta_0, \theta_1, \theta_2, \theta_3) \}`$ should be black, red, green, blue, respectively
- the optimization is performed using the training dataset ('data_train.csv')

##### 2. Plot the training error using the training dataset [4pt]
- plot the training error $`J(\theta_0, \theta_1, \theta_2, \theta_3)`$ at every iteration of gradient descent until convergence (in blue color)

##### 3. Plot the testing error using the testing dataset at every iteration of gradient descent until convergence [5pt]
- plot the testing error $`J(\theta_0, \theta_1, \theta_2, \theta_3)`$ at every iteration of gradient descent until convergence (in red color)

### 2. Commit History

_PDF file that is exported from the commit history at github_

#### [Apply `commit` at the end of the implementations at least for the following steps]

You can `commit` as many as you want and the order of commits does not matter, but you have to make meaningful and proper comments for commit messages

1. Plotting the estimated parameters [1pt]
2. Plotting the training error [1pt]
3. Plotting the testing error [1pt]