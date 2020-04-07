# Linear Regression

1. Data

- generate a set of $`m`$ point pairs $`\{ (x^{(i)}, y^{(i)}) \}_{i = 1}^m`$ from random perturbations using `random` function based on a linear function that you define
- $`\hat{y} = a x + b`$ where $`a, b \in \mathbb{R}`$
- $`y = \hat{y} + n`$ where $`n \sim \mathcal{N}(0, \sigma^2)`$ is drawn from the normal distribution with mean $`0`$ and standard deviation $`\sigma`$
- you can choose $`m, a, b`$ and $`\sigma > 0`$

2. Linear Model

- $`h_\theta(x) = \theta_0 + \theta_1 x`$, $`\quad`$ where $`\theta = (\theta_0, \theta_1)`$ and $`\theta_0, \theta_1 \in \mathbb{R}`$

3. Objective Function

- $`J(\theta) = \frac{1}{2 m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2`$

4. Gradient Descent
 
- $`\theta_0^{(t+1)} \coloneqq \theta_0^{(t)} - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})`$
- $`\theta_1^{(t+1)} \coloneqq \theta_1^{(t)} - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x^{(i)}`$
- you can choose a step-size (learning rate) $`\alpha > 0`$ in $`\mathbb{R}`$
- you can choose any initial conditions for $`\theta_0^{(0)}`$ and $`\theta_1^{(0)}`$

## Submission

### 1. Codes, Comments and Results

_PDF file that is exported from Notebook including codes, comments, and results for the above problems at `github`_

#### [Plotting the results]

1. Input data [2pt]
- a straight line that is the graph of a linear function (in blue color)
- a set of points that have random perturbations with respect to the straight line (in black color)

2. Output results [2pt]
- the set of points that have random perturbations with respect to the straight line (in black color)
- a straight line that is the graph of a solution obtained by linear regression (in red color)

3. Plotting the energy values [3pt]
- the value of the objective function at every optimization step by the gradient descent algorithm (in blue color)
- the optimization should be performed until convergence

4. Plotting the model parameters [3pt]
- the value of the model parameters $`\theta_0`$ and $`\theta_1`$ at every optimization step (in red ($`\theta_0`$) and blue ($`\theta_1`$) colors)
- the optimization should be performed until convergence

### 2. Commit History

_PDF file that is exported from the commit history at github_

#### [Apply `commit` at the end of the implementations at least for the following steps]

You can `commit` as many as you want and the order of commits does not matter at all, but you have to make meaningful and proper comments at commit message

1. Plotting the input data [1pt]
2. Plotting the output results [1pt]
3. Plotting the energy values [1pt]
4. Plotting the model parameters [1pt]


