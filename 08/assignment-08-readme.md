# Forward Propagation in the Neural Networks

#### 1. Input Data

- load the data file ('mnist_test.csv')
- each row of the data consists of the label and the image pixel values in a vector form
- the label is one of the 10 digits from 0 to 9
- the image represents one of the 10 digits from 0 to 9 in grey scale and its size is 28x28

#### 2. Average Image for Each Digit

- take the arithmatic average of the image data with the same label for each digit

#### 3. Forward Propagation with Random Weights

- consider a neural network with a fully connected layer using a logistic unit
- the weights of the fully connected layer are given by random numbers sampled from the Normal distribution $`\mathcal{N}(0, 1)`$ with mean 0 and standard deviation 1
- compute the average of the output of the neural network using a logistic unit for the images of the same label for each digit

## Code

- load the data from the file and plot the images

``` python
import matplotlib.pyplot as plt
import numpy as np

file_data   = "mnist_test.csv"
handle_file = open(file_data, "r")
data        = handle_file.readlines()
handle_file.close()

size_row    = 28    # height of the image
size_col    = 28    # width of the image

num_image   = len(data)
count       = 0     # count for the number of images

#
# make a matrix each column of which represents an images in a vector form 
#
list_image  = np.empty((size_row * size_col, num_image), dtype=float)
list_label  = np.empty(num_image, dtype=int)

for line in data:

    line_data   = line.split(',')
    label       = line_data[0]
    im_vector   = np.asfarray(line_data[1:])

    list_label[count]       = label
    list_image[:, count]    = im_vector    

    count += 1

# 
# plot first 100 images out of 10,000 with their labels
# 
f1 = plt.figure(1)

for i in range(100):

    label       = list_label[i]
    im_vector   = list_image[:, i]
    im_matrix   = im_vector.reshape((size_row, size_col))

    plt.subplot(10, 10, i+1)
    plt.title(label)
    plt.imshow(im_matrix, cmap='Greys', interpolation='None')

    frame   = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)

plt.show()
```

## Submission

### 1. Codes, Comments and Results

_PDF file that is exported from Notebook including codes, comments, and results for the above problem using Jupyter Notebook or Colab_

##### 1. Plot the average image [5pt]
- plot the average images of the same label for each digit
- arrange the subplots in 2x5 array for the 10 average images and present the label at the title of each subplot in the increasing order of the label

##### 2. Present the output of the neural network with random weights [9pt]
- consider a neural network with a fully connected layer using a logistic unit without a bias
- assign random values from the normal distribution $`\mathcal{N}(0, 1)`$ with mean 0 and standard deviation 1 to the weights of the fully connected layer using a logistic unit without a bias
- compute the forward propagation and take the average of the output values for the images of the same label
- present the average values for each label in the increasing order of the label

### 2. Commit History [1pt]

_PDF file that is exported from the commit history at github_

You should `git commit` at least 4 times with meaningful and proper commit messages in such a way that you can demonstrate the progress of your programming in an effective way