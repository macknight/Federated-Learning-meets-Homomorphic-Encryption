# -*- coding: utf-8 -*-
"""## Datasets"""

import pandas as pd
import os

# we use another datasets to demo our Logistic Regression
df = pd.read_csv("data/breast-cancer.csv")

# define function to swap columns
def swap_columns(df, col1, col2):
    col_list = list(df.columns)
    x, y = col_list.index(col1), col_list.index(col2)
    col_list[y], col_list[x] = col_list[x], col_list[y]
    df = df[col_list]
    return df

# swap "diagnosis" and "fractal_dimension_worst" columns
df = swap_columns(df, 'diagnosis', 'fractal_dimension_worst')

print(df.head())

"""## Data Preprocessing"""

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from torch.autograd import Variable

"""Replace "M" with 1 and "B" with 0 at "diagnosis" column"""

# replace "M" with 1 and "B" with 0
df["diagnosis"] = (df["diagnosis"] == "M").astype(int)

"""We define some functions in order to randomly split this dataset to:
- Training dataset (80%)
- Testing dataset  (20%)
"""

def print_dataset(name, data):
    print('Dataset {}. Shape: {}'.format(name, data.shape))
    print(data[:5])

def scale_dataset(df, overSample=False):
    # split to fetures and diagnostic result
    X = df[df.columns[:-1]].values
    Y = df[df.columns[-1]].values

    # standardize the input features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # balance the class distribution
    if overSample:
        ros = RandomOverSampler()
        X, Y = ros.fit_resample(X, Y)

    data = np.hstack((X, np.reshape(Y, (-1, 1))))

    # convert to tensor context
    X_train_tensor = Variable(torch.tensor(X, dtype = torch.float32))
    Y_train_tensor = Variable(torch.tensor(Y, dtype = torch.float32))
    data_tensor    = Variable(torch.tensor(data, dtype = torch.float32))

    return data_tensor, X_train_tensor, Y_train_tensor

# split dataframe to train and test df
df_train, df_test = np.split(df.sample(frac=1), [int(0.8 * len(df))])

# scaling and convert to tensor context
train, X_train, Y_train = scale_dataset(df_train, True)
test , X_test , Y_test  = scale_dataset(df_test , False)
train

"""## Logistic Regression

We define our machine learning model, which is a logistic regression model. Why? Because this medical dataset is linearly separable, which simplifies things a lot.

We can create the logistic regression model with the following code (we will using [pyTorch](https://en.wikipedia.org/wiki/PyTorch) in our project)
"""

# Commented out IPython magic to ensure Python compatibility.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
# %matplotlib inline

class LogisticRegression(torch.nn.Module):
    def __init__(self, num_features):
        # init super class of LogisticRegression
        super(LogisticRegression, self).__init__()

        # create "linear neural network"
        input_dim  = num_features
        output_dim = 1
        self.linear = torch.nn.Linear(input_dim, output_dim)

        # initialize Weights and Bias
        self.linear.weight.detach().zero_()
        self.linear.bias.detach().zero_()

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

"""In our “forward” pass of the PyTorch neural network (really just a perceptron), Logistic regression can also be visualized as a network of features feeding into a single logistic function, the visual representation and corresponding equations are shown below:

<p align='center'>
    <img src='https://github.com/viensea1106/Federated-Learning-meets-Homomorphic-Encryption/blob/main/images/logistic_model_overview.png?raw=1'>
</p>

### Sigmoid function

The sigmoid function is extremely useful for two main reasons:

* It transforms our linear regression output to a probability from 0 to 1. We can then take any probability greater than 0.5 as being 1 and below as being 0.

* Unlike a stepwise function (which would transform the data into the binary case as well), the sigmoid is differentiable, which is necessary for optimizing the parameters using gradient descent (we will show later).

<p align='center'>
    <img src='https://github.com/viensea1106/Federated-Learning-meets-Homomorphic-Encryption/blob/main/images/sigmoid.png?raw=1'>
</p>

### Training process

Firstly, we should assign some hyper-parameters
"""

input_dim = 31 # numbers of features
epochs = 10000
learning_rate = 0.01

"""Parameter Definitions:

* **Epoch**: Indicates the number of passes through the entire training dataset the network has completed.
* **learning_rate**: A tuning parameter in an optimization algorithm that determines the step size at each iteration while moving toward a minimum of a loss function.
"""

# from tqdm.notebook import tqdm
from tqdm import tqdm

def decide(y):
    return 1. if y >= 0.5 else 0.

decide_vectorized = np.vectorize(decide)
to_percent = lambda x: '{:.2f}%'.format(x)

def compute_accuracy(model, input, output):
    prediction = model(input).data.numpy()[:, 0]
    n_samples = prediction.shape[0] + 0.
    prediction = decide_vectorized(prediction)
    equal = prediction == output.data.numpy()
    return 100. * equal.sum() / n_samples

def Training(X_train, Y_train, X_test, Y_test, debug=True):
    model = LogisticRegression(input_dim)
    n_samples, _ = X_train.shape

    # record losses and accuracies during training
    losses = []
    accuracies = []

    # define criterion function and set up optimizer
    criterion = torch.nn.BCELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # main process
    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()
        #### Compute outputs ####
        prediction = model(X_train)

        #### Compute gradients ####
        loss = criterion(prediction.squeeze(), Y_train)

        loss.backward()

        #### Update weights ####
        optimizer.step()

        #### Logging ####
        if debug and (epoch + 1)%50 == 0:
            # compute accuracy and loss
            train_acc = compute_accuracy(model, X_train, Y_train)
            train_loss = loss.item()
            losses.append(train_loss)
            accuracies.append(train_acc)

            print('[LOG] Epoch: %05d' % (epoch + 1), end="")
            print('    | Train ACC: %s' % to_percent(train_acc), end="")
            print('    | Loss: %.3f' % train_loss)

    recorded = [accuracies, losses]
    return model, recorded

"""In this above code, we introduce two important functions: the Loss Function and the Optimizer

**Binary Cross Entropy Loss Function**

```python
criterion = torch.nn.BCELoss(reduction='mean')
```
<p align='center'>
    <img src='https://github.com/viensea1106/Federated-Learning-meets-Homomorphic-Encryption/blob/main/images/loss_function.png?raw=1'>
</p>

* $m$: Number of training examples
* $y$: The true $y$ value
* $\hat{y}$: Predicted $y$ value

**Stochastic Gradient Descent Optimizer**

```python
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  
```

We update the parameters to minimize the loss function with the following equations:

* Update model **Weights**:

<p align='center'>
    <img src='https://github.com/viensea1106/Federated-Learning-meets-Homomorphic-Encryption/blob/main/images/update_w.webp?raw=1'>
</p>

* Update model **Bias**:

<p align='center'>
    <img src='https://github.com/viensea1106/Federated-Learning-meets-Homomorphic-Encryption/blob/main/images/update_b.webp?raw=1'>
</p>

where $\alpha$ is the **learning_rate**

<p align='center'>
    <img src='https://github.com/viensea1106/Federated-Learning-meets-Homomorphic-Encryption/blob/main/images/gradient_descent.png?raw=1' title='Gradient Descent'>
</p>

Demo our training model
"""

# training process
final_model, recorded = Training(X_train, Y_train, X_test, Y_test)

"""Model parameters after training process"""

print('Model parameters:')
print('  | Weights: %s' % final_model.linear.weight)
print('  | Bias: %s'    % final_model.linear.bias)

"""### Virtualize record of training process"""

def plot_graphs(diagnosis_title, record):
    accuracies, losses = record
    plt.plot(losses)
    plt.title(f"{diagnosis_title} - Training Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Training Loss")
    plt.show()
    plt.plot(accuracies)
    plt.title(f"{diagnosis_title} - Training Accuracy")
    plt.xlabel("Iterations")
    plt.ylabel("Training Accuracy (Percent %)")
    plt.show()

diagnosis_title = 'Breast cancer'
plot_graphs(diagnosis_title, recorded)

"""### Evaluating the Model"""

test_acc = compute_accuracy(final_model, X_test, Y_test)
print('[+] Testing Accuracy = {}'.format(to_percent(test_acc)))

"""We actually train the machine learning model to diagnose the Breast cancer. As you can see in the graphs, the training loss drops quickly to almost zero and the training accuracy reaches the 98%. The testing accuracy is also 94.74. Notice that this machine learning system diagnoses this disease in a perfect way; whereas human doctors can commit mistakes.

## Federated Learning

So far, we have used machine learning in an insecure way. Now, we introduce our proposed encrypted learning model. This is a combination of federated learning and homomorphic encryption!

<p align='center'>
    <img src='https://github.com/viensea1106/Federated-Learning-meets-Homomorphic-Encryption/blob/main/images/encrypted_learning.png?raw=1'>
</p>

**Set up**

- 1. Define HE scheme, Hospitals shared a HE private key, and share with the Aggregator a public key.
- 2. Define the model that use to train.

**Local training**
- 3. Hospitals train locally with the model in plaintext, extract model updates (the final weights) and encrypt them using the private HE key.
- 4. Each hospital sends its encrypted model's weights to the Aggregator (untrusted).

**Global model weight aggregator**
- 5. Aggregator uses its public key to perform homomorphic operation on the encrypted weights to obtain new encrypted model, and send its to hospitals.

**Repeated**
- 6. Each hospital receives the decrypted model, then uses their private HE key, decrypts its to get the new model.
- 7. This process ís repeated until hospitals find the new desired model or after a termination criteria is met.

### Datasets
"""

import pandas as pd

"""### Dataset 1

[source](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset)

**Description:**
Breast cancer is the most common cancer amongst women in the world. It accounts for 25% of all cancer cases, and affected over 2.1 Million people in 2015 alone. It starts when cells in the breast begin to grow out of control. These cells usually form tumors that can be seen via X-ray or felt as lumps in the breast area.

The key challenges against it’s detection is how to classify tumors into malignant (cancerous) or benign(non cancerous). We ask you to complete the analysis of classifying these tumors using machine learning (with SVMs) and the Breast Cancer Wisconsin (Diagnostic) Dataset.

**Acknowledgements:**
This dataset has been referred from Kaggle.

**Objective:**
* Understand the Dataset & cleanup (if required).
* Build classification models to predict whether the cancer type is Malignant or Benign.
* Also fine-tune the hyperparameters & compare the evaluation metrics of various classification algorithms.
"""

df1 = pd.read_csv('data/dataset1.csv')
print("Shape:", df1.shape)
df1.head(5)

"""### Dataset 2

[source](https://www.kaggle.com/code/a3amat02/breast-cancer-classification/input)

**About this file**

* y. The outcomes. A factor with two levels denoting whether a mass is malignant ("M") or benign ("B").
* x. The predictors. A matrix with the mean, standard error and worst value of each of 10 nuclear measurements on the slide, for 30 total features per biopsy:
* radius. Nucleus radius (mean of distances from center to points on perimeter).
* texture. Nucleus texture (standard deviation of grayscale values).
* perimeter. Nucleus perimeter.
* area. Nucleus area.
* smoothness. Nucleus smoothness (local variation in radius lengths).
* compactness. Nucleus compactness (perimeter^2/area - 1).
* concavity, Nucleus concavity (severity of concave portions of the contour).
* concave_pts. Number of concave portions of the nucleus contour.
* symmetry. Nucleus symmetry.
* fractal_dim. Nucleus fractal dimension ("coastline approximation" -1).
"""

df2 = pd.read_csv('data/dataset2.csv')
print("Shape:", df2.shape)
df2.head(5)

"""### Dataset 3

[source](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))

**Data Set Information:**

Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image. A few of the images can be found at [Web Link]

Separating plane described above was obtained using Multisurface Method-Tree (MSM-T) [K. P. Bennett, "Decision Tree Construction Via Linear Programming." Proceedings of the 4th Midwest Artificial Intelligence and Cognitive Science Society, pp. 97-101, 1992], a classification method which uses linear programming to construct a decision tree. Relevant features were selected using an exhaustive search in the space of 1-4 features and 1-3 separating planes.

The actual linear program used to obtain the separating plane in the 3-dimensional space is that described in: [K. P. Bennett and O. L. Mangasarian: "Robust Linear Programming Discrimination of Two Linearly Inseparable Sets", Optimization Methods and Software 1, 1992, 23-34].

This database is also available through the UW CS ftp server:
ftp ftp.cs.wisc.edu
cd math-prog/cpo-dataset/machine-learn/WDBC/


**Attribute Information:**

1) ID number

2) Diagnosis (M = malignant, B = benign)

3-32)

Ten real-valued features are computed for each cell nucleus:

a) radius (mean of distances from center to points on the perimeter)

b) texture (standard deviation of gray-scale values)

c) perimeter

d) area

e) smoothness (local variation in radius lengths)

f) compactness (perimeter^2 / area - 1.0)

g) concavity (severity of concave portions of the contour)

h) concave points (number of concave portions of the contour)

i) symmetry

j) fractal dimension ("coastline approximation" - 1)
"""

df3 = pd.read_csv('data/dataset3.csv')
print("Shape:", df3.shape)
df3.head(5)

"""In this demo, there are 3 hospitals (with corresponding datasets above) but there could be more hospitals. The 3 hospitals cannot share the cases of their patients because they are competitors and it is necessary to protect the privacy of patients. Hence, the ML model will be learned in a federated way.

## Clients

Import openFHE library. Then run `generate_key()`, this is just a python wrapper to run `openfhe-lib/build/key_gen`. After running this files, it will create `openfhe-lib/data` folder that holds:
- `crypto_context.txt` : contain CKKS CryptoContext object
- `public_key.txt`     : contain CKKS public key
- `private_key.txt`    : contain CKKS private key
- `mult_key.txt`       : contain CKKS multiplication key
"""

from openfhe_lib.ckks.openFHE import *

# === Generate Key-pairs of CKKS Context ===
generate_keys()

"""First, we start by creating the `Client` class that simulate the computers of each hospital. This just rewrite the Logistic Regression process that we have implemented previously."""

class Client:
    def __init__(self, name, data_url, enc_file, n_features, iters):
        self.id = name
        self.enc_file = enc_file  # place wher clients save encrypted weights

        # split data into train and test
        self.X_train, self.Y_train, self.X_test, self.Y_test = self.preprocessing(data_url)

        # define local training model
        self.local_model = LogisticRegression(n_features)

        # some helpfull stuffs
        self.decide_vectorized = np.vectorize(self.decide)
        self.to_percent = lambda x: '{:.2f}%'.format(x)
        self.num_epochs = iters
        self.accuracies = []
        self.losses = []

    def preprocessing(self, data_url):
        df = pd.read_csv(data_url)
        # Replace "M" with 1 and "B" with 0 at "diagnostic" column
        df["diagnostic"] = (df["diagnostic"] == "M").astype(int)

        # split dataframe to train and test df
        df_train, df_test = np.split(df.sample(frac=1), [int(0.8 * len(df))])

        # scaling and convert to tensor context
        train, X_train, Y_train = scale_dataset(df_train, True)
        test , X_test , Y_test  = scale_dataset(df_test , False)
        return X_train, Y_train, X_test, Y_test

    def decide(self, y):
        return 1. if y >= 0.5 else 0.

    def compute_accuracy(self, input, output):
        prediction = self.local_model(input).data.numpy()[:, 0]
        n_samples = prediction.shape[0] + 0.
        prediction = self.decide_vectorized(prediction)
        equal = prediction == output.data.numpy()
        return 100. * equal.sum() / n_samples

    def local_training(self, debug=True):
        n_samples, _ = self.X_train.shape

        # define criterion function and set up optimizer
        criterion = torch.nn.BCELoss(reduction='mean')
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=0.01)

        # main process
        for epoch in range(self.num_epochs):
            optimizer.zero_grad()
            #### Compute outputs ####
            prediction = self.local_model(self.X_train)

            #### Compute gradients ####
            loss = criterion(prediction.squeeze(), self.Y_train)
            loss.backward()

            #### Update weights ####
            optimizer.step()

            # compute accuracy and loss
            train_acc = self.compute_accuracy(self.X_train, self.Y_train)
            train_loss = loss.item()

            self.losses.append(train_loss)
            self.accuracies.append(train_acc)

            #### Logging ####
            if debug and (epoch + 1)%50 == 0:
                print('[LOG] Epoch: %05d' % (epoch + 1), end="")
                print('    | Train ACC: %s' % self.to_percent(train_acc), end="")
                print('    | Loss: %.3f' % train_loss)

    def encrypted_model_params(self):
        print('function: encrypted_model_params')
        model_weights = self.local_model.linear.weight.data.squeeze().tolist()
        model_bias    = self.local_model.linear.bias.data.squeeze().tolist()

        model_params  = model_weights + [model_bias]
        encrypt_weights(model_params, self.enc_file)

    def decrypted_model_params(self):
        print('function: decrypted_model_params')
        params = decrypt_weights("/content/Federated-Learning-meets-Homomorphic-Encryption/data/ckks/enc_aggregator_weight_server.txt")
        # convert float to tensor context
        W = Variable(torch.tensor([params[:-1]], dtype = torch.float32))
        B = Variable(torch.tensor( params[-1], dtype = torch.float32))

        self.local_model.linear.weight = nn.Parameter(W)
        self.local_model.linear.bias   = nn.Parameter(B)

    def plot_graphs(self, diagnosis_title = 'Breast cancer'):
        plt.plot(self.losses)
        plt.title(f"{diagnosis_title} - Training Loss")
        plt.xlabel("Iterations")
        plt.ylabel("Training Loss")
        plt.show()
        plt.plot(self.accuracies)
        plt.title(f"{diagnosis_title} - Training Accuracy")
        plt.xlabel("Iterations")
        plt.ylabel("Training Accuracy (Percent %)")
        plt.show()

    def print_result_after_training(self):
        print('Model parameters:')
        print('  | Weights: %s' % self.local_model.linear.weight)
        print('  | Bias: %s' % self.local_model.linear.bias)
        self.plot_graphs()

    def evaluating_model(self):
        test_acc = self.compute_accuracy(self.X_test, self.Y_test)
        print('[+] Testing Accuracy = {}'.format(self.to_percent(test_acc)))

"""## Server

We define some functions to train the machine learning model in a federated way while keeping track of the training loss and the training accuracy, for each hospital separately.
"""

clients = [
    Client('Hostpital1', 'data/dataset1.csv', "/content/Federated-Learning-meets-Homomorphic-Encryption/data/ckks/enc_weight_client1.txt", n_features=31, iters=10),
    Client('Hostpital2', 'data/dataset2.csv', "/content/Federated-Learning-meets-Homomorphic-Encryption/data/ckks/enc_weight_client2.txt", n_features=31, iters=10),
    Client('Hostpital3', 'data/dataset3.csv', "/content/Federated-Learning-meets-Homomorphic-Encryption/data/ckks/enc_weight_client3.txt", n_features=31, iters=10),
    Client('Hostpital4', 'data/dataset4.csv', "/content/Federated-Learning-meets-Homomorphic-Encryption/data/ckks/enc_weight_client4.txt", n_features=31, iters=10)
]

"""The whole process is done in a server aggregator, in 1000 iterations (we can vary the number of iterations.) At each iteration."""

iterations = 1000 #2000
worker_iterations = 5
to_percent = lambda x: '{:.2f}%'.format(x)
n_hospitals = len(clients)
n_features = 31

def compute_federated_accuracy(model, input, output):
    prediction = model(input)
    n_samples = prediction.shape[0]
    s = 0.
    for i in range(n_samples):
        p = 1. if prediction[i] >= 0.5 else 0.
        e = 1. if p == output[i] else 0.
        s += e
    return 100. * s / n_samples

def federated_learning(clients):
    # init global training model
    global_model = LogisticRegression(n_features)

    # record losses and accuracies report from clients
    losses = [[] for i in range(n_hospitals)]
    accuracies = [[] for i in range(n_hospitals)]

    pbar = tqdm(range(iterations), desc='Federated Learning Process')
    for iteration in pbar:
        if iteration:
            # copy global model to clients
            # clients will receive the weight-aggregated from server, extract
            # the ciphertext then decrypt it to get to global_model's weights
            for i in range(n_hospitals):
                clients[i].decrypted_model_params()

        # perform local training for each clients then report acc and loss to server
        for i in range(n_hospitals):
            clients[i].local_training(debug=False)

            # report to server
            losses[i].append(clients[i].losses[-1])
            accuracies[i].append(clients[i].accuracies[-1])

        # clients encrypt the final weights of local model after training
        for i in range(n_hospitals):
            clients[i].encrypted_model_params()

        # server collect clients's encrypted weights then perform weight-aggregation
        # by using homomorphic operation
        with torch.no_grad():
            # avg_weight = sum([clients[i].local_model.linear.weight.data for i in range(n_hospitals)]) / n_hospitals
            # global_model.linear.weight = nn.Parameter(avg_weight)
            # avg_bias = sum([clients[i].local_model.linear.bias.data for i in range(n_hospitals)]) / n_hospitals
            # global_model.linear.bias = nn.Parameter(avg_bias)
            aggregator()

        # logging
        if (iteration + 1) % 100 == 0:
            losses_str = ['{:.4f}'.format(losses[i][-1]) for i in range(n_hospitals)]
            accuracies_str = [to_percent(accuracies[i][-1]) for i in range(n_hospitals)]
            print('[LOG] Epoch = {0:04d}\n> Losses = {1}\n> Accuracies = {2}'.format(iteration + 1, losses_str, accuracies_str))

    return losses, accuracies

print(os.getcwd())

losses, accuracies = federated_learning(clients)

"""### Virtualize record of training process"""

def plot_federated_graphs(diagnosis_title, losses, accuracies):
    for i in range(n_hospitals):
        plt.plot(losses[i], label=f'Hospital {i+1}')
    legend = plt.legend(loc='upper right', shadow=True)
    plt.title(f"{diagnosis_title} - Training Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Training Loss")
    plt.show()
    for i in range(n_hospitals):
        plt.plot(accuracies[i], label=f'Hospital {i+1}')
    legend = plt.legend(loc='lower right', shadow=True)
    plt.title(f"{diagnosis_title} - Training Accuracy")
    plt.xlabel("Iterations")
    plt.ylabel("Training Accuracy (Percent %)")
    plt.show()

plot_federated_graphs('Breast cancer', losses, accuracies)

"""### Model parameters after training proccess"""

clients[0].decrypted_model_params()
global_model = clients[0].local_model

print('\nModel parameters:')
print('  | Weights: %s' % global_model.linear.weight) ### Virtualize record of training processlobal_model.linear.weight)
print('  | Bias: %s' % global_model.linear.bias)

"""### Evaluating the Model"""

# prepare data for testing model
df_test = pd.read_csv('data/test.csv')
df_test["diagnostic"] = (df_test["diagnostic"] == "M").astype(int)
test , X_test , Y_test  = scale_dataset(df_test , False)

test_acc = compute_federated_accuracy(global_model, X_test, Y_test)
print('\nTesting Accuracy = {}'.format(to_percent(test_acc)))

"""## Thanks for reading!
I hope you have enjoyed the explanations of this machine learning system with federated learning.

### References

- [1] Wibawa, F., Catak, F. O., Kuzlu, M., Sarp, S., & Cali, U. (2022, June). Homomorphic encryption and federated learning based privacy-preserving cnn training: Covid-19 detection use-case. In Proceedings of the 2022 European Interdisciplinary Cybersecurity Conference (pp. 85-90).​

- [2] Al Badawi, A., Bates, J., Bergamaschi, F., Cousins, D. B., Erabelli, S., Genise, N., ... & Zucca, V. (2022, November). OpenFHE: Open-source fully homomorphic encryption library. In Proceedings of the 10th Workshop on Encrypted Computing & Applied Homomorphic Cryptography (pp. 53-63).
"""
