


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd
import numpy as np


# Convert 'iris.data' numpy array to 'iris.dataframe' pandas dataframe
# complete the iris dataset by adding species
iris = datasets.load_iris()
iris = pd.DataFrame(
    data= np.c_[iris['data'], iris['target']],
    columns= iris['feature_names'] + ['target']
    )

species = []

for i in range(len(iris['target'])):
    if iris['target'][i] == 0:
        species.append("setosa")
    elif iris['target'][i] == 1:
        species.append('versicolor')
    else:
        species.append('virginica')


iris['species'] = species
iris

# 1. splitting the dataset into training and test sets

X = iris.iloc[:, 0:4]
y = iris.iloc[:, 4]

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size = 0.7)

# 2. save entire dataset, training and testing datasets
# save a copy of the dataset in .csv
iris.to_csv(r'C:/Users/julia/OneDrive/Desktop/github/24. Machine_learning_toolbox_Python/iris.csv', index=False)


iris.to_csv(r'C:/Users/julia/OneDrive/Desktop/github/24. Machine_learning_toolbox_Python/iris_training.csv',
          index = False)

iris.to_csv(r'C:/Users/julia/OneDrive/Desktop/github/24. Machine_learning_toolbox_Python/iris_testing.csv',
          index = False)