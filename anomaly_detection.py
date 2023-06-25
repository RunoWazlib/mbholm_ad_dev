import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pyod.models.iforest import IForest
import pyod.utils.data

def pyod_data_test():
    #Output of .generate_data()
    print("[*] Output of .generate_data():\n")
    fake_data = pyod.utils.data.generate_data(random_state=42)
    print(f"X_train: \n{fake_data[0]}")
    print(f"X_test: \n{fake_data[1]}")
    print(f"y_train: \n{fake_data[2]}")
    print(f"y_test: \n{fake_data[3]}")

def data_generation(train_samples, test_samples, features, contamination):
    """Generates a sample set of data for use in the IForest algorithm

    Args:
        train_samples (int): How many training data sets to make
        test_samples (int): How many test data sets to make
        features (int): How many features each data should include
        contamination (int): Percentage of data predicted to be an outlier

    Returns:
        X_train = np.array: an array of training values, based on Gaussian distribution
        X_test = np.array:  an array of test values, based on Gaussian distribution
    """
    X_train, X_test, _, _ = pyod.utils.data.generate_data(n_train=train_samples, n_test=test_samples, n_features=features, contamination=contamination, random_state=42)
    print("[*] Data Generated")
    return X_train, X_test

def apply_Iforest(X_train, X_test):
    #Create instance
    model = IForest()

    #Train model
    model.fit(X_train)

    #Evaluate test data for anomalies
    scores = model.decision_function(X_test)
    return scores
    
def graph_data(X_train, X_test):
    #Graph X_Train data ***In Progress***
    fig, axs = plt.subplots(figsize=(5, 2.7), layout='constrained')
    y = np.array(range(0, 10))

    axs.plot(X_train[0], y)
    plt.show()

    #Graph X_test data ***In Progress***

train_data, test_data = data_generation(100, 10, 10, 0.1)
data_scores = apply_Iforest(train_data, test_data)
print("[-] Anomaly scores: ")
i = 0
for score in data_scores:
    print(f"Score {i+1}: {score}")
    i += 1

#Smaller values are less anomalous, Larger
graph_data(train_data, test_data)