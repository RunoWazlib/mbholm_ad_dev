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
        test_samples (_type_): How many test data sets to make
        features (_type_): How many features each data should include
        contamination (_type_): Percentage of data predicted to be an outlier

    Returns:
        X_train = np.array: an array of training values, based on Gaussian distribution
        X_test = np.array:  an array of test values, based on Gaussian distribution
    """
    X_train, X_test, _, _ = pyod.utils.data.generate_data(n_train=train_samples, n_test=test_samples, n_features=features, contamination=contamination, random_state=42)
    print("[*] Data Generated")
    return X_train, X_test

def apply_Iforest():
    #Create instance
    model = IForest() #could put in an expected amount of anomalous data

    #Train model
    model.fit(data_generation(100, 10, 10, 0.1)[0])

    #Evaluate test data for anomalies
    scores = model.decision_function(data_generation(100, 10, 10, 0.1)[0])
    return scores
    
def graph_data(X_train, X_test):
    #Graph X_Train data ***In Progress***
    fig, axs = plt.subplots(figsize=(5, 2.7), layout='constrained')
    y = np.array(range(0, 100))

    axs.plot(X_train[0], X_train[1])
    plt.show()

    #Graph X_test data ***In Progress***

data_scores = apply_Iforest()
#Smaller values are less anomalous, Larger
print("[-] Anomaly scores: ")
i = 0
for score in data_scores:
    print(f"Score {i+1}: {score}")
    i += 1
