import matplotlib.pyplot as plt
import numpy as np
from pyod.models.iforest import IForest
import pyod.utils.data

#Generate training data
n_train = 100
n_features = 10
contamination = 0.1
X_train, _, _, _ = pyod.utils.data.generate_data(n_train, n_features, contamination=contamination)
print("[-] Training Data Generated")

#Create instance
model = IForest() #could put in an expected amount of anomalous data

#Train model
model.fit(X_train)

#Generate test data
n_test = 10
n_features = 10
X_test, _, _, _ = pyod.utils.data.generate_data(n_test, n_features, contamination=contamination)
print("[-] Test Data Generated")

#Evaluate test data for anomalies
scores = model.decision_function(X_test)

#Smaller values are less anomalous, Larger
print("[-] Anomaly scores: ")
i = 0
for score in scores:
    print(f"Score {i}: {score}")
    i += 1