import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pyod.models.iforest import IForest
import pyod.utils.data

#Output of .generate_data()
print("[*] Output of .generate_data():\n")
fake_data = pyod.utils.data.generate_data()
print(f"X_train: \n{fake_data[0]}")
print(f"X_test: \n{fake_data[1]}")
print(f"y_train: \n{fake_data[2]}")
print(f"y_test: \n{fake_data[3]}")

#Generate training data
samples = 100
features = 10
contamination = 0.1
X_train, _, _, _ = pyod.utils.data.generate_data(n_train=samples, n_features=features, contamination=contamination)
print("[*] Training Data Generated")

#Graph X_Train data ***In Progress***
fig, axs = plt.subplots(figsize=(5, 2.7), layout='constrained')
y = np.array(range(0, 100))

plt.show()
axs.plot(X_train, )


#Create instance
model = IForest() #could put in an expected amount of anomalous data

#Train model
model.fit(X_train)

#Generate test data
samples = 10
features = 10
X_test, _, _, _ = pyod.utils.data.generate_data(n_test=samples, n_features=features, contamination=contamination)
print("[*] Test Data Generated")

#Evaluate test data for anomalies
scores = model.decision_function(X_test)

#Smaller values are less anomalous, Larger
print("[-] Anomaly scores: ")
i = 0
for score in scores:
    print(f"Score {i}: {score}")
    i += 1