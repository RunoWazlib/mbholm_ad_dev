import matplotlib.pyplot as plt
import numpy as np
from pyod.models.iforest import IForest
import pyod.utils.data

fake_data = pyod.utils.data.generate_data()
#IForest()
#print(fake_data)
for i in range(4):
    if i == 0:
        print("[*] X_train:")
    if i == 1:
        print("[*] X_test:")
    if i == 2:
        print("[*] Y_train:")
    if i == 3:
        print("[*] Y_test:")
    print(f"%s \n" % fake_data[i])

print("[-] Data Generated")

