#%%

import torch
from kan import *
from kan.utils import create_dataset
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
This code is used to load the trained model and show the test result on the test set. 
You can see that although the test result is good visually, but the model itself in fact is not physically meaningful, and of course, it cannot be the one with good generalization performance.
"""

device="cuda" if torch.cuda.is_available() else "cpu"

# 1 create dataset for test
f = lambda x: x[:,[0]]/torch.sqrt(1-x[:,[1]]**2/x[:,[2]]**2)
dataset = create_dataset(f, n_var=3, ranges=[[0,1],[0,0.9],[1.1,2]])

#%% 1 load model and do the prediction
model = KAN.loadckpt('./model/' + '0.6')
pred=model(dataset["test_input"])
pred=pred.detach().cpu().numpy()

#%% 2 plot to check the performance
ground=dataset["test_label"]
fig=plt.figure()
plt.subplot(3,1,1)
plt.plot(dataset["test_input"][:,0],pred, "*",label="prediction")
plt.plot(dataset["test_input"][:,0],ground, "o",label="groundtruth")
plt.legend()
plt.xlabel("m_0")
plt.ylabel("m")

plt.subplot(3,1,2)
plt.plot(dataset["test_input"][:,1],pred, "*",label="prediction")
plt.plot(dataset["test_input"][:,1],ground, "o",label="groundtruth")
plt.legend()
plt.xlabel("v")
plt.ylabel("m")

plt.subplot(3,1,3)
plt.plot(dataset["test_input"][:,2],pred, "*",label="prediction")
plt.plot(dataset["test_input"][:,2],ground, "o",label="groundtruth")
plt.legend()
plt.xlabel("c")
plt.ylabel("m")

plt.tight_layout()
plt.savefig("prediction")
#plt.show()
# %%
