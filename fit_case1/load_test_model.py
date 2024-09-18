import torch
from kan import *
from kan.utils import create_dataset
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
"""
The code is used to demonstrate the prediction of the model
"""
device="cuda" if torch.cuda.is_available() else "cpu"
# # create dataset f(x,y) = exp(sin(pi*x)+y^2)
f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)

#%% 1 plot the groundtruth
x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x, y)
X_reshaped = X.reshape(-1, 1)
Y_reshaped = Y.reshape(-1, 1)
stacked_xy = torch.from_numpy(np.hstack((X_reshaped, Y_reshaped))).float()
Z = f(stacked_xy)
Z_reshaped = Z.reshape(100, 100) 

# Create the 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(X, Y, Z_reshaped, cmap='viridis')

# Customize the plot
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_title('Test Groundtruth')

# Add a color bar
fig.colorbar(surf)
plt.savefig("groundtruth_3D")
plt.close()

#%% 2 plot the prediction
model = KAN.loadckpt('./model/' + '0.6')
pred=model(stacked_xy)
pred_reshaped = pred.reshape(100, 100)
pred_reshaped = pred_reshaped.detach().cpu().numpy()

# Create the 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(X, Y, pred_reshaped, cmap='viridis')

# Customize the plot
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_title('Prediction')

# Add a color bar
fig.colorbar(surf)
plt.savefig("prediction_3D")