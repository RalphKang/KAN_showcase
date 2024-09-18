#%%
from kan import *
torch.set_default_dtype(torch.float64)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

"""
In this code, we try to fit Einstein's mass-velocity relation using KAN. The purpose is to show you why simply training ML model via *data* is not enough, because:
1, thousands of solutions exist to fit the data, but they may not be physically meaningful
2, Accordingly, researchers and engineers need the tool to whiten the ML model
3, And, incoporating their knowledge in the ML model can improve the accuracy and generalization of the model
"""

#%% 1 create a KAN: 2D inputs, 1D output, and 5 hidden neurons. cubic spline (k=3), 5 grid intervals (grid=5).
model = KAN(width=[3,5,1], grid=3, k=3, seed=42, device=device)
#%% 2 create dataset
from kan.utils import create_dataset
f = lambda x: x[:,[0]]/torch.sqrt(1-x[:,[1]]**2/x[:,[2]]**2)
dataset = create_dataset(f, n_var=3, ranges=[[0,1],[0,0.9],[1.1,2]])
# dataset['train_input'].shape, dataset['train_label'].shape
#%% 3 plot KAN at initialization
model(dataset['train_input']);
model.plot()
plt.savefig("initialized network.png")
plt.close()
# plt.show()
#%% 4 train the model use LBFGS optimizer
model.fit(dataset, opt="LBFGS", steps=50, lamb=0.001);
model.plot()
plt.savefig("trained network.png")
plt.close()
#plt.show()
#%% 5 Analyzing the trained model
print(model.feature_score) # check the importance of the features
model.tree(sym_th=1e-2, sep_th=5e-1) # plot the modular structure of the model
plt.savefig("model tree.png")
#%% 6 Pruning the model and refining the model, for better accuracy and symbolic regression
model = model.prune()
model=model.refine(10)
# model.plot()
# model.fit(dataset, opt="LBFGS", steps=50)
# model = model.refine(10)
model.fit(dataset, opt="LBFGS", steps=50)

#%% 7 Symbolic regression on previous model, and check the formula
model.auto_symbolic()


model.fit(dataset, opt="LBFGS", steps=50)
pred_output=model(dataset['test_input'])
from kan.utils import ex_round

symbolic_result=model.symbolic_formula()
print(ex_round(model.symbolic_formula()[0][0],6))

