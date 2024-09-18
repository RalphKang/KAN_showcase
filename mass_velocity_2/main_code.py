#%%
from kan.MultKAN import MultKAN
from sympy import *
from kan.utils import create_dataset, augment_input
import torch
import matplotlib.pyplot as plt
"""
This code is to show you a demo how you can include physics-driven features in KAN, so that you can simply the learning problem and guide the learning process by domain knowledge.

"""
seed = 1
torch.manual_seed(seed)
torch.set_default_dtype(torch.float64)

#%% create dataset for learning
f = lambda x: x[:,[0]]/torch.sqrt(1-x[:,[1]]**2/x[:,[2]]**2)
dataset = create_dataset(f, n_var=3, ranges=[[0,1],[0,0.9],[1.1,2]])
input_variables = m0, v, c = symbols('m0 v c')

#%% define domain_realted physics-driven features
beta = v/c 
gamma = 1/sqrt(1-beta**2) # these are two physical variables from domain knowledges

aux_vars = (beta, gamma)



# add auxillary variables to original feature input and dataset
dataset = augment_input(input_variables, aux_vars, dataset)
input_variables = aux_vars + input_variables

#%% initialize KAN
model = MultKAN(width=[5,[0,1]], mult_arity=2, grid=3, k=3, seed=seed)

model(dataset['train_input'])
model.plot()
plt.savefig('initialized network')
#%% fit the model
model.fit(dataset, steps=50, lamb=1e-5, lamb_coef=1.0)
# model.plot(in_vars=input_variables, out_vars=[m0/sqrt(1-v**2/c**2)], scale=0.8, varscale=0.5)
model.plot()
plt.savefig('trained_network')
#%% prune the model to better see the structure of network
model = model.prune(edge_th=5e-2)
# model.plot(in_vars=input_variables, out_vars=[m0/sqrt(1-v**2/c**2)], scale=0.8, varscale=0.5)
# plt.tight_layout()
model.plot()
plt.savefig('pruned_network')
# plt.show()
#%% retrain the pruned model and get the symbolic formula based on that
model.fit(dataset, steps=100, lamb=0e-3)
model.auto_symbolic()
sf = model.symbolic_formula(var=input_variables)[0][0]
from kan.utils import ex_round

print(nsimplify(ex_round(ex_round(ex_round(sf,6),3),3)))