#%%
from kan.MultKAN import MultKAN
from sympy import *
from kan.utils import create_dataset, augment_input
import torch
import matplotlib.pyplot as plt
from kan.compiler import kanpiler
import numpy as np
from kan.utils import ex_round
"""
This code is to show you how you can encode the mathematical formula in KAN, and further train it to better fit the data
Therefore, you both get the physically meaningful and performance accurate model.
"""

seed = 1
torch.manual_seed(seed)
torch.set_default_dtype(torch.float64)

#%% 0. suppose a higher order term exits in the formula while current theoritical formula does not consider.
f = lambda x: x[:,[0]]/torch.sqrt(1-x[:,[1]]**2/x[:,[2]]**2+x[:,[1]]**4/x[:,[2]]**4) # fake actual truth, which may more fit the experiment data


dataset = create_dataset(f, n_var=3, ranges=[[0,1],[0,0.9],[1.1,2]])
f_1=lambda x: x[:,[0]]/torch.sqrt(1-x[:,[1]]**2/x[:,[2]]**2) # theory formula

train_input=dataset['test_input']
theory_output=f_1(train_input)

x_range=np.arange(0,100)
figure = plt.figure()
plt.subplot(2,1,1)
plt.plot(x_range, dataset['test_label'][:100],"*-")
plt.plot(x_range, theory_output[:100],"o-")
plt.legend(['Actual','Theory'])
plt.xlabel("sample number")
plt.ylabel("m")
plt.subplot(2,1,2)
plt.plot(x_range, dataset['test_label'][:100]-theory_output[:100],"*-")
plt.xlabel("sample number")
plt.ylabel("$dm$")
plt.tight_layout()
plt.savefig('uncertainty_demonstration')
#%% 1 compile the formula into KAN
input_variables = m0, v, c = symbols('m0 v c')
expr=m0/sqrt(1-v**2/c**2)
model = kanpiler(input_variables, expr)
model.get_act(dataset)
# model.plot(in_vars=input_variables, out_vars=[m0/sqrt(1-v**2/c**2)], scale=0.7, varscale=0.4)
model.plot()
# plt.tight_layout()
plt.savefig('original_formula')
plt.close("all")
#%% 2. Reactivate all terms in KAN
model.perturb(mag=0.1, mode='all')
model.plot()
plt.savefig("perturbed_formula")
plt.close()

model.fit(dataset, steps=200, lamb=1e-5, lamb_coef=1.0) # fine tune the model based on the formula
model.plot()
plt.savefig("retained_formula")
plt.close("all")
#%% 3 test the model-------------------------------
pred=model(dataset['test_input'])

pred=pred.detach().numpy()

x_range=np.arange(0,100)
figure = plt.figure()
plt.subplot(2,1,1)
plt.plot(x_range, dataset['test_label'][:100],"*-")
plt.plot(x_range, pred[:100],"o-")
plt.legend(['Actual','Refitted'])
plt.xlabel("sample number")
plt.ylabel("m")
plt.subplot(2,1,2)
plt.plot(x_range, dataset['test_label'][:100]-pred[:100],"*-")
plt.xlabel("sample number")
plt.ylabel("Refitted $dm$")
plt.tight_layout()
plt.savefig('refitted error')
#%% get the symbolic formula
model.auto_symbolic()
sf = model.symbolic_formula(var=input_variables)[0][0]
print(ex_round(ex_round(ex_round(sf,6),3),3))
# %%
