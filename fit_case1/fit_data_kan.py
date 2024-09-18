#%%
from kan import *
from kan.utils import create_dataset
import matplotlib.pyplot as plt
"""
This code is used to demonstrate the general process of using KAN for data fitting.
"""

torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#%% 0. create dataset f(x,y) = exp(sin(pi*x)+y^2)
f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
dataset = create_dataset(f, n_var=2, device=device)
#%% 1. create a KAN: 2D inputs, 1D output, and 5 hidden neurons. cubic spline (k=3), 5 grid intervals (grid=5).
model = KAN(width=[2,5,1], grid=3, k=3, seed=42, device=device)
model(dataset['train_input'])
model.plot()
plt.savefig("initialized network")
plt.close()
#%% 2. fit the model

model.fit(dataset, opt="LBFGS", steps=50, lamb=0.001);
model.plot()
plt.savefig("trained network")
plt.close()
# plt.show()
#%% 3. Model analysis
print(model.feature_score) # print the feature importance
model.tree(sym_th=1e-2, sep_th=5e-1)
plt.savefig("model structure")
plt.close()
model = model.prune()

#%% 4. refine and train the model for further symbolic regression
model=model.refine(10)
model.fit(dataset, opt="LBFGS", steps=50)

#%% 5. symbolic regression
mode = "auto" # "manual"

if mode == "manual":
    # manual mode
    model.fix_symbolic(0,0,0,'sin');
    model.fix_symbolic(0,1,0,'x^2');
    model.fix_symbolic(1,0,0,'exp');
elif mode == "auto":
    # automatic mode
    lib = ['x','x^2','x^3','x^4','exp','log','sqrt','tanh','sin','abs']
    model.auto_symbolic(lib=lib)


model.fit(dataset, opt="LBFGS", steps=50)
pred_output=model(dataset['test_input'])
from kan.utils import ex_round
print(ex_round(model.symbolic_formula()[0][0],4))
# %%
