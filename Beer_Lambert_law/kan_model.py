from kan import *
from dataset_gen import *
# from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt

torch.manual_seed(42)
torch.set_default_dtype(torch.float64)
device='cpu'
model = KAN(width=[2,4,1], grid=5, k=3, seed=42, device=device)


train_data, train_label = data_generation_train()
test_data, test_label = data_generation_test()


f = lambda x: x[:,[0]]*torch.exp(-x[:,[1]]*0.1)
dataset = create_dataset(f, n_var=2, device=device)

model.fit(dataset, opt="LBFGS", steps=100, lamb=0.002, lamb_entropy=2.)
model = model.prune()
model.fit(dataset, opt="LBFGS", steps=50,lamb=0.002, lamb_entropy=2.)

model.plot()
plt.show()
mode="auto"
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

from kan.utils import ex_round

print(ex_round(model.symbolic_formula()[0][0],4))
