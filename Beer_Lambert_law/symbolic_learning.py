from beer_lambert_law import calculate_output_intensity
from dataset_gen import data_generation_train, data_generation_test
train_data, train_label = data_generation_train()
from pysr import PySRRegressor
import numpy as np

random_seed =2
np.random.seed(random_seed)

model = PySRRegressor(
    niterations=40,  # < Increase me for better results
    binary_operators=["+", "*"],
    unary_operators=[
        "cos",
        "exp",
        "sin",
        "inv(x) = 1/x",
        # ^ Custom operator (julia syntax)
    ],
    extra_sympy_mappings={"inv": lambda x: 1 / x},
    # ^ Define operator for SymPy as well
    elementwise_loss="loss(prediction, target) = (prediction - target)^2",
    # ^ Custom loss function (julia syntax)
)

model.fit(train_data, train_label)
print(model)