from beer_lambert_law import calculate_output_intensity
import numpy as np
import torch
def data_generation_train():
    initial_intensity=np.random.rand(100)*50.+50.
    absorp_coef=0.1
    path_length=np.random.rand(100)*10.+1.0

    output_intensities = [
        calculate_output_intensity(initial_intensity[i], absorp_coef, path_length[i])
        for i in range(len(initial_intensity))
    ]

    # transform data into torch
    train_data = torch.tensor(np.array([initial_intensity, path_length]).T, dtype=torch.float) # need to be float
    train_label=torch.tensor(output_intensities, dtype=torch.float)
    return train_data, train_label

def data_generation_test():
    initial_intensity=np.random.rand(100)*100.+50.
    absorp_coef=0.1
    path_length=np.random.rand(100)*10.+1.0

    output_intensities = [
        calculate_output_intensity(initial_intensity[i], absorp_coef, path_length[i])
        for i in range(len(initial_intensity))
    ]

    # transform data into torch
    train_data = torch.tensor(np.array([initial_intensity, path_length]).T, dtype=torch.float) # need to be float
    train_label=torch.tensor(output_intensities, dtype=torch.float)
    return train_data, train_label
