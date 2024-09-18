import numpy as np
import matplotlib.pyplot as plt

def calculate_output_intensity(initial_intensity, absorption_coefficient, path_length):
    """
    Calculate the output intensity using the Beer-Lambert law.
    
    Parameters:
    initial_intensity (float): The intensity of light before passing through the material
    absorption_coefficient (float): The absorption coefficient of the material (in cm^-1)
    path_length (float): The distance the light travels through the material (in cm)
    
    Returns:
    float: The output intensity after passing through the material
    """
    if initial_intensity <= 0:
        raise ValueError("Initial intensity must be positive")
    if absorption_coefficient < 0:
        raise ValueError("Absorption coefficient must be non-negative")
    if path_length <= 0:
        raise ValueError("Path length must be positive")
    
    return initial_intensity * np.exp(-absorption_coefficient * path_length)

def plot_output_intensity(path_lengths, output_intensities):
    """
    Plot the output intensity as a function of path length.
    
    Parameters:
    path_lengths (array-like): The path lengths
    output_intensities (array-like): The corresponding output intensities
    """
    plt.figure(figsize=(10, 6))
    plt.plot(path_lengths, output_intensities)
    plt.xlabel('Path Length (cm)')
    plt.ylabel('Output Intensity')
    plt.title('Output Intensity vs Path Length')
    plt.grid(True)
    plt.show()

# def main():
#     # Get user input
#     initial_intensity=np.random.rand(100)*50.+50.
#     absorp_coef=0.1
#     path_length=np.random.rand(100)*10.+1.0

    
    
#     # Calculate output intensities
#     output_intensities = [
#         calculate_output_intensity(initial_intensity[i], absorp_coef, path_length[i])
#         for i in range(len(initial_intensity))
#     ]
#     print("----")
    
#     # # Plot the results
#     # plot_output_intensity(path_lengths, output_intensities)
    
#     # # Print some example values
#     # print(f"\nOutput intensity at different path lengths:")
#     # for i in range(0, 101, 25):
#     #     path = path_lengths[i]
#     #     intensity = output_intensities[i]
#     #     print(f"  At {path:.2f} cm: {intensity:.4f}")

# if __name__ == "__main__":
#     main()