import matplotlib.pyplot as plt
import pandas as pd

def plot_data():
    """
    Reads data from a .dat file and plots a graph using matplotlib.

    Parameters:
    None

    Returns:
    None
    """
    # Read the data from the downloads folder
    data = pd.read_dat('C:\Users\pmzas2\Downloads\AmplitudeVectorNorm.dat')

    # Extract the x and y values from the data
    x = data['x']
    y = data['y']

    # Plot the graph
    plt.plot(x, y)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Graph from Data')
    plt.show()
