import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURATION ---
# >>> ADJUST THIS TO YOUR DATA FOLDER PATH <<<
data_dir = './data' 
log_file = os.path.join(data_dir, 'driving_log.csv')
# ---------------------

def plot_steering_histogram(log_path):
    """Loads the driving log, assigns correct headers, and plots the histogram."""
    
    # 1. Define correct column names based on PDF 
    # The simulator output often lacks a header row, so we define it manually.
    column_names = ['Center', 'Left', 'Right', 'Steering', 'Throttle', 'Brake', 'Speed']
    
    # 2. Load the CSV file WITHOUT assuming a header row, and assign names
    try:
        # Using header=None tells Pandas the file has no header row
        data = pd.read_csv(log_path, sep=',', header=None, names=column_names)
    except FileNotFoundError:
        print(f"Error: driving_log.csv not found at {log_path}")
        return

    # 3. Extract the steering angles (now correctly named)
    # The 'Steering' column is now correctly identified.
    steering_angles = data['Steering'].values
    
    # 4. Plot the Histogram
    plt.figure(figsize=(12, 6))
    num_bins = 25 
    
    # Plot the histogram
    plt.hist(steering_angles, bins=num_bins, width=0.04) 

    # Add a horizontal line at 1000 for visualization (as seen in Figure 5)
    plt.axhline(y=1000, color='lime', linestyle='-', linewidth=1, label='Target Threshold (Implied)')
    
    plt.title('Steering Angle Distribution')
    plt.xlabel('Steering Angle')
    plt.ylabel('Number of Samples')
    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    
    # Ensure the x-axis matches the standard range shown in the PDF
    plt.xlim(-1.0, 1.0) 
    
    plt.savefig('steering_histogram.png')
    print("Histogram saved as 'steering_histogram.png'")
    
if __name__ == '__main__':
    plot_steering_histogram(log_file)