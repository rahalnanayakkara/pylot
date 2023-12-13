import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
# Assuming your CSV file is named 'data.csv'
file_path = 'car_alley_sortmpc.csv'
columns = ['global_time', 'simulation_time', 'attribute', 'sub_attribute', 'value']
data = pd.read_csv(file_path, header=None, names=columns)

# Filter for longitudinal jerk
jerk_data = data[(data['attribute'] == 'jerk') & (data['sub_attribute'] == 'longitudinal')]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(jerk_data['simulation_time'], jerk_data['value'], label='Longitudinal Jerk')
plt.xlabel('Simulation Time')
plt.ylabel('Longitudinal Jerk')
plt.title('Longitudinal Jerk over Simulation Time')
plt.legend()
plt.show()
