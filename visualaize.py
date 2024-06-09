import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# push check

# Load the CSV file
file_path = "E:\Ahmed work space\Cycl_T45_SOC10-90_Dch0.5C_Ch0.5C_Vito_Cell63_AllData.csv"

# Define the number of rows to sample
num_rows_to_sample = 1000  # Adjust this number as needed

# Read a random sample of rows from the CSV file
data = pd.read_csv(file_path, nrows=num_rows_to_sample)

# Set the figure size
plt.figure(figsize=(12, 8))

# Plot voltage over time
plt.subplot(2, 2, 1)
sns.lineplot(data=data, x='Unnamed: 0', y='Voltage_V')
plt.title('Voltage Over Time')

# Plot current over time
plt.subplot(2, 2, 2)
sns.lineplot(data=data, x='Unnamed: 0', y='Current_A')
plt.title('Current Over Time')


# Plot discharge capacity over time
plt.subplot(2, 2, 3)
sns.lineplot(data=data, x='Unnamed: 0', y='Discharge_Capacity_Ah')
plt.title('Discharge Capacity Over Time')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()
