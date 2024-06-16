import pandas as pd

# Load the CSV file
file_path = "E:\Ahmed work space\Cycl_T45_SOC10-90_Dch0.5C_Ch0.5C_Vito_Cell63_AllData.csv"
data = pd.read_csv(file_path)

# Display the first few rows of the DataFrame
print(data.head(10000))
