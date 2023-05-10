import matplotlib.pyplot as plt
import pandas as pd


class NASAVisualizer():
    def __init__(self) -> None:
        self.data = None

    def format_impedance_df(impedance_df, columns):
        tmp_data = impedance_df
        for col in columns:
            new_col = tmp_data[col]
            new_col = new_col.str.strip()
            new_col = pd.Series([abs(complex(val)) for val in new_col])
            tmp_data[col] = new_col
        return tmp_data

    def plot_data(self.data):
        if "Voltage_charge" in data.columns:
            operation = "charge"
            columns = ["Time", "Voltage_measured", "Current_measured", "Voltage_charge", "Current_charge"]
        elif "Voltage_load" in data.columns:
            operation = "discharge"
            columns = ["Time", "Voltage_measured", "Current_measured", "Voltage_load", "Current_load"]
        elif "Battery_impedance" in data.columns:
            operation = "impedance"
            columns = ["Sense_current", "Battery_current","Battery_impedance", "Rectified_Impedance"]#, "Current_ratio"]
            data = self.format_impedance_df(data, columns)

        fig, axs = plt.subplots(2, 2)
        if operation == "impedance":
            axs[0, 0].plot(data[columns[0]])
            axs[0, 0].set_title(columns[0])
            axs[0, 1].plot(data[columns[1]])
            axs[0, 1].set_title(columns[1])
            axs[1, 0].plot(data[columns[2]])
            axs[1, 0].set_title(columns[2])
            axs[1, 1].plot(data[columns[3]])
            axs[1, 1].set_title(columns[3])
        else:
            axs[0, 0].plot(data[columns[0]], data[columns[1]])
            axs[0, 0].set_title(columns[1])
            axs[0, 1].plot(data[columns[0]], data[columns[2]])
            axs[0, 1].set_title(columns[2])
            axs[1, 0].plot(data[columns[0]], data[columns[3]])
            axs[1, 0].set_title(columns[3])
            axs[1, 1].plot(data[columns[0]], data[columns[4]])
            axs[1, 1].set_title(columns[4])

    def plot_cycle(self, data_dir, index):
        str_index = f'{index:05d}'
        data_1 = pd.read_csv(data_dir + f"{str_index}.csv")
        str_index = f'{index+1:05d}'
        data_2 = pd.read_csv(data_dir + f"{str_index}.csv", sep=",")
        self.plot_data(data_1)
        self.plot_data(data_2)
