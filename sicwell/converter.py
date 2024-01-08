import os
import pandas as pd
import h5py

class SiCWellConverter():
    def __init__(self, sicwell_root):
        self.sicwell_root = sicwell_root
        self.artificial_cycle_path = "cell_cycling_artificial_ripple/"
        self.realistic_cycle_path = "cell_cycling_realistic_ripple/"
        self.sinusoidal_cycle_path = "cell_cycling_sinusoidal/"
        

    def load_measurement(self, file):
        """Loads hdf5 file and extracts metadata from it.
        """
        with h5py.File(file,'r') as f:
            grp = f['measurement0000']
            capacity = grp.attrs['capacity']
            internal_resistance = grp.attrs['internal_resistance']
            sampling_rate = grp.attrs['sampling_rate']
            soh_capacity = grp.attrs['soh_capacity']
            temperature = grp.attrs['temperature']
            time = grp.attrs['time']

        return capacity, internal_resistance, sampling_rate, soh_capacity, temperature, time

    def load_cell_measurements(self, path, cell_id):
        """Loads all measurements of a specific cell and returns it as a pandas DataFrame.
        """
        print(f"Loading {cell_id} measurements")
        capacities = []
        sohs = []
        times = []
        internal_resistances = []
        sampling_rates = []
        temperatures = []

        measurement_path = path + cell_id
        file_list = [measurement_path + "/" + file for file in os.listdir(measurement_path) if file.endswith(".hdf5")]
        filename_list = [os.path.split(file)[1] for file in file_list]
        filename_list_sorted = sorted(filename_list)
        for file in filename_list_sorted:
            file_path = measurement_path + "/" + file
            try:
                capacity, internal_resistance, sampling_rate, soh_capacity, temperature, time = self.load_measurement(file_path)
            except OSError:
                continue
            capacities.append(capacity)
            sohs.append(soh_capacity)
            times.append(time)
            internal_resistances.append(internal_resistance)
            sampling_rates.append(sampling_rate)
            temperatures.append(temperature)

        times = [(time - times[0]) for time in times]
        ids = range(len(capacities))
        cell_ids = [cell_id] * len(capacities)

        cell_data = pd.DataFrame(
            list(zip(ids, cell_ids, filename_list_sorted, times, capacities, internal_resistances, sohs, temperatures, sampling_rates)),
            columns=["ID", "Cell_ID", "Filename", "Time", "Capacity", "Internal_Resistance", "SoH", "Temperature", "Sampling_Rate"]
        )
        return cell_data

    def convert_cell_data(self, cycle_path, cycling_type):
        """Collects data from all cell measurements and saves it as a csv file.
        """
        cycling_data = pd.DataFrame()
        output_path = cycle_path + "cycling_data_overview.csv"

        if not os.path.exists(output_path):
            print(f"Converting {cycling_type} cycling data")
            for _, subdirs, _ in os.walk(cycle_path):
                cell_ids = subdirs
                break

            for cell_id in cell_ids:
                # Skip measurements for cells AC22, AC23 and AC24 
                # since there is an error for capacity degradation in the dataset (no degradation)
                if cell_id in ["AC22", "AC23", "AC24"]:
                    continue
                cell_df = self.load_cell_measurements(cycle_path, cell_id)
                if cycling_data.empty:
                    cycling_data = cell_df
                else:
                    cycling_data = pd.concat([cycling_data, cell_df])
            cycling_data = cycling_data.reset_index(drop=True)

            cycling_data.to_csv(output_path)

    def convert(self):
        """Converts all cell data.
        """
        if os.path.exists(self.sicwell_root + self.artificial_cycle_path):
            self.convert_cell_data(self.sicwell_root + self.artificial_cycle_path, "artificial")
        if os.path.exists(self.sicwell_root + self.realistic_cycle_path):  
            self.convert_cell_data(self.sicwell_root + self.realistic_cycle_path, "realistic")
        if os.path.exists(self.sicwell_root + self.sinusoidal_cycle_path):
            self.convert_cell_data(self.sicwell_root + self.sinusoidal_cycle_path, "sinusoidal")
        
