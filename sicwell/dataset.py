import os
import pandas as pd
import numpy as np
from .converter import SiCWellConverter

class SiCWellDataset():
    def __init__(self, dataset_config):
        self.capacities = {}
        self.measurement_times = {}
        self.sohs = {}

        self.dataset_config = dataset_config
        self.sicwell_root = self.dataset_config.dataset_root_dir
        self.batteries = self.dataset_config.battery_list
        self.battery_cells = self.batteries
        self.normalize = self.dataset_config.normalize_data
        self.clean_data = self.dataset_config.clean_data
        self.rated_capacity = self.dataset_config.rated_capacity
        self.smooth_data = self.dataset_config.smooth_data
        self.smoothing_kernel_width = self.dataset_config.smoothing_kernel_width
        self.sinusoidal_cycle_path = "cell_cycling_sinusoidal/"
        self.artificial_cycle_path = "cell_cycling_artificial_ripple/"
        self.realistic_cycle_path = "cell_cycling_realistic_ripple/"

        self.sicwell_converter = SiCWellConverter(self.sicwell_root)
        self.get_batteries()
        self.load()
        self.get_dataset_length()
    
    def get_positional_information(self):
        """Extracts positional information from data.
        """
        self.positions = {}
        for data_index, data in self.capacities.items():
            data_length = len(data)
            self.positions[data_index] = list(range(1, data_length+1))

    def get_temporal_information(self):
        """Extracts temporal information from data.
        """
        for cell_id in self.batteries:
            self.measurement_times[cell_id] = self.normalize_data(self.measurement_times[cell_id])
    
    def get_dataset_length(self):
        """Calculates the length of the loaded dataset.
        """
        self.dataset_length = 0
        for caps in self.capacities.values():
            self.dataset_length += len(caps)

    def get_batteries(self):
        if self.batteries in [None, "all"]:
            # load all batteries
            self.batteries = ["AC01", "AC02", "AC03", "AC04", "AC05", "AC06", "AC07", "AC08", "AC09", "AC10", "AC11", "AC12", "AC13", "AC14", "AC15", "AC16", "AC18", "AC18", "AC19", "AC21", "AC22", "AC23", "AC24", "AC25", "AC26", "AC27"]
        elif self.batteries == "artificial":
            # load batteries from artificial cycling
            self.batteries = ["AC19", "AC21"]
        elif self.batteries == "realistic":
            # load batteries from realistic cycling
            self.batteries = ["AC22", "AC23", "AC24", "AC25", "AC26", "AC27"]
        elif self.batteries == "sinusoidal":
            # load batteries from sinusoidal cycling
            self.batteries = ["AC01", "AC02", "AC03", "AC04", "AC05", "AC06", "AC07", "AC08", "AC09", "AC10", "AC11", "AC12", "AC13", "AC14", "AC15", "AC16", "AC17", "AC18"]

    def normalize_data(self, data_list):
        """Normalizes data.
        """
        deltas = [i - min(data_list) for i in data_list]
        normlized_deltas = [i/max(deltas) for i in deltas]
        return normlized_deltas

    def smooth_capacities(self):
        box = np.ones(self.smoothing_kernel_width) / self.smoothing_kernel_width
        box_pts_half = self.smoothing_kernel_width // 2
        for cell_id, cap in self.capacities.items():
            cap_smooth = np.convolve(cap, box, mode="same").flatten().tolist()
            # remove very different values at start and end
            cap_smooth[:box_pts_half] = cap[:box_pts_half]
            cap_smooth[-box_pts_half:] = cap[-box_pts_half:]
            self.capacities[cell_id] = cap_smooth
            # self.capacities[f"{cell_id}_smoothed"] = cap_smooth
        return cap_smooth

    def load_cycling_data(self, path, cycling_type):
        """Loads the overview csv file of the cycling data. 
        """
        print(f"Loading {cycling_type} cycling data")
        cycling_data_file = path + "cycling_data_overview.csv"
        cycling_data = pd.read_csv(cycling_data_file)
        if self.batteries is None:
            self.batteries = cycling_data["Cell_ID"].unique()
        
        for cell_id in self.batteries:
            filtered_cycling_data = cycling_data[cycling_data["Cell_ID"] == cell_id]
            if not filtered_cycling_data.empty:
                self.capacities[cell_id] = list(filtered_cycling_data["Capacity"])
                self.measurement_times[cell_id] = list(filtered_cycling_data["Time"])
                self.sohs[cell_id] = list(filtered_cycling_data["SoH"])
                if self.normalize == "max":
                    self.capacities[cell_id] = [capacity/max(self.capacities[cell_id]) for capacity in self.capacities[cell_id]]
                elif self.normalize == "first":
                    self.capacities[cell_id] = [capacity/self.capacities[cell_id][0] for capacity in self.capacities[cell_id]]
        return cycling_data, self.batteries

    def load(self):
        """Loads all cycling data for SiCWell dataset.
        """
        self.sicwell_converter.convert()
        artificial_cycle_path = self.sicwell_root + self.artificial_cycle_path
        realistic_cycle_path = self.sicwell_root + self.realistic_cycle_path
        sinusoidal_cycle_path = self.sicwell_root + self.sinusoidal_cycle_path
        if os.path.exists(artificial_cycle_path):
            self.artificial_cycle_data, self.artificial_cycle_cells = self.load_cycling_data(artificial_cycle_path, "artificial")
        else:
            print(f"Path {artificial_cycle_path} does not exist")
        if os.path.exists(realistic_cycle_path):  
            self.realistic_cycle_data, self.realistic_cycle_cells = self.load_cycling_data(realistic_cycle_path, "realistic")
        else:
            print(f"Path {realistic_cycle_path} does not exist")
        if os.path.exists(sinusoidal_cycle_path):
            self.sinusoidal_cycle_data, self.sinusoidal_cycle_cells = self.load_cycling_data(sinusoidal_cycle_path, "sinusoidal")
        else:
            print(f"Path {sinusoidal_cycle_path} does not exist")

        self.raw_capacities = self.capacities.copy()
        if self.smooth_data:
            self.smooth_capacities()

