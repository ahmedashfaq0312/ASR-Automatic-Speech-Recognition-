import os
import pandas as pd
import numpy as np
from .converter import FastChargeConverter

class FastChargeDataset():
    def __init__(self, dataset_config):
        self.capacities = {}
        self.eols = {}
        self.measurement_times = {}
        self.sohs = {}

        self.rated_capacity = 1.1
        self.dataset_config = dataset_config
        self.fastcharge_root = self.dataset_config.dataset_root_dir
        self.batteries = self.dataset_config.battery_list
        self.battery_cells = self.batteries
        self.normalize = self.dataset_config.normalize_data
        self.clean_data = self.dataset_config.clean_data
        self.rated_capacity = self.dataset_config.rated_capacity
        self.smooth_data = self.dataset_config.smooth_data
        self.smoothing_kernel_width = self.dataset_config.smoothing_kernel_width
        self.concat_data_file_times = ["201705", "201706", "201802"]

        self.fastcharge_converter = FastChargeConverter(self.fastcharge_root)
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

    def normalize_capacities(self):
        """Normalizes capacities.
        """
        for cell_id, cap in self.capacities.items():
            normalized_capacities = [i/self.rated_capacity for i in cap]
            self.capacities[cell_id] = normalized_capacities

    def smooth_capacities(self):
        box = np.ones(self.smoothing_kernel_width) / self.smoothing_kernel_width
        box_pts_half = self.smoothing_kernel_width // 2
        for cell_id, cap in self.capacities.items():
            cap_smooth = np.convolve(cap, box, mode="same").flatten().tolist()
            # remove very different values at start and end
            cap_smooth[:box_pts_half] = cap[:box_pts_half]
            cap_smooth[-box_pts_half:] = cap[-box_pts_half:]
            self.capacities[cell_id] = cap_smooth
        return cap_smooth
    
    def get_eol_information(self):
        self.cycles_until_eol = {}
        for cell, caps in self.capacities.items():
            eol = self.eols[cell]
            cycles_until_eol = [eol - i for i in range(len(caps))]
            self.cycles_until_eol[cell] = cycles_until_eol

    def load(self):
        """Loads all cycling data for FastCharge dataset.
        """
        self.fastcharge_converter.convert()
        
        for file_time in self.concat_data_file_times:
            file = f"{self.fastcharge_root}/FastCharge_{file_time}.csv"
            if os.path.exists(file):
                data = pd.read_csv(file, index_col=0)
                experiments = data["Experiment"].unique().tolist()
                for experiment in experiments:
                    tmp_data = data[data["Experiment"] == experiment]
                    self.capacities[experiment] = tmp_data["QDischarge"].tolist()
                    eol = int(tmp_data["Cycle_Life"].tolist()[0])
                    self.eols[experiment] = eol
            else:
                print(f"Path {file} does not exist")
        if self.normalize:
            self.normalize_capacities()
        if self.smooth_data:
            self.smooth_capacities()
        self.get_eol_information()

