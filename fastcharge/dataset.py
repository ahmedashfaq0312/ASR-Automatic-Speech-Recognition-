import os
import pandas as pd
import numpy as np
from .converter import FastChargeConverter
try:
    from rul_estimation_datasets.dataset_utils import filter_rows, get_eol_information
except:
    from rul_estimation.rul_estimation_datasets.dataset_utils import filter_rows, get_eol_information

class FastChargeDataset():
    def __init__(self, dataset_config):
        self.rated_capacity = 1.1
        self.dataset_config = dataset_config
        self.fastcharge_root = self.dataset_config.dataset_root_dir
        self.train_cells = self.dataset_config.train_cells
        self.test_cells = self.dataset_config.test_cells
        self.normalize = self.dataset_config.normalize_data
        self.clean_data = self.dataset_config.clean_data
        self.rated_capacity = self.dataset_config.rated_capacity
        self.smooth_data = self.dataset_config.smooth_data
        self.smoothing_kernel_width = self.dataset_config.smoothing_kernel_width
        self.concat_data_file_times = ["201705", "201706", "201802"]

        self.fastcharge_converter = FastChargeConverter(self.fastcharge_root)
        self.load()
        self.get_dataset_length()
    
    def get_positional_information(self, data_df):
        """Extracts positional information from data.
        """
        return data_df["cycle"].to_list()

    def get_temporal_information(self, data_df):
        """Extracts temporal information from data.
        """
        return data_df["Time"].to_list()

    def get_dataset_length(self):
        """Calculates the length of the loaded dataset.
        """
        self.train_dataset_length = len(self.train_df["Capacity"])
        self.test_dataset_length = len(self.test_df["Capacity"])

    def normalize_capacities(self):
        """Normalizes capacities.
        """
        train_capacities = self.raw_train_df["QDischarge"].astype(float)
        normalized__train_capacities = [i/self.rated_capacity for i in train_capacities]
        self.train_df["Cell_ID"] = self.raw_train_df["Experiment"].to_list()
        self.train_df["Capacity"] = normalized__train_capacities
        test_capacities = self.raw_test_df["QDischarge"].astype(float)
        normalized_test_capacities = [i/self.rated_capacity for i in test_capacities]
        self.test_df["Cell_ID"] = self.raw_test_df["Experiment"].to_list()
        self.test_df["Capacity"] = normalized_test_capacities

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

    def load_cell_data(self):
        self.data_df = pd.DataFrame([])
        for file_time in self.concat_data_file_times:
            file = f"{self.fastcharge_root}/FastCharge_{file_time}.csv"
            if os.path.exists(file):
                data = pd.read_csv(file, index_col=0)
                # experiments = data["Experiment"].unique().tolist()
                # for experiment in experiments:
                #     tmp_data = data[data["Experiment"] == experiment]
                #     self.capacities[experiment] = tmp_data["QDischarge"].tolist()
                #     eol = int(tmp_data["Cycle_Life"].tolist()[0])
                #     self.eols[experiment] = eol
                # if self.data_df is None:
                #     self.data_df = data
                # else:
                self.data_df = pd.concat([self.data_df, data])
            else:
                print(f"Path {file} does not exist")

    def preprocess(self):
        self.train_df = pd.DataFrame([])
        self.test_df = pd.DataFrame([])
        self.raw_train_df = filter_rows(self.data_df, "Experiment", self.train_cells)
        self.raw_test_df = filter_rows(self.data_df, "Experiment", self.test_cells)
        self.train_df["Cycle"] = self.get_positional_information(self.raw_train_df)
        self.test_df["Cycle"] = self.get_positional_information(self.raw_test_df)
        # self.train_df["Time"] = self.get_temporal_information(self.raw_train_df)
        # self.test_df["Time"] = self.get_temporal_information(self.raw_test_df)

        if self.normalize:
            self.normalize_capacities()
        if self.smooth_data:
            self.smooth_capacities()
        get_eol_information(self.train_df, self.normalize, self.rated_capacity)
        get_eol_information(self.test_df, self.normalize, self.rated_capacity)
        i=0

    def load(self):
        """Loads all cycling data for FastCharge dataset.
        """
        self.fastcharge_converter.convert()
        self.load_cell_data()
        self.preprocess()

