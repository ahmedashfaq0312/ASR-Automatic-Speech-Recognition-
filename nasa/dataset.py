import pandas as pd
import numpy as np
import ast
import re
from datetime import datetime
import collections
from .downloader import NASADownoader
from .converter import NASAConverter

class NASADataset():
    """Class for preprocessing and loading NASA battery dataset.
    """
    def __init__(self, dataset_config) -> None: # batteries: ["all", "B0005", "["B0005", "B0006", "B0007", ...]"]; normalize: [None, "max", "first"]
        self.dataset_config = dataset_config
        self.nasa_root = self.dataset_config.dataset_root_dir
        self.dataset_dir = f"{self.nasa_root}/data"
        self.train_batteries = self.dataset_config.train_cells
        if self.train_batteries == "all":
            self.train_cells = ['B0005', 'B0006', 'B0007', 'B0018', 'B0025', 'B0026', 'B0027', 'B0028', 'B0029', 'B0030', 'B0031', 'B0032', 'B0033', 'B0034', 'B0036', 'B0038', 'B0039', 'B0040', 'B0041', 'B0042', 'B0043', 'B0044', 'B0045', 'B0046', 'B0047', 'B0048', 'B0049', 'B0050', 'B0051', 'B0052', 'B0053', 'B0054', 'B0055', 'B0056']
        else:
            self.train_cells = self.train_batteries
        self.test_cells = self.dataset_config.test_cells
        self.normalize = self.dataset_config.normalize_data
        self.clean_data = self.dataset_config.clean_data
        self.rated_capacity = self.dataset_config.rated_capacity
        self.smooth_data = self.dataset_config.smooth_data
        self.smoothing_kernel_width = self.dataset_config.smoothing_kernel_width
        self.downloader = NASADownoader(output_path=self.nasa_root+"_raw")
        self.converter = NASAConverter(nasa_dir=self.nasa_root+"_raw", output_dir=self.nasa_root)
        self.load()
        self.get_dataset_length()

    def clean(self, data, thresh=0.1):
        """Clean peaks from data.

        Args:
            data (list): Data to be cleaned
            thresh (float, optional): Threshold for minimum peak height. Defaults to 0.1.
        """
        peaks = []
        for i in range(len(data)-2):
            diff_1 = abs(data[i] - data[i+1]) # difference to the next value
            if diff_1 > thresh: # and diff_2 < thresh: # only detect and remove single value peaks
                if i not in peaks:
                    data[i+1] = data[i]
                    peaks.append(i+1)
        return data

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
    
    def extract_capacities(self):
        """Extracts capacities from metadata.
        """
        Capacities = {}
        discharge_df = self.metadata[self.metadata["type"] == "discharge"]
        for _, df_line in discharge_df.iterrows():
            try:
                capacity = float(df_line["Capacity"])
            except ValueError:
                pass
            if df_line["battery_id"] not in Capacities:
                Capacities[df_line["battery_id"]] = []
            Capacities[df_line["battery_id"]].append(capacity)
        for bat_id, capacities in Capacities.items():
            if self.clean_data:
                capacities = self.clean(capacities)
            if self.normalize == "max":
                Capacities[bat_id] = [capacity/max(capacities) for capacity in capacities]
            elif self.normalize == "first":
                Capacities[bat_id] = [capacity/capacities[0] for capacity in capacities]
        self.capacities = collections.OrderedDict(sorted(Capacities.items()))
    
    def extract_resistances(self):
        """Extracts resistances from metadata.
        """
        Res = {}
        Rcts = {}
        impedance_df = self.metadata[self.metadata["type"] == "impedance"]
        for _, df_line in impedance_df.iterrows():
            try:
                re = float(df_line["Re"])
                rct = float(df_line["Rct"])
            except ValueError:
                re = float(abs(complex(df_line["Re"])))
                rct = float(abs(complex(df_line["Rct"])))
            if df_line["battery_id"] not in Res:
                Res[df_line["battery_id"]] = []
            if df_line["battery_id"] not in Rcts:
                Rcts[df_line["battery_id"]] = []
            Res[df_line["battery_id"]].append(re)
            Rcts[df_line["battery_id"]].append(rct)
        
        self.Res = collections.OrderedDict(sorted(Res.items()))
        self.Rcts = collections.OrderedDict(sorted(Rcts.items()))

    def extract_measurement_times(self, battery_id="", measurement_type="discharge"):
        """Extracts measurement times from metadata.
        """
        discharge_times = {}
        impedance_times = {}
        
        epoch_time = datetime(1970, 1, 1)
        
        data_df = self.metadata[self.metadata["type"] == measurement_type]
        if battery_id != "":
            data_df = data_df[data_df["battery_id"] == battery_id]
        for _, df_line in data_df.iterrows():
            try:
                timestamp = df_line["start_time"]
                timestamp = re.sub(' +', ', ', timestamp) # get string representation of list into right format
                timestamp = ast.literal_eval(timestamp) # convert string representation of list to list
                timestamp = [int(stamp) for stamp in timestamp] # convert float list values to integer
                timestamp = datetime(*timestamp) # create datetime object from list of date information
                timestamp_seconds = (timestamp-epoch_time).total_seconds() # calculate total seconds from starttime
            except ValueError:
                print("Error when reading timestamp")
                pass
            if df_line["battery_id"] not in discharge_times:
                discharge_times[df_line["battery_id"]] = []
            discharge_times[df_line["battery_id"]].append(timestamp_seconds)
        return discharge_times

    def extract_temperatures(self):
        """Extracts resistances from metadata.
        """
        ambient_temperatures_charge = {}
        ambient_temperatures_discharge = {}
        charge_df = self.metadata[self.metadata["type"] == "charge"]
        discharge_df = self.metadata[self.metadata["type"] == "discharge"]
        for _, df_line in charge_df.iterrows():
            ambient_temperature = float(df_line["ambient_temperature"])
            if df_line["battery_id"] not in ambient_temperatures_charge:
                ambient_temperatures_charge[df_line["battery_id"]] = []
            ambient_temperatures_charge[df_line["battery_id"]].append(ambient_temperature)
        
        for _, df_line in discharge_df.iterrows():
            ambient_temperature = float(df_line["ambient_temperature"])
            if df_line["battery_id"] not in ambient_temperatures_discharge:
                ambient_temperatures_discharge[df_line["battery_id"]] = []
            ambient_temperatures_discharge[df_line["battery_id"]].append(ambient_temperature)
        
        self.ambient_temperatures_charge = collections.OrderedDict(sorted(ambient_temperatures_charge.items()))
        self.ambient_temperatures_discharge = collections.OrderedDict(sorted(ambient_temperatures_discharge.items()))
        
    
    def filter_rows(self, data_df, column_name, attribute):
        """Filters rows of specific colums with specific values.
        """
        return_df = pd.DataFrame([0])
        # return_df = None
        if type(attribute) == str:
            return_df = data_df[data_df[column_name] == attribute]
        elif type(attribute) == list:
            return_df = data_df[data_df[column_name].isin(attribute)]
        return return_df

    def normalize_data(self, data_list):
        """Normalizes data.
        """
        deltas = [i - min(data_list) for i in data_list]
        normlized_deltas = [i/max(deltas) for i in deltas]
        return normlized_deltas
    
    def normalize_capacities(self):
        """Normalizes capacities.
        """
        train_capacities = self.raw_train_df[self.raw_train_df["type"] == "discharge"]["Capacity"].astype(float)
        normalized__train_capacities = [i/self.rated_capacity for i in train_capacities]
        self.train_df["Cell_ID"] = self.raw_train_df[self.raw_train_df["type"] == "discharge"]["battery_id"]
        self.train_df["Capacity"] = normalized__train_capacities
        test_capacities = self.raw_test_df[self.raw_test_df["type"] == "discharge"]["Capacity"].astype(float)
        normalized_test_capacities = [i/self.rated_capacity for i in test_capacities]
        self.test_df["Cell_ID"] = self.raw_test_df[self.raw_test_df["type"] == "discharge"]["battery_id"]
        self.test_df["Capacity"] = normalized_test_capacities

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
        self.measurement_times = {}
        for data_index, _ in self.capacities.items():
            measurement_times = self.extract_measurement_times(battery_id=data_index)
            normalized_measurement_times = self.normalize_data(measurement_times[data_index])
            self.measurement_times[data_index] = normalized_measurement_times
    
    def get_eol_information_df(self, data_df):
        self.unique_cells = data_df["Cell_ID"].unique()
        eol_criterion = 0.7 if self.normalize else self.rated_capacity*0.7
        eols = []
        cycles_until_eols = []
        for cell in self.unique_cells:
            caps = data_df[data_df["Cell_ID"] == cell]["Capacity"].astype(float)
            try:
                # calculate cycle where EOL is reached (if EOL not reached, cycle is set to -1)
                eol_idx = next(x for x, val in enumerate(caps) if val <= eol_criterion)
            except StopIteration:
                eol_idx = -1
            self.eols[cell] = eol_idx
            eols.extend([eol_idx for _ in range(len(caps))])
            if eol_idx == -1:
                cycles_until_eol = [-1 for i in range(len(caps))]
            else:
                cycles_until_eol = [eol_idx - i for i in range(len(caps))]
            cycles_until_eols.extend(cycles_until_eol)
        data_df["EOL_cycle"] = eols
        data_df["Cycles_to_EOL"] = cycles_until_eols


    def get_eol_information(self):
        self.eols = {}
        self.cycles_until_eol = {}
        
        self.get_eol_information_df(self.train_df)
        self.get_eol_information_df(self.test_df)
        

    def get_dataset_length(self):
        self.train_dataset_length = len(self.train_df["Capacity"])
        self.trest_dataset_length = len(self.test_df["Capacity"])


    def load(self):
        """Loads NASA dataset.
        """
        self.train_df = pd.DataFrame([])
        self.test_df = pd.DataFrame([])
        self.downloader.download_and_extract()
        self.converter.convert()
        self.metadata = pd.read_csv(f"{self.nasa_root}/metadata.csv")

        self.raw_train_df = self.filter_rows(self.metadata, "battery_id", self.train_cells)
        self.raw_test_df = self.filter_rows(self.metadata, "battery_id", self.test_cells)

        if self.normalize:
            self.normalize_capacities()
        if self.smooth_data:
            self.smooth_capacities()
        self.get_eol_information()

