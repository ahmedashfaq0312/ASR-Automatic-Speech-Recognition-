import pandas as pd
import ast
import re
from datetime import datetime
import collections
from .downloader import NASADownoader
from .converter import NASAConverter

class NASADataset():
    """Class for preprocessing and loading NASA battery dataset.
    """
    def __init__(self, batteries="all", normalize=None, clean_dataset=True) -> None: # batteries: ["all", "B0005", "["B0005", "B0006", "B0007", ...]"]; normalize: [None, "max", "first"]
        self.nasa_root = "NASA"
        self.dataset_dir = f"{self.nasa_root}/data"
        self.batteries = batteries
        self.normalize = normalize
        self.clean_dataset = clean_dataset
        self.downloader = NASADownoader()
        self.converter = NASAConverter()
        self.load()

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
            if self.clean_dataset:
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

        
    
    def load(self):
        """Loads NASA dataset.
        """
        self.downloader.download_and_extract()
        self.converter.convert(self.batteries)
        self.metadata = pd.read_csv(f"{self.nasa_root}/metadata.csv")
        if self.batteries == "all":
            self.extract_capacities()
            self.extract_resistances()
        else:
            self.metadata = self.filter_rows(self.metadata, "battery_id", self.batteries)
            self.extract_capacities()
            self.extract_resistances()