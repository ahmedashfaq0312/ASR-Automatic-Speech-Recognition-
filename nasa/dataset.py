import pandas as pd
import ast
import re
from datetime import datetime
import collections

class NASADataset():
    def __init__(self, batteries="all", normalize=None, clean_dataset=True) -> None: # batteries: ["all", "B0005", "["B0005", "B0006", "B0007", ...]"]; normalize: [None, "max", "first"]
        self.nasa_root = "NASA"
        self.dataset_dir = f"{self.nasa_root}/data"
        self.normalize = normalize
        self.clean_dataset = clean_dataset
        self.metadata = pd.read_csv(f"{self.nasa_root}/metadata.csv")
        if batteries == "all":
            self.capacities = self.extract_capacities()
            self.Res, self.Rcts = self.extract_resistances()
        else:
            self.metadata = self.filter_rows(self.metadata, "battery_id", batteries)
            self.capacities = self.extract_capacities()
            self.Res, self.Rcts = self.extract_resistances()

    def clean(self, data, thresh=0.1):
        peaks = []
        for i in range(len(data)-2):
            diff_1 = abs(data[i] - data[i+1]) # difference to the next value
            # diff_2 = abs(data[i] - data[i+2]) # difference to the value after the next value
            if diff_1 > thresh: # and diff_2 < thresh: # only detect and remove single value peaks
                if i not in peaks:
                    data[i+1] = data[i]
                    peaks.append(i+1)
        return data
    
    def extract_capacities(self):
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
        Capacities = collections.OrderedDict(sorted(Capacities.items()))
        return Capacities
    
    def extract_resistances(self):
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
        
        Res = collections.OrderedDict(sorted(Res.items()))
        Rcts = collections.OrderedDict(sorted(Rcts.items()))
        return Res, Rcts

    def extract_measurement_times(self, battery_id="", measurement_type="discharge"):
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
        return_df = pd.DataFrame([0])
        # return_df = None
        if type(attribute) == str:
            return_df = data_df[data_df[column_name] == attribute]
        elif type(attribute) == list:
            return_df = data_df[data_df[column_name].isin(attribute)]
        return return_df

    def normalize_data(self, data_list):
        deltas = [i - min(data_list) for i in data_list]
        normlized_deltas = [i/max(deltas) for i in deltas]
        return normlized_deltas
    
    def attach_positional_information(self, data_dict):
        return_data_dict = {}
        for data_index, data in data_dict.items():
            data_length = len(data)
            positional_information = list(range(1, data_length+1))
            return_data_dict[data_index] = [positional_information, data]
        return return_data_dict
    
    def attach_temporal_information(self, data_dict):
        return_data_dict = {}
        for data_index, data in data_dict.items():
            measurement_times = self.extract_measurement_times(battery_id=data_index)
            normalized_measurement_times = self.normalize_data(measurement_times[data_index])
            if len(data) > 2:
                data = [data, normalized_measurement_times]
            else:
                data.append(normalized_measurement_times)
            return_data_dict[data_index] = data
        return return_data_dict
        
    
    def load(self, filter_attributes=[]):
        pass