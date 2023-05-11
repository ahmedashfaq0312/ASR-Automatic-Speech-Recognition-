import pandas as pd
import ast
import re
from datetime import datetime
import collections

class NASADataset():
    def __init__(self, batteries="all", normalize=None) -> None: # batteries: ["all", "B0005", "["B0005", "B0006", "B0007", ...]"]; normalize: [None, "max", "first"]
        self.nasa_root = "NASA"
        self.dataset_dir = f"{self.nasa_root}/data"
        self.normalize = normalize
        self.metadata = pd.read_csv(f"{self.nasa_root}/metadata.csv")
        if batteries == "all":
            self.capacities = self.extract_capacities()
            self.Res, self.Rcts = self.extract_resistances()
        else:
            self.metadata = self.filter_rows(self.metadata, "battery_id", batteries)
            self.capacities = self.extract_capacities()
            self.Res, self.Rcts = self.extract_resistances()
        
    
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
        if self.normalize != None:
            for bat_id, capacities in Capacities.items():
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

    def extract_measurement_times(self, measurement_type="discharge"):
        discharge_times = {}
        impedance_times = {}
        
        epoch_time = datetime(1970, 1, 1)
        
        discharge_df = self.metadata[self.metadata["type"] == "discharge"]
        for _, df_line in discharge_df.iterrows():
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
    
    def attach_positional_information(self, data_dict):
        return_data_dict = {}
        for data_index, data in data_dict.items():
            data_length = len(data)
            positional_information = list(range(1, data_length+1))
            return_data_dict[data_index] = [positional_information, data]
        return return_data_dict
            
        
    
    def load(self, filter_attributes=[]):
        pass