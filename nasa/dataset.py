import pandas as pd
import numpy as np
import ast
import re
from datetime import datetime
import collections
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
try:
    from __init__ import RUL_DATASETS_ROOT_PATH
    from nasa.converter import NASAConverter
    from nasa.downloader import NASADownloader
    from dataset_utils import filter_rows, get_positional_information, get_eol_information
    from util import get_config
except:
    try:
        from rul_estimation_datasets import RUL_DATASETS_ROOT_PATH
        from rul_estimation_datasets.nasa.converter import NASAConverter
        from rul_estimation_datasets.nasa.downloader import NASADownloader
        from rul_estimation_datasets.dataset_utils import filter_rows, get_positional_information, get_eol_information
        from util import get_config
    except:
        from rul_estimation.rul_estimation_datasets import RUL_DATASETS_ROOT_PATH
        from rul_estimation.rul_estimation_datasets.nasa.converter import NASAConverter
        from rul_estimation.rul_estimation_datasets.nasa.downloader import NASADownloader
        from rul_estimation.rul_estimation_datasets.dataset_utils import filter_rows, get_positional_information, get_eol_information
        from rul_estimation.util import get_config

class NASADataset():
    """Class for preprocessing and loading NASA battery dataset.
    """
    def __init__(self, nasa_config=None) -> None: # batteries: ["all", "B0005", "["B0005", "B0006", "B0007", ...]"]; normalize: [None, "max", "first"]
        if nasa_config is None:
            self.nasa_config = get_config(f"{RUL_DATASETS_ROOT_PATH}/configs/nasa_config.json")
            self.nasa_root = self.nasa_config.dataset_root_dir
        else:
            self.nasa_config = nasa_config.dataset_config
            self.nasa_root = self.nasa_config.dataset_root_dir
        
        self.dataset_dir = f"{self.nasa_root}/data"
        self.train_batteries = self.nasa_config.train_cells
        if self.train_batteries == "all":
            self.train_cells = ['B0005', 'B0006', 'B0007', 'B0018', 'B0025', 'B0026', 'B0027', 'B0028', 'B0029', 'B0030', 'B0031', 'B0032', 'B0033', 'B0034', 'B0036', 'B0038', 'B0039', 'B0040', 'B0041', 'B0042', 'B0043', 'B0044', 'B0045', 'B0046', 'B0047', 'B0048', 'B0049', 'B0050', 'B0051', 'B0052', 'B0053', 'B0054', 'B0055', 'B0056']
        else:
            self.train_cells = self.train_batteries
        self.test_cells = self.nasa_config.test_cells
        self.normalize = self.nasa_config.normalize_data
        self.clean_data = self.nasa_config.clean_data
        self.rated_capacity = 2.0
        self.smooth_data = self.nasa_config.smooth_data
        self.smoothing_kernel_width = self.nasa_config.smoothing_kernel_width
        self.downloader = NASADownloader(output_path=self.nasa_root+"_raw")
        self.converter = NASAConverter(nasa_dir=self.nasa_root+"_raw", output_dir=self.nasa_root)
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

    def extract_measurement_times(self, data_df, battery_id="", measurement_type="discharge"):
        """Extracts measurement times from metadata.
        """
        measurement_times = []
        epoch_time = datetime(1970, 1, 1)
    
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
            measurement_times.append(timestamp_seconds)
        return measurement_times

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

    def normalize_measurement_times(self, measurement_times):
        """Normalizes data.
        """
        normalized_times = [i-min(measurement_times) for i in measurement_times]
        return normalized_times
    
    def normalize_capacities(self):
        """Normalizes capacities.
        """
        train_capacities = self.raw_train_df["Capacity"].astype(float)
        normalized__train_capacities = [i/self.rated_capacity for i in train_capacities]
        self.train_df["Cell_ID"] = self.raw_train_df["battery_id"].to_list()
        self.train_df["Capacity"] = normalized__train_capacities
        test_capacities = self.raw_test_df["Capacity"].astype(float)
        normalized_test_capacities = [i/self.rated_capacity for i in test_capacities]
        self.test_df["Cell_ID"] = self.raw_test_df["battery_id"].to_list()
        self.test_df["Capacity"] = normalized_test_capacities
        
    def normalize_times(self):
        """Normalizes times.
        """
        train_times = self.train_df["Time"].astype(float)
        normalized_train_times = [i/max(train_times) for i in train_times]
        remaining_train_times = [1-i for i in normalized_train_times]
        self.train_df["Time"] = normalized_train_times
        self.train_df["Remaining Time"] = remaining_train_times
        test_times = self.test_df["Time"].astype(float)
        normalized_test_times = [i/max(train_times) for i in test_times]
        remaining_test_times = [1-i for i in normalized_test_times]
        self.test_df["Time"] = normalized_test_times
        self.test_df["Remaining Time"] = remaining_test_times

    def get_temporal_information(self, data_df):
        """Extracts temporal information from data.
        """
        measurement_times = []
        unique_cells = data_df["battery_id"].unique()
        for cell in unique_cells:
            # cell_df = data_df[data_df["battery_id"] == cell]
            tmp_times = self.extract_measurement_times(data_df, battery_id=cell)
            normalized_measurement_times = self.normalize_measurement_times(tmp_times)
            measurement_times.extend(normalized_measurement_times)
        return measurement_times


    def get_eol_information(self):
        get_eol_information(self.train_df, self.normalize, self.rated_capacity)
        get_eol_information(self.test_df, self.normalize, self.rated_capacity)
        
        # if train_method is oneshot
        self.train_df = self.train_df[self.train_df["Cycles_to_EOL"] >= 0]
        self.test_df = self.test_df[self.test_df["Cycles_to_EOL"] >= 0]

    def get_dataset_length(self):
        self.train_dataset_length = len(self.train_df["Capacity"])
        self.trest_dataset_length = len(self.test_df["Capacity"])

    def get_additional_features(self):
        filtered_train_features = filter_rows(self.additional_features, "battery_id", self.train_cells)
        filtered_train_features = filtered_train_features.drop(["cycle", "battery_id"], axis=1)
        filtered_test_features = filter_rows(self.additional_features, "battery_id", self.test_cells)
        filtered_test_features = filtered_test_features.drop(["cycle", "battery_id"], axis=1)
        return filtered_train_features, filtered_test_features
        
    def preprocessing(self):
        self.train_df = pd.DataFrame([])
        self.test_df = pd.DataFrame([])
        self.raw_train_df = filter_rows(self.metadata, "battery_id", self.train_cells)
        self.raw_test_df = filter_rows(self.metadata, "battery_id", self.test_cells)
        self.raw_train_df = self.raw_train_df[self.raw_train_df["type"] == "discharge"]
        self.raw_test_df = self.raw_test_df[self.raw_test_df["type"] == "discharge"]
        self.train_df["Cycle"] = get_positional_information(self.raw_train_df, cell_column_name="battery_id")
        self.test_df["Cycle"] = get_positional_information(self.raw_test_df, cell_column_name="battery_id")
        self.train_df["Time"] = self.get_temporal_information(self.raw_train_df)
        self.test_df["Time"] = self.get_temporal_information(self.raw_test_df)
        
        self.additional_train_features, self.additional_test_features = self.get_additional_features()
        self.train_df = pd.concat([self.train_df, self.additional_train_features], axis=1)
        self.test_df = pd.concat([self.test_df, self.additional_test_features], axis=1)

        if self.normalize:
            self.normalize_capacities()
            self.normalize_times()
        # if self.smooth_data:
        #     self.smooth_capacities()
        self.get_eol_information()


    def load(self):
        """Loads NASA dataset.
        """
        self.downloader.download_and_extract()
        self.converter.convert()
        self.metadata = pd.read_csv(f"{self.nasa_root}/metadata.csv")
        self.additional_features = pd.read_csv(f"{self.nasa_root}/features.csv")
        self.preprocessing()
        self.get_dataset_length()
    
    # def load(self):
    #     """Loads NASA dataset.
    #     """
    #     username = "powerized_db_user"
    #     password = "powerized_db_data"
    #     host = "192.168.111.4"
    #     port = 5432
    #     engine = create_engine(f"postgresql+pg8000://{username}:{password}@{host}:{port}/NASA", echo=False) # PostgreSQL
    #     self.metadata = pd.read_sql_table('metadata', con=engine, index_col=0)
    #     self.preprocessing()
    #     self.get_dataset_length()
    #     i=0
    
    def load_cycle_data_to_db(self, engine):
        for bat_id in self.metadata["battery_id"].unique():
            bat_metadata = self.metadata[self.metadata["battery_id"] == bat_id]
            bat_cycle_files = bat_metadata["filename"].to_list()
            cycle_number = 1
            battery_cycle_data = pd.DataFrame([])
            print(bat_id)
            for cycle_file in bat_cycle_files:
                cycle_data = pd.read_csv(f"{self.dataset_dir}/{cycle_file}")
                # create array with cycle number and insert to cycle data
                cycle_number_array = [cycle_number for _ in range(len(cycle_data))]
                cycle_data.insert(0, "cycle", cycle_number_array, allow_duplicates=True)
                # replace dataframe or append to dataframe
                if battery_cycle_data.empty:
                    battery_cycle_data = cycle_data
                else:
                    battery_cycle_data = pd.concat([battery_cycle_data, cycle_data], ignore_index=True)

                cycle_number += 1

            battery_cycle_data.to_sql(name=bat_id, con=engine, chunksize=65535, if_exists="replace")

    def load_data_to_db(self, username, password, host, port):
        # create connection to database
        engine = create_engine(f"postgresql+pg8000://{username}:{password}@{host}:{port}/NASA", echo=False) # PostgreSQL
        # check if database exists
        if not database_exists(engine.url):
            create_database(engine.url)
        # write metadata to db
        self.metadata.to_sql(name='metadata', con=engine, if_exists="replace")
        # load cycle data for each cell to db
        self.load_cycle_data_to_db(engine)
        # close connection
        engine.dispose()