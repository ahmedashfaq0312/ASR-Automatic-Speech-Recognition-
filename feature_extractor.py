import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json

from nasa.dataset import NASADataset
from util import get_config


class NASAFeatureExtractor():
    def __init__(self, config_file_path=f"configs/nasa_config.json"):
        self.config_file_path = config_file_path
        self.config = get_config(config_file_path)
        self.dataset_root = self.config.dataset_config.dataset_root_dir

    def __get_absolute_time_in_range(self, data_values_in_range):
        absolute_time = 0.0
        indices = data_values_in_range.index.to_list()
        for i in range(0, indices[-1]):
            if i in data_values_in_range.index and i+2 in data_values_in_range.index:
                timedelta = self.current_cycle_df.iloc[i+1]["Time"] - self.current_cycle_df.iloc[i]["Time"]
                absolute_time += timedelta
        return absolute_time

    def __get_relative_times_in_ranges(self, df_column):
        """Calculate time of cycle curve in specific ranges."""
        # calculate range thresholds based on percentiles
        per_1_quantile_threshold, per_33_quantile_threshold, per_67_quantile_threshold, per_99_quantile_threshold = self.__calculate_thresholds(df_column)

        # get data in ranges
        data_in_upper_range = self.current_cycle_df[self.current_cycle_df[df_column].between(per_67_quantile_threshold, per_99_quantile_threshold)]
        data_in_middle_range = self.current_cycle_df[self.current_cycle_df[df_column].between(per_33_quantile_threshold, per_67_quantile_threshold)]
        data_in_lower_range = self.current_cycle_df[self.current_cycle_df[df_column].between(per_1_quantile_threshold, per_33_quantile_threshold)]

        # get absolut times in ranges
        absolute_time_upper = self.__get_absolute_time_in_range(data_in_upper_range)
        absolute_time_middle = self.__get_absolute_time_in_range(data_in_middle_range)
        absolute_time_lower = self.__get_absolute_time_in_range(data_in_lower_range)
        
        # get relative times in ranges
        cycle_time = self.current_cycle_df["Time"].iloc[-1] - self.current_cycle_df["Time"].iloc[0]
        relative_time_upper = absolute_time_upper / cycle_time
        relative_time_middle = absolute_time_middle / cycle_time
        relative_time_lower = absolute_time_lower / cycle_time
        
        # return absolute_time_upper, absolute_time_middle, absolute_time_lower, relative_time_upper, relative_time_middle, relative_time_lower
        return relative_time_upper, relative_time_middle, relative_time_lower

    def __calculate_thresholds(self, df_column):
        """Calculate quantiles of data for thresholds."""
        
        per_1_quantile_threshold = self.current_cycle_df[df_column].quantile(0.01)
        per_33_quantile_threshold = self.current_cycle_df[df_column].quantile(0.33)
        per_67_quantile_threshold = self.current_cycle_df[df_column].quantile(0.67)
        per_99_quantile_threshold = self.current_cycle_df[df_column].quantile(0.99)
        
        return per_1_quantile_threshold, per_33_quantile_threshold, per_67_quantile_threshold, per_99_quantile_threshold
    
    def extract_features(self):
        print("Starting feature extraction for NASA dataset")
        features = pd.DataFrame([])

        cells = ["B0005", "B0006", "B0007", "B0018"]
        for cell in cells:
            print(f"Extracting features from {cell} data")
            # get filenames of cycles from metadata
            metadata = pd.read_csv(f"{self.dataset_root}/metadata.csv")
            cell_data = metadata[metadata["battery_id"] == cell]
            cell_charge_data = cell_data[cell_data["type"] == "charge"]
            cell_discharge_data = cell_data[cell_data["type"] == "discharge"]
            charge_filenames = [file for file in cell_charge_data["filename"]]
            discharge_filenames = [file for file in cell_discharge_data["filename"]]

            # process cycle data
            next_charge_time = 0.0
            self.cycle_columns = ["Current_measured", "Voltage_measured", "Temperature_measured"]
            battery_features_dict = {
                "cycle": [],
                "battery_id": [],
                "V_rel_time_upper_range": [],
                "V_rel_time_middle_range": [],
                "V_rel_time_lower_range": [],
                "C_rel_time_upper_range": [],
                "C_rel_time_middle_range": [],
                "C_rel_time_lower_range": [],
                "T_rel_time_upper_range": [],
                "T_rel_time_middle_range": [],
                "T_rel_time_lower_range": []
            }
            for i in range(len(cell_discharge_data)):
                # read and filter cycle data
                self.current_charge_data = pd.read_csv(f"{self.dataset_root}/data/{charge_filenames[i]}")
                self.current_discharge_data = pd.read_csv(f"{self.dataset_root}/data/{discharge_filenames[i]}")
                self.current_charge_data = self.current_charge_data.drop(columns=["Current_charge", "Voltage_charge"])
                self.current_discharge_data = self.current_discharge_data.drop(columns=["Current_load", "Voltage_load"])
                
                # correct time column
                self.current_charge_data["Time"] = self.current_charge_data["Time"].add(next_charge_time)
                self.current_discharge_data["Time"] = self.current_discharge_data["Time"].add(self.current_charge_data["Time"].iloc[-1])
                next_charge_time = self.current_discharge_data["Time"].iloc[-1]
                
                # concat cycle dataframe
                self.current_cycle_df = pd.concat([self.current_charge_data, self.current_discharge_data], ignore_index=True)

                # extract features
                battery_features_dict["cycle"].append(i+1)
                battery_features_dict["battery_id"].append(cell)
                for df_column in self.cycle_columns:
                    relative_time_upper, relative_time_middle, relative_time_lower = self.__get_relative_times_in_ranges(df_column)
                    # append features to dictionary
                    if df_column == "Voltage_measured":
                        battery_features_dict["V_rel_time_upper_range"].append(relative_time_upper)
                        battery_features_dict["V_rel_time_middle_range"].append(relative_time_middle)
                        battery_features_dict["V_rel_time_lower_range"].append(relative_time_lower)
                    elif df_column == "Current_measured":
                        battery_features_dict["C_rel_time_upper_range"].append(relative_time_upper)
                        battery_features_dict["C_rel_time_middle_range"].append(relative_time_middle)
                        battery_features_dict["C_rel_time_lower_range"].append(relative_time_lower)
                    elif df_column == "Temperature_measured":
                        battery_features_dict["T_rel_time_upper_range"].append(relative_time_upper)
                        battery_features_dict["T_rel_time_middle_range"].append(relative_time_middle)
                        battery_features_dict["T_rel_time_lower_range"].append(relative_time_lower)

            # create feature dataframe for cell
            battery_features = pd.DataFrame(battery_features_dict)
            if features.empty:
                features = battery_features
            else:
                features = pd.concat([features, battery_features])

        # save
        features.to_csv(f"{self.dataset_root}/features.csv", index=False)


def extract_nasa_features():
    feature_extractor = NASAFeatureExtractor()
    feature_extractor.extract_features()
    

if __name__ == "__main__":
    extract_nasa_features()