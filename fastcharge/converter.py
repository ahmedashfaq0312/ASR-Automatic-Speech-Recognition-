import os
import mat73
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class FastChargeConverter():
    def __init__(self, fastcharge_root):
        self.fastcharge_root = fastcharge_root
        self.dataset_files = ["2017-05-12_batchdata_updated_struct_errorcorrect.mat", "2017-06-30_batchdata_updated_struct_errorcorrect.mat", "2018-02-20_batchdata_updated_struct_errorcorrect.mat"]
        

    def load_measurement(self, file):
        """Loads mat file.
        """
        return mat73.loadmat(file)["batch"]

    def get_cycle_times(self, data_df):
        times = []
        for cycle in data_df["cycles"]:
            tmp_times = [0.0]
            cycle_times = cycle["t"]
            for time in cycle_times:
                if time is not None:
                    tmp_time = time[-1]*60 # get time in minutes
                    tmp_times.append(tmp_time+tmp_times[-1])
            times.append(tmp_times)
        return times
    
    def concat_and_save_data(self, data, out_file):
        summary = data["summary"]
        policy = data["policy"]
        cycle_lifes = data["cycle_life"]
        fastcharge_summary_df = pd.DataFrame([])
        cycle_times = self.get_cycle_times(data)
        for i in range(len(summary)):
            if policy[i] not in ["3_6C_80PER_3_6C_SLOWCYCLE", "4_8C_80PER_4_8C_SLOWCYCLE"]: # bad measurements
                print(f"Processing {policy[i]}")
                if np.isnan(cycle_lifes[i]):
                    cycle_lifes[i] = -1
                tmp_df = pd.DataFrame(summary[i])
                tmp_df["Experiment"] = policy[i]
                tmp_df["Cycle_Life"] = cycle_lifes[i]
                tmp_df["Time"] = cycle_times[i]
                fastcharge_summary_df = pd.concat([fastcharge_summary_df, tmp_df], ignore_index=True)
        fastcharge_summary_df.to_csv(out_file)

    def convert(self):
        """Converts all cell data.
        """
        for file in self.dataset_files:
            print(f"Converting {file}")
            if file == "2018-02-20_batchdata_updated_struct_errorcorrect.mat":
                out_file = "FastCharge_201802.csv"
            elif file == "2017-06-30_batchdata_updated_struct_errorcorrect.mat":
                out_file = "FastCharge_201706.csv"
            elif file == "2017-05-12_batchdata_updated_struct_errorcorrect.mat":
                out_file = "FastCharge_201705.csv"
            else:
                out_file = "error.csv"
            if not os.path.exists(f"{self.fastcharge_root}/{out_file}") and os.path.exists(f"{self.fastcharge_root}/{file}"):
                data = self.load_measurement(f"{self.fastcharge_root}/{file}")
                self.concat_and_save_data(data, f"{self.fastcharge_root}/{out_file}")
            
        
