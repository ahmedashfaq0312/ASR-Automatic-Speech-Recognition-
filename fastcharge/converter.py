import os
import mat73
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import Counter # Counter counts the number of occurrences of each item
from itertools import tee, count

def uniquify(seq, suffs = count(1)):
    """Make all the items unique by adding a suffix (1, 2, etc).

    `seq` is mutable sequence of strings.
    `suffs` is an optional alternative suffix iterable.
    """
    not_unique = [k for k,v in Counter(seq).items() if v>1] # so we have: ['name', 'zip']
    # suffix generator dict - e.g., {'name': <my_gen>, 'zip': <my_gen>}
    suff_gens = dict(zip(not_unique, tee(suffs, len(not_unique))))  
    for idx,s in enumerate(seq):
        try:
            suffix = f"_{str(next(suff_gens[s]))}"
        except KeyError:
            # s was unique
            continue
        else:
            seq[idx] += suffix

class FastChargeConverter():
    def __init__(self, fastcharge_root):
        self.fastcharge_root = fastcharge_root
        self.dataset_files = ["2017-05-12_batchdata_updated_struct_errorcorrect.mat", "2017-06-30_batchdata_updated_struct_errorcorrect.mat", "2018-02-20_batchdata_updated_struct_errorcorrect.mat"]
        

    def load_measurement(self, file):
        """Loads mat file.
        """
        return mat73.loadmat(file)["batch"]
    
    def clean_zero_rows(self, df):
        """Replace rows containing only zero values.
        """
        for index, row in df.iterrows():
            if row.eq(0).any():
                new_row_index = index+1
                if index == 0:
                    new_row_index = index+1
                elif index == len(df)-1:
                    new_row_index = index-1
                tmp_row = df.loc[new_row_index]
                df.loc[index] = tmp_row
                df.loc[index, "cycle"] = index+1
        return df

    def concat_and_save_data(self, data, out_file):
        summary = data["summary"]
        policy = data["policy"]
        cycle_lifes = data["cycle_life"]
        uniquify(policy)
        fastcharge_summary_df = pd.DataFrame([])
        for i in range(len(summary)):
            if policy[i] not in ["3_6C_80PER_3_6C_SLOWCYCLE", "4_8C_80PER_4_8C_SLOWCYCLE"]: # bad measurements
                print(f"Processing {policy[i]}")
                if np.isnan(cycle_lifes[i]):
                    cycle_lifes[i] = -1
                tmp_df = pd.DataFrame(summary[i])
                tmp_df["Experiment"] = policy[i]
                tmp_df["Cycle_Life"] = cycle_lifes[i]
                tmp_df = self.clean_zero_rows(tmp_df)
                fastcharge_summary_df = pd.concat([fastcharge_summary_df, tmp_df], ignore_index=True)
        fastcharge_summary_df.to_csv(out_file)

    def convert(self):
        """Converts all cell data.
        """
        for file in self.dataset_files:
            if file == "2018-02-20_batchdata_updated_struct_errorcorrect.mat":
                out_file = "FastCharge_201802.csv"
            elif file == "2017-06-30_batchdata_updated_struct_errorcorrect.mat":
                out_file = "FastCharge_201706.csv"
            elif file == "2017-05-12_batchdata_updated_struct_errorcorrect.mat":
                out_file = "FastCharge_201705.csv"
            else:
                out_file = "error.csv"
            data_file = f"{self.fastcharge_root}/{file}"
            output_file = f"{self.fastcharge_root}/{out_file}"
            if not os.path.exists(output_file) and os.path.exists(data_file):
                print(f"Converting {file}")
                data = self.load_measurement(data_file)
                self.concat_and_save_data(data, output_file)
            
        
