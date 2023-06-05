import os
import numpy as np
import pandas as pd
import glob
from calce.converter import CALCEConverter
from calce.downloader import CALCEDownloader
class CALCEDataset():
    def __init__(self, batteries, calce_root="CALCE", file_type=".csv"):
        self.calce_root = calce_root
        self.batteries = batteries
        self.file_type = file_type
        self.data_dict = {}
        self.capacities = {}
        self.sohs = {}
        self.resistances = {}
        self.timestamps = {}
        self.ccct = {}
        self.cvct = {}
        self.raw_output_path = self.calce_root + "_raw"
        self.downloader = CALCEDownloader(batteries=self.batteries, output_path=self.raw_output_path)
        self.converter = CALCEConverter()
        self.load()

    def drop_outlier(self, array,count,bins):
        index = []
        range_ = np.arange(1,count,bins)
        for i in range_[:-1]:
            array_lim = array[i:i+bins]
            sigma = np.std(array_lim)
            mean = np.mean(array_lim)
            th_max,th_min = mean + sigma*2, mean - sigma*2
            idx = np.where((array_lim < th_max) & (array_lim > th_min))
            idx = idx[0] + i
            index.extend(list(idx))
        return np.array(index)

    def load(self):
        self.downloader.download_and_extract()
        for name in self.batteries:
            print('Load CALCE Dataset ' + name + ' ...')
            path = glob.glob(self.calce_root + name + '/*.xlsx')
            dates = []
            paths = []
            for p in path:
                if self.file_type == ".csv":
                    csv_path = p.replace(".xlsx", ".csv")
                    if not os.path.exists(csv_path):
                        self.converter.convert(p)
                    p = csv_path
                    df = pd.read_csv(p)
                    paths.append(csv_path)
                    date = pd.Timestamp(df['Date_Time'][0])
                elif self.file_type == ".xlsx":
                    df = pd.read_excel(p, sheet_name=1)
                    date = df['Date_Time'][0]
                print('Load ' + str(p) + ' ...')
                dates.append(date)
            if self.file_type == ".csv":
                path = paths
            elif self.file_type == ".xlsx":
                path = glob.glob(self.calce_root + name + '/*.xlsx')
            idx = np.argsort(dates)
            path_sorted = (np.array(path)[idx]).tolist()

            count = 0
            time_offset = 0
            discharge_capacities = []
            health_indicator = []
            internal_resistance = []
            timestamps = []
            CCCT = []
            CVCT = []
            for p in path_sorted:
                if self.file_type == ".csv":
                    df = pd.read_csv(p)
                elif self.file_type == ".xlsx":
                    df = pd.read_excel(p, sheet_name=1)
                print('Load ' + str(p) + ' ...')
                cycles = list(set(df['Cycle_Index']))
                for c in cycles:
                    df_lim = df[df['Cycle_Index'] == c]
                    #Charging
                    df_c = df_lim[(df_lim['Step_Index'] == 2)|(df_lim['Step_Index'] == 4)]
                    c_v = df_c['Voltage(V)']
                    c_c = df_c['Current(A)']
                    c_t = df_c['Test_Time(s)']
                    #CC or CV
                    df_cc = df_lim[df_lim['Step_Index'] == 2]
                    df_cv = df_lim[df_lim['Step_Index'] == 4]
                    CCCT.append(np.max(df_cc['Test_Time(s)'])-np.min(df_cc['Test_Time(s)']))
                    CVCT.append(np.max(df_cv['Test_Time(s)'])-np.min(df_cv['Test_Time(s)']))

                    #Discharging
                    df_d = df_lim[df_lim['Step_Index'] == 7]
                    d_v = df_d['Voltage(V)']
                    d_c = df_d['Current(A)']
                    d_t = df_d['Test_Time(s)']
                    d_im = df_d['Internal_Resistance(Ohm)']

                    if(len(list(d_c)) != 0):
                        time_diff = np.diff(list(d_t))
                        timestamps.append((list(d_t)[-1]/3600) + time_offset)
                        d_c = np.array(list(d_c))[1:]
                        discharge_capacity = time_diff*d_c/3600 # Q = A*h
                        discharge_capacity = [np.sum(discharge_capacity[:n]) for n in range(discharge_capacity.shape[0])]
                        discharge_capacities.append(-1*discharge_capacity[-1])

                        dec = np.abs(np.array(d_v) - 3.8)[1:]
                        start = np.array(discharge_capacity)[np.argmin(dec)]
                        dec = np.abs(np.array(d_v) - 3.4)[1:]
                        end = np.array(discharge_capacity)[np.argmin(dec)]
                        health_indicator.append(-1 * (end - start))

                        internal_resistance.append(np.mean(np.array(d_im)))
                        count += 1
                time_offset = timestamps[-1]
            discharge_capacities = np.array(discharge_capacities)
            health_indicator = np.array(health_indicator)
            internal_resistance = np.array(internal_resistance)
            timestamps = np.array(timestamps)
            CCCT = np.array(CCCT)
            CVCT = np.array(CVCT)
            
            idx = self.drop_outlier(discharge_capacities, count, 40)
            df_result = pd.DataFrame({'cycle':np.linspace(1,idx.shape[0],idx.shape[0]),
                                    'capacity':discharge_capacities[idx],
                                    'SoH':health_indicator[idx],
                                    'resistance':internal_resistance[idx],
                                    'timestamps':timestamps[idx],
                                    'CCCT':CCCT[idx],
                                    'CVCT':CVCT[idx]})
            self.data_dict[name] = df_result
            self.capacities[name] = discharge_capacities[idx]
            self.sohs[name] = health_indicator[idx]
            self.resistances[name] = internal_resistance[idx]
            self.timestamps[name] = timestamps[idx]
            self.ccct[name] = CCCT[idx]
            self.cvct[name] = CVCT[idx]
