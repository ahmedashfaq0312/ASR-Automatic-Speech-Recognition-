import os
import numpy as np
import pandas as pd
import glob
import openpyxl
import csv


class CALCEDataset():
    def __init__(self, calce_root, battery_list, file_type=".csv"):
        self.calce_root = calce_root
        self.battery_list = battery_list
        self.file_type = file_type
        self.data_dict = {}
        self.capacities = {}
        self.sohs = {}
        self.resistances = {}
        self.timestamps = {}
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

    def convert(self, xlsx_path):
        print(f"Converting {xlsx_path}")
        csv_path = xlsx_path.replace(".xlsx", ".csv")
        read_file = pd.read_excel(xlsx_path, sheet_name=1, dtype=str)
        newWorkbook = openpyxl.load_workbook(xlsx_path)
        sheet = newWorkbook.worksheets[1]
        OutputCsvFile = csv.writer(open(csv_path, 'w'), delimiter=",")
        for eachrow in sheet.rows:
            OutputCsvFile.writerow([cell.value for cell in eachrow])
        # read_file.to_csv(csv_path, encoding='utf-8', index=False, header=True)

    def load(self):
        for name in self.battery_list:
            print('Load CALCE Dataset ' + name + ' ...')
            path = glob.glob(self.calce_root + name + '/*.xlsx')
            dates = []
            for p in path:
                if self.file_type == ".csv":
                    csv_path = p.replace(".xlsx", ".csv")
                    if not os.path.exists(csv_path):
                        self.convert(p)
                    p = csv_path
                    df = pd.read_csv(p)
                elif self.file_type == ".xlsx":
                    df = pd.read_excel(p, sheet_name=1)
                print('Load ' + str(p) + ' ...')
                dates.append(df['Date_Time'][0])
            if self.file_type == ".csv":
                path = glob.glob(self.calce_root + name + '/*.csv')
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
