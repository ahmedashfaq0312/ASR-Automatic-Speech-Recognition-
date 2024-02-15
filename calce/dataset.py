import os
import numpy as np
import pandas as pd
import glob
import scipy
try:
    from calce.converter import CALCEConverter
    from calce.downloader import CALCEDownloader
except ModuleNotFoundError:
    from rul_estimation_datasets.calce.converter import CALCEConverter
    from rul_estimation_datasets.calce.downloader import CALCEDownloader
from rul_estimation_datasets.dataset_utils import filter_rows, get_eol_information


class CALCEDataset():
    """Class for preprocessing and loading CALCE battery dataset.
    """
    def __init__(self, dataset_config):
        self.dataset_config = dataset_config
        self.calce_root = self.dataset_config.dataset_root_dir
        self.train_cells = self.dataset_config.train_cells
        self.test_cells = self.dataset_config.test_cells
        self.dataset_cells = self.train_cells + self.test_cells
        self.file_type = self.dataset_config.file_type
        self.normalize = self.dataset_config.normalize_data
        self.clean_data = self.dataset_config.clean_data
        self.rated_capacity = self.dataset_config.rated_capacity
        self.smooth_data = self.dataset_config.smooth_data
        self.smoothing_kernel_width = self.dataset_config.smoothing_kernel_width

        self.downloader = CALCEDownloader(battery_list=self.dataset_cells, output_path=self.calce_root)
        self.converter = CALCEConverter()
        self.load()
        self.get_dataset_length()

    def drop_outlier(self, array,count,bins):
        """Filters and drops outliers in data.
        """
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

    def has_column_same_values(self, data_column):
        """Checks if all values in column are the same.
        """
        data_column_numpy = data_column.to_numpy()
        return (data_column_numpy[0] == data_column_numpy).all()

    def remove_column_with_same_value(self, dataframe):
        """Filters dataframe and drops columns with same values.
        """
        for column in dataframe:
            df_col = dataframe[column]
            if self.has_column_same_values(df_col):
                dataframe = dataframe.drop(column, axis=1)
        return dataframe

    def clean_peaks(self, data, thresh=0.1):
        """Cleans eajs frin data,
        """
        peaks = []
        for i in range(len(data)-1):
            diff = abs(data[i] - data[i+1])
            if diff > thresh:
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

    def get_positional_information(self, data_df):
        """Extracts positional information from data.
        """
        return data_df["cycle"].to_list()

    
    def get_temporal_information(self, data_df):
        """Extracts temporal information from data.
        """
        return data_df["timestamps"].to_list()

    def get_eol_information(self):
        get_eol_information(self.train_df, self.normalize, self.rated_capacity)
        get_eol_information(self.test_df, self.normalize, self.rated_capacity)

    def get_dataset_length(self):
        self.train_dataset_length = len(self.train_df["Capacity"])
        self.trest_dataset_length = len(self.test_df["Capacity"])

    def load_cell_data(self):
        overview_path = self.calce_root + "/overview.csv"
        if not os.path.exists(overview_path):
            overview_df = pd.DataFrame([])
            for name in self.train_cells:
                print('Load CALCE Dataset ' + name + ' ...')
                if name in ["CS2_8", "CS2_21"]:
                    path = glob.glob(self.calce_root + "/" + name + '/*.txt')
                else:
                    path = glob.glob(self.calce_root + "/" + name + '/*.xlsx')
                # path = sorted(path)
                dates = []
                paths = []
                cell_dfs = []
                for p in path:
                    if name in ["CS2_8", "CS2_21"]:
                        df = pd.read_csv(p, sep="\t", dtype=float)
                        date = df['Time'][0]
                    else:
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
                    cell_id_list = [name for _ in range(len(df))]
                    df.insert(1, "Cell_ID", cell_id_list, True)
                    cell_dfs.append(df)
                    print('Load ' + str(p) + ' ...')
                    dates.append(date)
                if name in ["CS2_8", "CS2_21"]:
                    path = glob.glob(self.calce_root + "/" + name + '/*.txt')
                    path_sorted = sorted(path)
                    self.load_txt(df, name, path_sorted) 
                else:
                    if self.file_type == ".csv":
                        path = paths
                    elif self.file_type == ".xlsx":
                        path = glob.glob(self.calce_root + name + '/*.xlsx')
                    idx = np.argsort(dates)
                    path_sorted = (np.array(path)[idx]).tolist()
                    df_result = self.load_excel_csv(df, name, path_sorted)
                    cell_id_list = [name for _ in range(len(df_result))]
                    df_result.insert(1, "cell_id", cell_id_list, True)
                    cell_dfs = pd.concat(cell_dfs)
                    overview_df = pd.concat([overview_df, df_result], ignore_index=True)
            overview_df.to_csv(overview_path)
        else:
            overview_df = pd.read_csv(overview_path, index_col=0)
        self.data_df = overview_df

    def normalize_capacities(self):
        """Normalizes capacities.
        """
        train_capacities = self.raw_train_df["capacity"].astype(float)
        normalized__train_capacities = [i/self.rated_capacity for i in train_capacities]
        self.train_df["Cell_ID"] = self.raw_train_df["cell_id"].to_list()
        self.train_df["Capacity"] = normalized__train_capacities
        test_capacities = self.raw_test_df["capacity"].astype(float)
        normalized_test_capacities = [i/self.rated_capacity for i in test_capacities]
        self.test_df["Cell_ID"] = self.raw_test_df["cell_id"].to_list()
        self.test_df["Capacity"] = normalized_test_capacities

    def preprocess(self):
        self.train_df = pd.DataFrame([])
        self.test_df = pd.DataFrame([])
        self.raw_train_df = filter_rows(self.data_df, "cell_id", self.train_cells)
        self.raw_test_df = filter_rows(self.data_df, "cell_id", self.test_cells)
        self.train_df["Cycle"] = self.get_positional_information(self.raw_train_df)
        self.test_df["Cycle"] = self.get_positional_information(self.raw_test_df)
        self.train_df["Time"] = self.get_temporal_information(self.raw_train_df)
        self.test_df["Time"] = self.get_temporal_information(self.raw_test_df)

        if self.normalize:
            self.normalize_capacities()
        # if self.smooth_data:
        #     self.smooth_capacities()
        get_eol_information(self.train_df, self.normalize, self.rated_capacity)
        get_eol_information(self.test_df, self.normalize, self.rated_capacity)

    def load(self):
        """Loads CALCE dataset.
        """
        self.downloader.download_and_extract()
        self.load_cell_data()

        self.preprocess()


        
    def load_txt(self, df, name, path_sorted):
        """Wrapper for loading txt data.
        """
        time_offset = 0
        discharge_capacities = []
        measurement_times = []
        for p in path_sorted:
            df = pd.read_csv(p, sep="\t", dtype=float)
            df = self.remove_column_with_same_value(df)
            if "Capacity" in df:
                raw_capacities = df["Capacity"]
                peaks, _ = scipy.signal.find_peaks(raw_capacities)
                discharge_capacities.extend(raw_capacities[peaks].tolist())
                raw_time = df["Time"][peaks].tolist()
                raw_time_hours = [time/3600 for time in raw_time]
                measurement_times.extend([time + time_offset for time in raw_time_hours])
                time_offset += raw_time_hours[-1]
        capacities = [c / 100 for c in discharge_capacities]
        if self.clean_data:
            capacities = self.clean_peaks(capacities)
        self.capacities[name] = capacities
        self.measurement_times[name] = measurement_times


    def load_excel_csv(self, df, name, path_sorted):
        """Wrapper for loading excel or csv data.
        """
        count = 0
        time_offset = 0
        discharge_capacities = []
        health_indicator = []
        internal_resistance = []
        measurement_times = []
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
                    measurement_times.append((list(d_t)[-1]/3600) + time_offset)
                    d_c = np.array(list(d_c))[1:]
                    try:
                        discharge_capacity = time_diff*d_c/3600 # Q = A*h
                        discharge_capacity = [np.sum(discharge_capacity[:n]) for n in range(discharge_capacity.shape[0])]
                        discharge_capacities.append(-1*discharge_capacity[-1])
                    except:
                        continue
                    dec = np.abs(np.array(d_v) - 3.8)[1:]
                    start = np.array(discharge_capacity)[np.argmin(dec)]
                    dec = np.abs(np.array(d_v) - 3.4)[1:]
                    end = np.array(discharge_capacity)[np.argmin(dec)]
                    health_indicator.append(-1 * (end - start))

                    internal_resistance.append(np.mean(np.array(d_im)))
                    count += 1
            time_offset = measurement_times[-1]
        discharge_capacities = np.array(discharge_capacities)
        health_indicator = np.array(health_indicator)
        internal_resistance = np.array(internal_resistance)
        measurement_times = np.array(measurement_times)
        CCCT = np.array(CCCT)
        CVCT = np.array(CVCT)

        idx = self.drop_outlier(discharge_capacities, count, 40)
        capacities = discharge_capacities[idx]
        health_indicator = health_indicator[idx]
        internal_resistance = internal_resistance[idx]
        measurement_times = measurement_times[idx]
        CCCT = CCCT[idx]
        CVCT = CVCT[idx]
        if self.clean_data:
            capacities = self.clean_peaks(capacities)
            health_indicator = self.clean_peaks(health_indicator)
            internal_resistance = self.clean_peaks(internal_resistance)
            CCCT = self.clean_peaks(CCCT)
            CVCT = self.clean_peaks(CVCT)

        df_result = pd.DataFrame({'cycle':np.linspace(1,idx.shape[0],idx.shape[0]),
                                'capacity':capacities,
                                'SoH':health_indicator,
                                'resistance':internal_resistance,
                                'timestamps':measurement_times,
                                'CCCT':CCCT,
                                'CVCT':CVCT})
        self.data_dict[name] = df_result
        self.capacities[name] = capacities
        self.sohs[name] = health_indicator
        self.resistances[name] = internal_resistance
        self.measurement_times[name] = measurement_times
        self.ccct[name] = CCCT
        self.cvct[name] = CVCT
        return df_result
