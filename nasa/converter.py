import os
import numpy as np
import pandas as pd
from scipy import io

# helper and converter methods from this notebook: https://www.kaggle.com/code/patrickfleith/nasa-battery-life-prediction-dataset-cleaning
class NASAConverter():
    def __init__(self, nasa_dir="NASA_raw", output_dir="NASA") -> None:
        self.NASA_root = nasa_dir
        self.output_dir = output_dir
        self.filelist = []

    def get_filelist(self):
        """Get file paths in source directory, filter for matlab files and sort.
        """
        self.load_filelist()
        self.filter_matfiles_list()
        self.filelist = sorted(self.filelist)        

    # Helper functions
    def load_filelist(self):
        """Load file paths from source directory.
        """
        for dirname, _, filenames in os.walk(self.NASA_root):
            for filename in filenames:
                self.filelist.append(os.path.join(dirname, filename))

    def filter_matfiles_list(self):
        """Filter filelist for matlab files.
        """
        self.filelist = [filepath for filepath in self.filelist if filepath.endswith('.mat')]
        self.filelist = [filepath for filepath in self.filelist if "BatteryAgingARC_25_26_27_28_P1" not in filepath] # removing duplicates

    def loadmat(self, filepath):
        """Load matlab file.
        """
        return io.loadmat(filepath, simplify_cells=True)

    def process_data_dict(self, data_dict):
        """ Creates two dictionaries:
        - ndict: new dictionary with the test data to build a corresponding dataframe
        - metadata_dict: anything that doesn't fit in ndict ('Capacity' is just a float)
        """
        
        ndict = {}
        metadata_dict = {}
        for k, v in data_dict.items():
            if k not in ['Capacity', 'Re', 'Rct']:
                ndict[k]=v
            elif k == 'Capacity':
                metadata_dict[k]=v
            elif k == 'Re':
                metadata_dict[k]=v
            elif k == 'Rct':
                metadata_dict[k]=v
            else:
                print("No value found")
        
        return ndict, metadata_dict


    def fill_metadata_row(self, test_type, test_start_time, test_temperature, battery_name, test_id, uid, filename, capacity, re, rct):
        """Add row to metadata file.
        """
        tmp_df = pd.DataFrame(data=[test_type, test_start_time, test_temperature, battery_name, test_id, uid, filename, capacity, re, rct])
        tmp_df = tmp_df.transpose()
        tmp_df.columns = self.metadata.columns
        self.metadata = pd.concat((self.metadata, tmp_df), axis=0)

    def extract_more_metadata(self, metadata_dict):
        """Extract capacity and resistances from metadata.
        """
        if 'Capacity' in metadata_dict.keys():
            capacity = metadata_dict['Capacity']
        else:
            capacity = np.nan
            
        if 'Re' in metadata_dict.keys():
            re = metadata_dict['Re']
        else:
            re = np.nan
            
        if 'Rct' in metadata_dict.keys():
            rct = metadata_dict['Rct']
        else:
            rct = np.nan
        
        return capacity, re, rct

    def save_metadata(self):
        """Save metadata as csv file.
        """
        if not os.path.exists(f'{self.output_dir}/metadata.csv'):
            self.metadata.to_csv(f'{self.output_dir}/metadata.csv', index=False)

    def convert(self):
        """Convert data and extract metadata.
        """
        self.get_filelist()
        print("Converting NASA dataset")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.metadata = pd.DataFrame(data=None, columns=['type', 'start_time', 'ambient_temperature', 'battery_id', 'test_id', 'uid', 'filename', 'Capacity', 'Re', 'Rct'])
        self.battery_list = [item.split('/')[-1].split('.')[0] for item in self.filelist]
    
        uid = 0
        for battery_name, mat_filepath in zip(self.battery_list, self.filelist):
            mat_data = io.loadmat(mat_filepath, simplify_cells=True)
            print(mat_filepath[-10:],"-->", battery_name)
            test_list = mat_data[battery_name]['cycle']
            
            for test_id in range(len(test_list)):
                
                uid += 1
                filename = str(uid).zfill(5)+'.csv'
                data_dir = f"{self.output_dir}/data"
                if not os.path.exists(data_dir):
                    os.makedirs(data_dir)
                filepath = f'./{data_dir}/{filename}'
                
                if not os.path.exists(filepath):
                    # Extract the specific test data and save it as CSV! 
                    ndict, metadata_dict = self.process_data_dict(test_list[test_id]['data'])
                    test_df = pd.DataFrame.from_dict(ndict, orient='index')
                    test_df = test_df.transpose()

                    test_df.to_csv(filepath, index=False)
                            
                    # Add test information to the metadata
                    test_type = test_list[test_id]['type']
                    test_start_time = test_list[test_id]['time']
                    test_temperature = test_list[test_id]['ambient_temperature']
                    
                    capacity, re, rct = self.extract_more_metadata(metadata_dict)
                    self.fill_metadata_row(test_type, test_start_time, test_temperature, battery_name, test_id, uid, filename, capacity, re, rct)
        self.save_metadata()
