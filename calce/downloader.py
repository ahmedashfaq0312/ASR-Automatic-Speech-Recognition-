import os
import re
import requests
import zipfile

class CALCEDownloader():
    """Class for downloading CALCE battery dataset.
    """
    def __init__(self, battery_list, output_path="CALCE") -> None:
        self.battery_list = battery_list
        self.download_base_url = "https://web.calce.umd.edu/batteries/data/" # CS2_35.zip
        self.output_path = output_path
        self.raw_output_path = self.output_path + "_raw"
        
    def download(self):
        """Download dataset.
        """
        # print(f"Downloading CALCE dataset from {self.download_base_url}")
        for battery in self.battery_list:
            output_file = f"{battery}.zip"
            output_name = f"{self.raw_output_path}/{output_file}"
            if not os.path.exists(self.raw_output_path):
                os.makedirs(self.raw_output_path)
            if not os.path.exists(output_name):
                print(f"Downloading {output_file}")
                with requests.get(self.download_base_url + output_file, stream=True) as r:
                    r.raise_for_status()
                    with open(output_name, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=65536):  
                            f.write(chunk)
    
    def extract(self):
        """ Extract a zip file including any nested zip files
            Delete the zip file(s) after extraction
        """
        for battery in self.battery_list:
            zipped_file = f"{self.raw_output_path}/{battery}.zip"
            if os.path.exists(zipped_file):
                if not os.path.exists(f"{self.output_path}/{battery}"):
                    print(f"Extracting {zipped_file}")
                    with zipfile.ZipFile(zipped_file, 'r') as zfile:
                        zfile.extractall(path=f"{self.output_path}/")
            if battery == "CS2_21" and not os.path.exists(f"{self.output_path}/{battery}/CS2_21_7_09_10.txt"):
                os.rename(f"{self.output_path}/{battery}/CS2_21_7_9b_10.txt", f"{self.output_path}/{battery}/CS2_21_7_09_10.txt")

    def download_and_extract(self):
        """Wrapper for download and extraction.
        """
        self.download()
        self.extract()