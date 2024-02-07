import os
import re
import requests
import zipfile

class NASADownoader():
    """Class for downloading NASA battery dataset.
    """
    def __init__(self, output_path="NASA_raw") -> None:
        self.download_url = "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip"
        self.output_name = "NASA.zip"
        self.output_path = output_path
        self.download_file_path = f"{self.output_path}/{self.output_name}"
        self.unzip_folder = f"{self.output_path}/5. Battery Data Set"
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        
    def download(self):
        """Download dataset.
        """
        if not os.path.exists(self.download_file_path):
            print(f"Downloading NASA dataset from {self.download_url}")
            with requests.get(self.download_url, stream=True) as r:
                r.raise_for_status()
                with open(self.download_file_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=65536):  
                        f.write(chunk)
        
        if os.path.exists(self.output_name):
            os.rename(self.output_name, self.download_file_path)
    
    def extract(self, zipped_file="NASA.zip", to_folder="."):
        """ Extract a zip file including any nested zip files.
            Delete the zip file(s) after extraction.
        """
        if os.path.exists(self.download_file_path):
            if not os.path.exists(self.unzip_folder):
                print(f"Extracting {self.download_file_path}")
                with zipfile.ZipFile(self.download_file_path, 'r') as zfile:
                    zfile.extractall(path=self.output_path)
            
            for root, _, files in os.walk(self.unzip_folder):
                for file in files:
                    name, ext = os.path.splitext(file)
                    if os.path.exists(f"{self.unzip_folder}/{file}") and ext == ".zip":
                        with zipfile.ZipFile(f"{self.unzip_folder}/{file}", 'r') as zfile:
                            zfile.extractall(path=self.unzip_folder) 
                        os.remove(f"{self.unzip_folder}/{file}")
                    elif ext == ".txt":
                        os.remove(f"{self.unzip_folder}/{file}")

    def download_and_extract(self):
        """Wrapper for download and extraction.
        """
        self.download()
        self.extract()
