import os
import re
import requests
import zipfile

class NASADownoader():
    def __init__(self, output_path="NASA_raw") -> None:
        self.download_url = "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip"
        self.output_name = "NASA.zip"
        self.output_path = output_path
        self.unzip_folder = "5. Battery Data Set"
        
    def download(self):
        if not os.path.exists(self.output_name):
            print(f"Downloading NASA dataset from {self.download_url}")
            with requests.get(self.download_url, stream=True) as r:
                r.raise_for_status()
                with open(self.output_name, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=65536):  
                        f.write(chunk)
    
    def extract(self, zipped_file="NASA.zip", to_folder="."):
        """ Extract a zip file including any nested zip files
            Delete the zip file(s) after extraction
        """
        if os.path.exists(zipped_file):
            print(f"Extracting {zipped_file}")
            with zipfile.ZipFile(zipped_file, 'r') as zfile:
                zfile.extractall(path=to_folder)
            os.remove(zipped_file)
            for root, _, files in os.walk(to_folder):
                for filename in files:
                    if re.search(r'\.zip$', filename):
                        filespec = os.path.join(root, filename)
                        self.extract(filespec, root)
        if os.path.exists(self.unzip_folder) and not os.path.exists(self.output_path):
            os.rename(self.unzip_folder, self.output_path)
