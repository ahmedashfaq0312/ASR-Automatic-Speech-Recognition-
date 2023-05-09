import os
import requests
from .converter import NASAConverter

class NASALoader():
    def __init__(self) -> None:
        self.download_url = "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip"
        self.output_name = "NASA.zip"
        self.output_path = "NASA"
        
    def download(self):
        if not os.path.exists(self.output_name):
            print(f"Downloading NASA dataset from {self.download_url}")
            with requests.get(self.download_url, stream=True) as r:
                r.raise_for_status()
                with open(self.output_name, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=65536):  
                        f.write(chunk)
    
    def convert(self):
        print("Converting NASA dataset")
        nasa_converter = NASAConverter()
        nasa_converter.convert()
