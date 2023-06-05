import pandas as pd

class CALCEConverter():
    def __init__(self):
        pass

    def convert(self, xlsx_path):
        print(f"Converting {xlsx_path}")
        csv_path = xlsx_path.replace(".xlsx", ".csv")
        read_file = pd.read_excel(xlsx_path, sheet_name=1, dtype=str)
        read_file.to_csv(csv_path, encoding='utf-8', index=False, header=True)