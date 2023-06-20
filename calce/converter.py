import pandas as pd

class CALCEConverter():
    """Class for conversion of CALCE battery dataset.
    """
    def convert(self, xlsx_path):
        """Convert excel file to csv.
        """
        print(f"Converting {xlsx_path}")
        csv_path = xlsx_path.replace(".xlsx", ".csv")
        read_file = pd.read_excel(xlsx_path, sheet_name=1, dtype=str)
        read_file.to_csv(csv_path, encoding='utf-8', index=False, header=True)