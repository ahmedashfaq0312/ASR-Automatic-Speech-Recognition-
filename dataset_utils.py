import pandas as pd

def get_positional_information(data_df, cell_column_name):
    """Extracts positional information from data.
    """
    positions = []
    unique_cells = data_df[cell_column_name].unique()
    for cell in unique_cells:
        cell_df = data_df[data_df[cell_column_name] == cell]
        positions.extend([i+1 for i in range(len(cell_df))])
    return positions

def get_eol_information(data_df, is_normalized, rated_capacity):
    eols = []
    cycles_until_eols = []
    unique_cells = data_df["Cell_ID"].unique()
    eol_criterion = 0.7 if is_normalized else rated_capacity*0.7
    for cell in unique_cells:
        caps = data_df[data_df["Cell_ID"] == cell]["Capacity"].astype(float)
        try:
            # calculate cycle where EOL is reached (if EOL not reached, cycle is set to -1)
            eol_idx = next(x for x, val in enumerate(caps) if val <= eol_criterion)
        except StopIteration:
            eol_idx = -1
        eols.extend([eol_idx for _ in range(len(caps))])
        if eol_idx == -1:
            cycles_until_eol = [-1 for i in range(len(caps))]
        else:
            cycles_until_eol = [eol_idx - i for i in range(len(caps))]
        cycles_until_eols.extend(cycles_until_eol)
    data_df["EOL_cycle"] = eols
    data_df["Cycles_to_EOL"] = cycles_until_eols

def filter_rows(data_df, column_name, attribute):
    """Filters rows of specific colums with specific values.
    """
    return_df = pd.DataFrame([0])
    if type(attribute) == str:
        return_df = data_df[data_df[column_name] == attribute]
    elif type(attribute) == list:
        return_df = data_df[data_df[column_name].isin(attribute)]
    return return_df