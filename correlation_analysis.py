import pandas as pd
import numpy as np

from utils import load_data


def get_unrolled_list_of_correlations(df):
    """Creates dataframe with columns: ['col1', 'col2', 'correlation between the two columns'] """
    corr = df.corr().abs()
    upper = corr.to_numpy().flatten()
    unrolled = pd.DataFrame(upper, columns=["correlation"])

    no_cols = len(df.columns)

    # Create indices
    idx_1 = []
    for col in df.columns:
        idx_1.extend([col] * no_cols)

    idx_2 = []
    for col in df.columns:
        idx_2.extend(df.columns)

    unrolled["col1"] = idx_1
    unrolled["col2"] = idx_2

    # Post-processing
    unrolled = unrolled.dropna()
    unrolled = unrolled[["col1", "col2", "correlation"]]  # reorder columns

    return unrolled


def drop_correlated_cols(df: pd.DataFrame, no_cols: int = 1, threshold=0.8) -> pd.DataFrame:
    dropped_cols_ordered = []
    internal_df = df

    while len(internal_df.columns) > 1:
        list_of_corrs = get_unrolled_list_of_correlations(internal_df)

        ''' Most correlated cols (threshold)
        highly_correlated = list_of_corrs[list_of_corrs["correlation"] > threshold]
        res = highly_correlated.groupby("col1").count().sort_values("col2", ascending=False)
        '''
        res = list_of_corrs.groupby("col1").mean("correlation").sort_values("correlation", ascending=False)
        col_to_drop = res.index[0]
        internal_df = internal_df.drop(col_to_drop, axis=1)

        dropped_cols_ordered.append(col_to_drop)

    print(dropped_cols_ordered)

    return df[dropped_cols_ordered[:no_cols]]


def drop_correlated_cols_by_component(df: pd.DataFrame, column_groups: list) -> pd.DataFrame:
    dropped_order = []

    for group in column_groups:
        subset = df[group]
        dropped_order_in_group = []
        while (len(subset.columns) > 1) and ():
            dropped_order_in_group.append()


DATA_DIRECTORY = r"C:\Users\Daniel\Documents\Studium\7. Semester (WS 2020.21)\Seminar Data Mining in der Produktion\Gruppenarbeit\Data"
COMPONENTS = [
    ['A_1', 'A_2', 'A_3', 'A_4', 'A_5'],
    ['B_1', 'B_2', 'B_3', 'B_4', 'B_5'],
    ['C_1', 'C_2', 'C_3', 'C_4', 'C_5'],
    ['L_1', 'L_2', 'L_3', 'L_4', 'L_5', 'L_6', 'L_7', 'L_8', 'L_9', 'L_10']
]
[    ['L_1', 'L_2'],
    ['L_3', 'L_6'],
    ['L_4', 'L_5'],
    ['L_7', 'L_8'],
    ['L_9', 'L_10']
]

files = load_data(DATA_DIRECTORY)

for component in COMPONENTS:
    print("\nNew component__________________")
    for file in files:
        temp = file[component]
        drop_correlated_cols(temp)

a = 1
