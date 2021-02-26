from typing import List

import pandas as pd
import statsmodels.api as sm


def feature_selection(df: pd.DataFrame, no_columns: int = 1, threshold: float = 0.0, verbose: bool = False) -> List[str]:
    """Iteratively drop the column most explained by the other columns, until threshold is reached.

    :param df: DataFrame to select features from.
    :param no_columns: Number of columns to select.
    :param threshold: Set a threshold for a minimum R2 to be reached before columns are discarded.
    :param verbose: Print internal values in each iteration.
    """
    while len(df.columns) > max(2, no_columns):
        rsquare_per_col = {}

        # Regression on each column individually
        for col in df.columns:
            x = sm.add_constant(df.drop(col, axis=1))
            y = df[col]

            rsquare_per_col[col] = sm.OLS(y, x).fit().rsquared

        max_rsquared = max(rsquare_per_col.values())

        if not (max_rsquared >= threshold):
            break
        else:
            col_with_max_r2 = max(rsquare_per_col, key=rsquare_per_col.get)
            df = df.drop(col_with_max_r2, axis=1)

        if verbose:
            print(f"Dropped col {col_with_max_r2} with R2 = {max_rsquared}")
            print("R2 per column:\n\t", rsquare_per_col)

    # if only one column should be found, select first column (the last two columns explain each other the same)
    if no_columns == 1:
        df = df.iloc[:, :1]

    if verbose:
        print(f"Remaining columns: {list(df.columns)}")

    return list(df.columns)


# Apply to data
DATA_DIRECTORY = r"..."

# Load data files
c11 = pd.read_csv(DATA_DIRECTORY + "/C11.csv")
c13_1 = pd.read_csv(DATA_DIRECTORY + "/C13-1.csv")
c13_2 = pd.read_csv(DATA_DIRECTORY + "/C13-2.csv")
c14 = pd.read_csv(DATA_DIRECTORY + "/C14.csv")
c15 = pd.read_csv(DATA_DIRECTORY + "/C15.csv")
c16 = pd.read_csv(DATA_DIRECTORY + "/C16.csv")
c7_1 = pd.read_csv(DATA_DIRECTORY + "/C7-1.csv")
c7_2 = pd.read_csv(DATA_DIRECTORY + "/C7-2.csv")
c8 = pd.read_csv(DATA_DIRECTORY + "/C8.csv")
c9 = pd.read_csv(DATA_DIRECTORY + "/C9.csv")

# Merge C13-1 and C13-2 as well as C7-1 and C7-2
c13 = pd.concat([c13_1, c13_2])
c7 = pd.concat([c7_1, c7_2])

# Construct dictionary of files
files = {
    "c7": c7,
    "c8": c8,
    "c9": c9,
    "c11": c11,
    "c13": c13,
    "c14": c14,
    "c15": c15,
    "c16": c16
}

# Drop timestamp column
for key, value in files.items():
    files[key] = value.drop(columns="Timestamp")

# Remove NaN
for key, value in files.items():
    files[key] = value.dropna()

ALL_COMPONENTS = ['A_1', 'A_2', 'A_3', 'A_4', 'A_5',
                  'B_1', 'B_2', 'B_3', 'B_4', 'B_5',
                  'C_1', 'C_2', 'C_3', 'C_4', 'C_5',
                  'L_1', 'L_2', 'L_3', 'L_4', 'L_5',
                  'L_6', 'L_7', 'L_8', 'L_9', 'L_10'
                  ]

COMPONENTS = [
    ['A_1', 'A_2', 'A_3', 'A_4', 'A_5'],
    ['B_1', 'B_2', 'B_3', 'B_4', 'B_5'],
    ['C_1', 'C_2', 'C_3', 'C_4', 'C_5'],
    ['L_1', 'L_2'],
    ['L_3', 'L_6'],
    ['L_4', 'L_5'],
    ['L_7', 'L_8'],
    ['L_9', 'L_10']
]

cols_df = pd.DataFrame(columns=ALL_COMPONENTS)
for name, data in files.items():
    cols_to_include = []
    for group in COMPONENTS:
        cols_to_include.extend(feature_selection(data[group], no_columns=1))

    print(f"Experiment {name}: ", cols_to_include)

    update = dict()
    for col in cols_df.columns:
        update[col] = 1 if (col in cols_to_include) else 0

    cols_df = cols_df.append(update, ignore_index=True)

cols_df = cols_df.T
cols_df["counts"] = cols_df.sum(axis="columns")
print(cols_df)
