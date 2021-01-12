import pandas as pd
import statsmodels.api as sm

from utils import load_data


def feature_selection(df: pd.DataFrame, threshold: float = 0.6, verbose: bool = False) -> pd.DataFrame:
    """Iteratively drop the column most explained by the other columns, until threshold is reached."""
    while len(df.columns) > 1:
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

    print(f"Remaining columns: {list(df.columns)}")
    print("R2 per column:\n\t", rsquare_per_col)

    return df


DATA_DIRECTORY = r"C:\Users\Daniel\Documents\Studium\7. Semester (WS 2020.21)\Seminar Data Mining in der Produktion\Gruppenarbeit\Data"
COMPONENTS = [
    ['A_1', 'A_2', 'A_3', 'A_4', 'A_5'],
    ['B_1', 'B_2', 'B_3', 'B_4', 'B_5'],
    ['C_1', 'C_2', 'C_3', 'C_4', 'C_5'],
    ['L_1', 'L_2', 'L_3', 'L_4', 'L_5', 'L_6', 'L_7', 'L_8', 'L_9', 'L_10']
]

files = load_data(DATA_DIRECTORY)
file = pd.concat(files)
THRESHOLD = 0.6

for group in COMPONENTS:
    feature_selection(file[group], threshold=THRESHOLD)

a = 1
