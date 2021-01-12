import pandas as pd
import statsmodels.api as sm


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
