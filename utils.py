import os
from typing import List, Tuple

import pandas as pd
from sklearn.decomposition import PCA
import sklearn.preprocessing
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import abline_plot


def load_data(directory: str):
    _filepaths = [directory + "/" + filename for filename in os.listdir(directory)]

    files = [pd.read_csv(path) for path in _filepaths]

    # Merge C13-1 and C13-2 as well as C7-1 and C7-2
    c13 = pd.concat([files[1], files[2]])
    c7 = pd.concat([files[6], files[7]])

    files[1] = c13
    files[6] = c7

    files.pop(2)
    files.pop(7)

    # Drop Timestamp column
    files = [df.drop("Timestamp", axis=1) for df in files]


    # Handle NaN
    files = [df.dropna() for df in files]

    return files


def normalize_files(files: List[pd.DataFrame], normalization_norm: str):
    column_names = files[0].columns
    files = [sklearn.preprocessing.normalize(df, norm=normalization_norm) for df in files]
    files = [pd.DataFrame(array, columns=column_names) for array in files]

    return files


def pca_componentwise(files: List[pd.DataFrame], components: list) -> List[pd.DataFrame]:
    processed_files = []

    for file in files:
        new_columns = []

        for component in components:
            component_subset = file[component]
            pca = PCA(n_components=1)

            new_col = pca.fit_transform(component_subset)
            component_column = pd.DataFrame(new_col)
            new_columns.append(component_column)

        result = pd.concat(new_columns, axis=1)
        processed_files.append(result)

    return processed_files


def plot_all_experiments(datasets: list, experiment_names: list, test_train_split: float = None,
                         savefig: str = "C:/Users/Daniel/Desktop/plot.png", ols_line: bool = False) -> None:
    fig, ax = plt.subplots(8, 1)
    fig.set_size_inches(6, 24)

    for i, dataset in enumerate(datasets):
            axis = ax[i]
            axis.scatter(x=dataset.index, y=dataset[0], s=1)
            axis.set_title(f"Experiment No. {experiment_names[i]}")

            if test_train_split is not None:    # plot vertical line
                test_train_split_index = int(len(dataset) * test_train_split)
                axis.axvline(x=test_train_split_index, color="grey", linestyle="--", linewidth=1)

            if ols_line:
                x = sm.add_constant(dataset.index)
                y = dataset[0]
                abline_plot(model_results=sm.OLS(y, x).fit(), ax=axis, color="black", linewidth=1)

    plt.tight_layout()

    if savefig:
        plt.savefig(savefig)
    plt.show()


def test_train_split(files: List[pd.DataFrame], split: float) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    # Split train and test data -> first x% taken (not randomly)
    def split_df_not_randomly(df, split: float):
        split_index = int(len(df) * split)
        return df[:split_index], df[split_index:]

    train_files, test_files = zip(*[split_df_not_randomly(df, split) for df in files])

    return train_files, test_files
