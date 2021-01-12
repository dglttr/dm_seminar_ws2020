#%% md
# Imports

#%%
import os

import pandas as pd
import numpy as np
from minisom import MiniSom as SOM
from sklearn import preprocessing
import matplotlib.pyplot as plt

#%% md
# Constants
#%%
DATA_DIRECTORY = r"C:\Users\Daniel\Documents\Studium\7. Semester (WS 2020.21)\Seminar Data Mining in der Produktion\Gruppenarbeit\Data"
NORMALIZATION_NORM = "l2"
TEST_TRAIN_SPLIT = 0.2
RANDOM_SEED = 1     # for reproducibility

# SOM
SOM_MAP_X_DIMENSION = 50      # 50 as taken from research paper
SOM_MAP_Y_DIMENSION = 50     # 50 as taken from research paper
SOM_TRAINING_EPOCHS = 100

#%% md
# Load and preprocess data
#%%
_filepaths = [DATA_DIRECTORY + "/" + filename for filename in os.listdir(DATA_DIRECTORY)]

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

# Normalize
files = [preprocessing.normalize(df, norm=NORMALIZATION_NORM) for df in files]

# Split train and test data -> first x% taken (not randomly)
def split_df_not_randomly(df, split: float):
    split_index = int(len(df) * split)
    return df[:split_index], df[split_index:]


train_files, test_files = zip(*[split_df_not_randomly(df, TEST_TRAIN_SPLIT) for df in files])

print("Data loaded and processed. First train file:\n")

#%% md
# Self-organizing map (SOM)
#%%
"""
data = train_files[0].iloc[:, 1:].to_numpy()   # don't use timestamp

som = SOM(x=SOM_MAP_X_DIMENSION, y=SOM_MAP_Y_DIMENSION, input_len=data.shape[1],
          random_seed=RANDOM_SEED)
som.train(data, num_iteration=SOM_TRAINING_EPOCHS, verbose=True)

som.distance_map()
"""

predecessors = [pd.DataFrame(file).shift(1) for file in files]
originals = [pd.DataFrame(file) for file in files]

distances = [np.linalg.norm(orig - pred, ord=2, axis=1) for orig, pred in zip(originals, predecessors)]

experiment_names = ['C11', 'C13', 'C14', 'C15', 'C16', 'C7', 'C8', 'C9']

fig, ax = plt.subplots(8, 1)
fig.set_size_inches(6, 24)

i = 0
for distance in distances:
    axis = ax[i]
    axis.plot(distance)
    axis.set_title(f"Experiment Nr. {experiment_names[i]}")

    i += 1

plt.tight_layout()
plt.show()
