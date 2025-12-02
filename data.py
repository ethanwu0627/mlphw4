import pandas as pd
from PIL import Image

splits = {'train': 'mnist/train-00000-of-00001.parquet', 'test': 'mnist/test-00000-of-00001.parquet'}
df_train = pd.read_parquet("hf://datasets/ylecun/mnist/" + splits["train"])
df_test = pd.read_parquet("hf://datasets/ylecun/mnist/" + splits["test"])


import kagglehub

# Download latest version
path = kagglehub.dataset_download("karnikakapoor/digits")

print("Path to dataset files:", path)