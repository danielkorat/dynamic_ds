import numpy as np


def get_data_conll_query(data_path):
    if isinstance(data_path, list):
        data_path = data_path[0]
    data = np.load(data_path)
    x = data['x']
    y = data['y']
    return x, y
