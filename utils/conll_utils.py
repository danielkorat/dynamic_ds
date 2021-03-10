import numpy as np


def get_data_conll_query(data_path):
    data = np.load(data_path)
    x = data['x']
    y = data['y']
    return x, y
