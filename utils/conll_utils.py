import json


def get_data_conll_query(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)
    x = list(data.keys())
    y = list(data.values())
    return x, y
