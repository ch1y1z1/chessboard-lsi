import json
from bunch import Bunch

def get_config_from_json(config_file):
    """
    get_config_from_json aims to extract the configuration from json file.
    :param config_file: open the configuration file
    :return: config dictionary
    """
    config_dict = json.load(config_file)
    config = Bunch(config_dict)
    return config

if __name__ == '__main__':
    config_file = open('./config_file.json')
    config = get_config_from_json(config_file)
    print(config.wavelength)