import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    from config import get_config_from_json
    config_file = open('./config_file.json')
    config = get_config_from_json(config_file)

    print("{:.4g}".format(0.25*2 - 0.1979))
    print("{:.4g}".format(0.15*2 - 0.1016))
    print("{:.4g}".format(0.05*2 + 0.03179))
    print("{:.4g}".format(0.2*2  - 0.1589))