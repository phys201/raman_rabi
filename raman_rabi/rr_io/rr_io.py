import os
import pandas as pd
from raman_rabi import RRDataContainer 

"""
This function assumes it's being given a relative
path to the data file from the working directory
"""
def load_data(data_file):
    loaded_data = RRDataContainer(data_file)
    return loaded_data

def get_example_data_file_path(filename, data_dir='testdata'):
    # __file__ is the location of the source file currently in use (so
    # in this case io.py). We can use it as base path to construct
    # other paths from that should end up correct on other machines or
    # when the package is installed
    start = os.path.abspath(__file__)
    start_dir = os.path.dirname(start)
    # Go up two directories
    up_dir1 = os.path.split(start_dir)[0]
    up_dir2 = os.path.split(up_dir1)[0]
    data_dir = os.path.join(up_dir2, data_dir)
    return os.path.join(up_dir2, data_dir, filename)