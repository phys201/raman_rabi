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
