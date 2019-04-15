# -*- coding: utf-8 -*-
__author__ = ["Jamelle Watson-Daniels", "Taylor Patti", "Soumya Gosh"]
__copyright__ = "Copyright 2019"
__version__ = "1.0.0"
__status__ = "Development"

import pandas as pd
class DataObject:
    """A class that puts data into a pandas data frame object.
    
    Parameters:
        filepath: path to data file (specified as a string)  
    """
    def __init__(self, filepath):        
        self.filename = filepath
        
    def df(self):
        """
        Returns data frame object that contains data from file.
        """
        dataframe = pd.read_csv(self.filename, delimiter="\t", header=None)
        return dataframe
