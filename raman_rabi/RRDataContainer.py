# -*- coding: utf-8 -*-
__author__ = ["Jamelle Watson-Daniels", "Taylor Patti", "Soumya Ghosh"]
__copyright__ = "Copyright 2019"
__version__ = "1.0.0"
__status__ = "Development"

import pandas as pd
class RRDataContainer:
    """A class that puts data into a pandas data frame object.
    
    Parameters:
        filepath: path to data file (specified as a string)  
    """
    def __init__(self, filepath):        
        """
        The constructor for an RRDataContainer object

        Parameters:
            filepath (string): the path to the data file, relative to the working directory

        Outputs:
            self (RRDataContainer): an RRDataContainer object, which contains a Pandas
                DataFrame holding the data loaded from the file
        """
        self.filename = filepath
        self.dataframe = pd.read_csv(filepath, encoding='utf-8', delimiter="\t", header=None)
        
    def get_df(self):
        """
        Returns data frame object that contains data from file.

        Parameters:
            none

        Outputs:
            dataframe (Pandas DataFrame): a Pandas DataFrame holding the data
        """
        return self.dataframe
