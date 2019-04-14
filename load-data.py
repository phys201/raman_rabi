# -*- coding: utf-8 -*-

class DataObject:
    import pandas as pd
    """A class that organizes data and functions"""
    def __init__(self, filepath):
        self.filename = filepath
        

    def f(self):
        data = pd.read_csv(self.filename, sep=" ", header=None)
        return 'hello world'
