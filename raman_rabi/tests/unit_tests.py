from unittest import TestCase

import raman_rabi
from raman_rabi import testing
from raman_rabi import RRDataContainer
from raman_rabi import rr_io

class TestTesting(TestCase):
    def test_is_string(self):
        s = testing.hello()
        self.assertTrue(isinstance(s, str))

class TestRR_IO(TestCase):
    def test_is_RRDataContainer(self):
        testfilepath = "testdata/21.07.56_Pulse Experiment_E1 Raman Pol p0 Rabi Rep Readout - -800 MHz, ND0.7 Vel1, Presel 6 (APD2, Win 3)_ID10708_image.txt"
        s = rr_io.load_data(testfilepath)
        self.assertTrue(isinstance(s,RRDataContainer.RRDataContainer))

class TestRRDataContainer(TestCase):
    def test_correct_dimensions(self):
        testfilepath = "testdata/21.07.56_Pulse Experiment_E1 Raman Pol p0 Rabi Rep Readout - -800 MHz, ND0.7 Vel1, Presel 6 (APD2, Win 3)_ID10708_image.txt"
        s = RRDataContainer.RRDataContainer(testfilepath)
        self.assertTrue(s.get_df().shape == (20,161))
