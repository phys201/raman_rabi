from unittest import TestCase

import raman_rabi
from raman_rabi import testing
from raman_rabi import RRDataContainer
from raman_rabi import rr_io
from raman_rabi import rr_model
import numpy as np

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

class TestRR_MODEL(TestCase):
    def test_likelihood_zero_for_nonesense(self):
        testfilepath = "testdata/21.07.56_Pulse Experiment_E1 Raman Pol p0 Rabi Rep Readout - -800 MHz, ND0.7 Vel1, Presel 6 (APD2, Win 3)_ID10708_image.txt"
        s = RRDataContainer.RRDataContainer(testfilepath)
        s_likelihood = rr_model.likelihood_mN1(s, 0, 0, 0, 0, 0, 0, 0, 0)
        self.assertTrue(s_likelihood == 0.0)

    def test_likelihood_ratio(self):
        testfilepath = "testdata/21.07.56_Pulse Experiment_E1 Raman Pol p0 Rabi Rep Readout - -800 MHz, ND0.7 Vel1, Presel 6 (APD2, Win 3)_ID10708_image.txt"
        s = RRDataContainer.RRDataContainer(testfilepath)
        s_likelihood = rr_model.likelihood_mN1(s, 0, 40, 6.10, 16.6881, 1/63.8806, 5.01886, -np.pi/8.77273, 1/8.5871)
        s2_likelihood = rr_model.likelihood_mN1(s, 0, 40, 10*6.10, 16.6881, 1/63.8806, 5.01886, -np.pi/8.77273, 1/8.5871)
        print(s_likelihood)
        self.assertTrue(s_likelihood > s2_likelihood)