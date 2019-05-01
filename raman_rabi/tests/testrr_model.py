from unittest import TestCase

import raman_rabi
from raman_rabi import RRDataContainer
from raman_rabi import rr_io
from raman_rabi import rr_model

import numpy as np
import pandas as pd

testfilename = "21.07.56_Pulse Experiment_E1 Raman Pol p0 Rabi Rep Readout - -800 MHz, ND0.7 Vel1, Presel 6 (APD2, Win 3)_ID10708_image.txt"
RRData = rr_io.load_data(rr_io.get_example_data_file_path(testfilename))

class TestRR_MODEL(TestCase):
    def test_likelihood_zero_for_nonesense(self):
        s_likelihood = rr_model.likelihood_mN1(RRData, 0, 0, 0, 0, 0, 0, 0, 0)[0]
        self.assertTrue(s_likelihood == 0.0)

    def test_likelihood_mN1(self):
        time_min = 0
        time_max = 40
        theta = np.array([6.10, 16.6881, 1/63.8806, 5.01886, -np.pi/8.77273, 1/8.5871])
        likelihood = rr_model.likelihood_mN1(RRData,time_min,time_max,*theta)[0]
        self.assertAlmostEqual(likelihood,3.2035056889734263e-74)


    def test_likelihood_ratio(self):
        s_likelihood = rr_model.likelihood_mN1(RRData, 0, 40, 6.10, 16.6881, 1/63.8806, 5.01886, -np.pi/8.77273, 1/8.5871)[0]
        s2_likelihood = rr_model.likelihood_mN1(RRData, 0, 40, 10*6.10, 16.6881, 1/63.8806, 5.01886, -np.pi/8.77273, 1/8.5871)[0]
        self.assertTrue(s_likelihood > s2_likelihood)

    def test_loglikelihood_for_nonesense(self):
        # previously estimated parameters:
        theta = np.array([6.10, 16.6881, 1/63.8806, 5.01886, -np.pi/8.77273, 1/8.5871])
        # generate some data
        test_data = rr_model.generate_test_data(theta, 161, 500, 0, 40)
        # initial guess with a deliberately bad parameter
        bad_guess = theta
        bad_guess[0] = -100
        # calculate loglikelihood
        loglikelihood = rr_model.unbinned_loglikelihood_mN1(bad_guess,
                test_data, 0, 40, False)
        self.assertTrue(np.isnan(loglikelihood))

    def test_unbinned_loglikelihood_mN1(self):
        theta = np.array([6.10, 16.6881, 1/63.8806, 5.01886, -np.pi/8.77273, 1/8.5871])
        time_min = 0
        time_max = 40
        fromcsv = True

        likelihood = rr_model.unbinned_loglikelihood_mN1(theta,RRData,time_min,time_max,fromcsv)
        self.assertTrue(np.round(likelihood,2) == -9253.76)

    def test_laserskew_unbinned_loglikelihood_mN1(self):
        theta = np.array([6.10, 16.6881, 1/63.8806, 5.01886, -np.pi/8.77273, 1/8.5871])
        data_length = len(RRData.get_df())
        theta = np.concatenate((theta,np.ones(data_length)),axis=0)
        time_min = 0
        time_max = 40
        fromcsv = True
        likelihood = rr_model.laserskew_unbinned_loglikelihood_mN1(theta,RRData,
            time_min,time_max,fromcsv)
        self.assertTrue(np.round(likelihood,2) == -9253.76)

    def test_generate_test_data(self):
        theta = np.array([6.10, 16.6881, 1/63.8806, 5.01886, -np.pi/8.77273, 1/8.5871])
        timesteps = 5
        samples = 5
        time_min = 0
        time_max = 40
        # set the seed for reproducibility
        np.random.seed(0)
        test_data = rr_model.generate_test_data(theta,timesteps,samples,time_min,time_max)
        correct_values = np.array([[ 0.0371,  0.0207,  0.0228,  0.0256,  0.0222],
               [ 0.0227,  0.0231,  0.018 ,  0.0161,  0.0166],
               [ 0.0286,  0.0253,  0.0219,  0.017 ,  0.0167],
               [ 0.0296,  0.0255,  0.0177,  0.0178,  0.0117],
               [ 0.0143,  0.0218,  0.0223,  0.0135,  0.0238]])
        self.assertTrue(np.all(np.round(test_data.values,4) == correct_values))

    def test_log_prior(self):
        theta = np.array([6.10, 16.6881, 1/63.8806, 5.01886, -np.pi/8.77273, 1/8.5871])
        logprior = rr_model.log_prior(theta)
        self.assertTrue(logprior == 0)

    def test_log_prior_uniform(self):
        # this time, use a (proper) uniform prior for one of the parameters
        # and make sure we get -inf when we plug in an out-of-bounds value
        theta = np.array([6.10, 16.6881, 1/63.8806, 5.01886, 
            -np.pi/8.77273, 1/8.5871])
        priors = np.array([['flat'],
                           ['uniform',10,18],
                           ['flat'],
                           ['flat'],
                           ['flat'],
                           ['flat']])
        wrong_theta = theta
        wrong_theta[1] = 9
                           
        logprior = rr_model.log_prior(wrong_theta,priors)
        self.assertTrue(logprior == -np.inf)


    def test_log_posterior(self):
        theta = np.array([6.10, 16.6881, 1/63.8806, 5.01886, -np.pi/8.77273, 1/8.5871])
        time_min = 0
        time_max = 40
        fromcsv = True

        logposterior = rr_model.log_posterior(theta,RRData,time_min,time_max,fromcsv)
        self.assertTrue(np.round(logposterior,2) == -9253.76)



    def test_ideal_model(self):
        nsteps = 5
        time_min = 0
        time_max = 40
        theta = np.array([6.10, 16.6881, 1/63.8806, 5.01886, -np.pi/8.77273, 1/8.5871])
        time, mu = rr_model.ideal_model(nsteps,time_min,time_max,*theta)
        self.assertTrue(np.all(time == np.array([0.,10.,20.,30.,40.])))
        self.assertTrue(np.all(np.round(mu,2) == np.array([27.81,18.95,18.61,16.5,15.01])))

    def test_laserskew_log_posterior(self):
        theta = np.array([6.10, 16.6881, 1/63.8806, 5.01886, -np.pi/8.77273, 1/8.5871])
        time_min = 0
        time_max = 40
        fromcsv = True
        data_length = len(RRData.get_df())
        theta = np.concatenate((theta,np.ones(data_length)),axis=0)
        logposterior = rr_model.laserskew_log_posterior(theta,RRData,time_min,
                time_max,fromcsv)
        self.assertTrue(np.round(logposterior,2) == -9253.76)



    def test_parameter_estimation(self):
        # previously estimated parameters:
        theta = np.array([6.10, 16.6881, 1/63.8806, 5.01886, -np.pi/8.77273, 1/8.5871])

        # generate some data
        test_data = rr_model.generate_test_data(theta, 161, 500, 0, 40)

        # run MCMC on the test data and see if it's pretty close to the original theta
        guesses = theta
        numdim = len(guesses)
        numwalkers = 100
        numsteps = 500
        np.random.seed(0)
        test_samples = rr_model.Walkers(test_data, guesses, 0, 40, False, dataN=10, scale_factor=100*100, nwalkers=numwalkers, nsteps=numsteps)
        burn_in_time = 100
        samples = test_samples.chain[:,burn_in_time:,:]
        traces = samples.reshape(-1, numdim).T
        parameter_samples = pd.DataFrame({'BG': traces[0], 'Ap': traces[1], 'Gammap': traces[2], 'Ah': traces[3], 'Omegah': traces[4], 'Gammadeph': traces[5]})
        MAP = parameter_samples.quantile([0.50], axis=0)
        self.assertTrue(abs((MAP['BG'].values[0]-guesses[0])/guesses[0]) < 0.2)
        self.assertTrue(abs((MAP['Ap'].values[0]-guesses[1])/guesses[1]) < 0.2)
        self.assertTrue(abs((MAP['Gammap'].values[0]-guesses[2])/guesses[2]) < 0.2)
        self.assertTrue(abs((MAP['Ah'].values[0]-guesses[3])/guesses[3]) < 0.2)
        self.assertTrue(abs((MAP['Omegah'].values[0]-guesses[4])/guesses[4]) < 0.2)
        self.assertTrue(abs((MAP['Gammadeph'].values[0]-guesses[5])/guesses[5]) < 0.2)

    def test_laserskew_parameter_estimation(self):
        # previously estimated parameters:
        data_length = 90
        theta = np.concatenate( (np.array([6.10, 16.6881, 
                                        1/63.8806, 5.01886, 
                                        -np.pi/8.77273, 1/8.5871]), 
                                    np.ones(data_length)), axis=0)

        # generate some data
        test_data = rr_model.generate_test_data(theta[0:6], 161, 
                                                data_length, 0, 40)

        # run MCMC on the test data and see if it's pretty close to the original theta
        guesses = theta
        numdim = len(guesses)
        numwalkers = 200
        numsteps = 500
        np.random.seed(0)
        test_samples = rr_model.laserskew_Walkers(test_data, guesses, 
                                                  0, 40, False, dataN=10, 
                                                  scale_factor=100*100, 
                                                  nwalkers=numwalkers, 
                                                  nsteps=numsteps)
        burn_in_time = 200
        samples = test_samples.chain[:,burn_in_time:,:]
        traces = samples.reshape(-1, numdim).T
        parameter_samples = pd.DataFrame({'BG': traces[0], 
                                          'Ap': traces[1], 
                                          'Gammap': traces[2], 
                                          'Ah': traces[3], 
                                          'Omegah': traces[4], 
                                          'Gammadeph': traces[5] })
        laserskew_samples = pd.DataFrame(traces[6:].T)
        MAP = parameter_samples.quantile([0.50], axis=0)
        laserskew_MAP = laserskew_samples.quantile([0.50],axis=0)
        self.assertTrue(abs((MAP['BG'].values[0]-guesses[0])/guesses[0]) < 0.2)
        self.assertTrue(abs((MAP['Ap'].values[0]-guesses[1])/guesses[1]) < 0.2)
        self.assertTrue(abs((MAP['Gammap'].values[0]-guesses[2])/guesses[2]) < 0.2)
        self.assertTrue(abs((MAP['Ah'].values[0]-guesses[3])/guesses[3]) < 0.2)
        self.assertTrue(abs((MAP['Omegah'].values[0]-guesses[4])/guesses[4]) < 0.2)
        self.assertTrue(abs((MAP['Gammadeph'].values[0]-guesses[5])/guesses[5]) < 0.2)
        laserskew_columns = list(laserskew_MAP)
        for column in laserskew_columns:
            self.assertTrue(abs((laserskew_MAP[column].values[0]-guesses[6+column])/guesses[6+column]) < 0.2)

