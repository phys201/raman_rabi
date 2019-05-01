from raman_rabi import RRDataContainer
import numpy as np
from numpy import random
import pandas as pd
import emcee

"""
This submodule provides the generative model (Model 1) for the Raman-Rabi data as well as
an MCMC sampler complete with log-prior and log-posterior functions for Model 1.
"""

def ideal_model(steps, time_min, time_max, BG, Ap, Gammap, Ah, Omegah, Gammadeph):
    """
    The generative model for the Raman-Rabi data, before adding noise. Gaussian noise
    is added by generate_test_data, below.
    
    Parameters:
        steps: the number of time divisions to use in the simulated data (int)
        time_min: minimum Raman-Rabi pulse time (float)
        time_max: maximum Raman-Rabi pulse time (float)
        BG: background fluoresence parameter (float)
        Ap: parasitic loss strength parameter (float)
        Gammap: parasitic loss time-scale parameter (float)
        Ah: hyperfine flip-flop strength parameter (float)
        Omegah: hyperfine flip-flop time-scale parameter (float)
        Gammadeph: Raman-Rabi dephasing time-scale parameter (float)

    Returns:
        time: the time points at which the simulated readouts were taken (array of floats)
        mu: the values of the readout at each time point (array of floats)
    """
    time = np.linspace(time_min, time_max, steps)
    mu = BG + Ap*np.exp(-Gammap*time) + Ah*np.cos(Omegah*time)*np.exp(-Gammadeph*time)
    return time, mu

def likelihood_mN1(mN1_data, time_min, time_max, BG, Ap, Gammap, Ah, Omegah, Gammadeph, dataN=10, scale_factor=100*100):
    """
    The likelihood function of mN1 spin Raman-Rabi electronic-nuclear flip-flop data, assuming only shot noise from
    continuous fluoresence spectrum (Poisson distribution becomes Gaussian
    
    Parameters:
        mN1_data: fluoresence data for each time step (RRDataContainer)
        time_min: minimum Raman-Rabi pulse time (float)
        time_max: maximum Raman-Rabi pulse time (float)
        BG: background fluoresence parameter (float)
        Ap: parasitic loss strength parameter (float)
        Gammap: parasitic loss time-scale parameter (float)
        Ah: hyperfine flip-flop strength parameter (float)
        Omegah: hyperfine flip-flop time-scale parameter (float)
        Gammadeph: Raman-Rabi dephasing time-scale parameter (float)
        dataN (optional): number of experiment repititions summed (int)
        scale_factor (optional): nuclear spin signal multiplier (float)

    Returns:
        likelihood: the value of the likelihood function with these parameters (float)
        mu: the values of the model given these parameters (array of floats)
    """
    mN1_size = mN1_data.get_df().shape[0]
    mN1_data = scale_factor*np.sum(mN1_data.get_df()) / (dataN*mN1_size)

    time, mu = ideal_model(len(mN1_data), time_min, time_max, BG, Ap, Gammap, Ah, Omegah, Gammadeph)

    #we make data into unit normal distribution, uncertainty sigma of shot noise = sqrt(mu)
    z_data = (mN1_data - mu)/np.sqrt(mu)
    likelihood = (2*np.pi)**(-len(mN1_data)/2) * np.exp(-np.sum(z_data**2)/2.)
    return likelihood, mu


def unbinned_loglikelihood_mN1(theta, mN1_data, time_min, time_max, fromcsv, dataN=10, scale_factor=100*100):
    """
    This function calculates the unbinned log-likelihood of model 1

    Parameters:
        theta: the model parameters to use in this log-posterior calculation (array)
        mN1_data: fluoresence data for each time step (RRDataContainer)
        time_min: minimum Raman-Rabi pulse time (float)
        time_max: maximum Raman-Rabi pulse time (float)
        fromcsv: marks whether these data were read from a CSV file (bool)
        dataN (optional): number of experiment repititions summed (int)
        scale_factor (optional): nuclear spin signal multiplier (float)

    Returns:
        loglikelihood: the log-likelihood of the data given the parameters in theta (float)
    """
    BG, Ap, Gammap, Ah, Omegah, Gammadeph = theta
    if fromcsv:
        mN1_data = mN1_data.get_df().values
    else:
        mN1_data = mN1_data.values
    mN1_data = scale_factor*mN1_data/dataN
    time, mu = ideal_model(mN1_data.shape[1], time_min, time_max, BG, Ap, Gammap, Ah, Omegah, Gammadeph)
    mu_mat = np.tile(mu, (mN1_data.shape[0], 1))
    z_data = (mN1_data - mu_mat)/np.sqrt(mu_mat)
    loglikelihood = np.log( (2*np.pi)**(-len(mN1_data)/2) ) - np.sum(z_data**2)/2.
    return loglikelihood

def laserskew_unbinned_loglikelihood_mN1(theta, mN1_data, time_min, time_max, fromcsv, dataN=10, scale_factor=100*100):
    """
    This function calculates the unbinned log-likelihood of model 1 for N data points INCLUDING N parameters a_i of laser skew strength (a between zero and infinity)

    Parameters:
        theta: the model parameters to use in this log-posterior calculation (array)
        mN1_data: fluoresence data for each time step (RRDataContainer)
        time_min: minimum Raman-Rabi pulse time (float)
        time_max: maximum Raman-Rabi pulse time (float)
        fromcsv: marks whether these data were read from a CSV file (bool)
        dataN (optional): number of experiment repititions summed (int)
        scale_factor (optional): nuclear spin signal multiplier (float)

    Returns:
        loglikelihood: the log-likelihood of the data given the parameters in theta (float)
    """
    BG, Ap, Gammap, Ah, Omegah, Gammadeph = theta[0:6]
    a_vec = theta[6:len(theta)]
    a_vec.shape = (len(a_vec), 1)

    if fromcsv:
        mN1_data = mN1_data.get_df().values
    else:
        mN1_data = mN1_data.values
    mN1_data = scale_factor*mN1_data/dataN
    time, mu = ideal_model(mN1_data.shape[1], time_min, time_max, BG, Ap, Gammap, Ah, Omegah, Gammadeph)
    mu_mat = np.tile(mu, (mN1_data.shape[0], 1))
    mu_mat = mu_mat*a_vec
    z_data = (mN1_data - mu_mat)/np.sqrt(mu_mat)
    loglikelihood = np.log( (2*np.pi)**(-len(mN1_data)/2) ) - np.sum(z_data**2)/2.
    return loglikelihood
    

def generate_test_data(theta, timesteps, samples, time_min, time_max, dataN=10, scale_factor=100*100):
    """
    This function wraps the ideal model function, adds Gaussian noise to the data, and returns
    a test dataset.

    Parameters:
        theta: the model parameters to use in this log-posterior calculation (array)
        timesteps: the number of time divisions to use in the simulated data (int)
        samples: the number of samples (trials) to take (int)
        time_min: minimum Raman-Rabi pulse time (float)
        time_max: maximum Raman-Rabi pulse time (float)
        dataN (optional): number of experiment repititions summed (int)
        scale_factor (optional): nuclear spin signal multiplier (float)

    Returns:
        test_data: a pandas DataFrame containing the test data (DataFrame)
    """
    BG, Ap, Gammap, Ah, Omegah, Gammadeph = theta
    time, mu = ideal_model(timesteps, time_min, time_max, BG, Ap, Gammap, Ah, Omegah, Gammadeph)
    uncertainty = np.random.normal(scale=np.sqrt(mu), size=(samples, timesteps))
    mu_mat = np.tile(mu, (samples, 1))
    test_data = dataN*(mu_mat + uncertainty)/scale_factor
    return pd.DataFrame(test_data)


def log_prior(theta,priors=None):
    """
    The log prior for all parameters in the Raman-Rabi model. All parameters
    have improper flat priors.

    Parameters:
        theta (array): a list of the input parameters for the model
        priors (optional): a list of tuples (all ('flat') by default) specifying
            what type of prior to use for each parameter, and the parameters of
            that prior. For 'uniform' these consist of the lower and upper limits,
            respectively.

    Returns:
        log_prior (float): the value of the prior, either 0 or -np.inf depending
            on priors and theta
    """
    if priors is None:
        priors=np.repeat([['flat']],len(theta),axis=0)
    logprior_arr = []
    for param, prior in zip(theta,priors):
        if prior[0] == 'flat':
            logprior_arr.append(0)
        elif prior[0] == 'uniform':
            if (param <= prior[1]) or (param >= prior[2]):
                return -np.inf
            else:
                logprior_arr.append(0)
        else:
            print('>>> Unknown prior specified')
            
    return np.sum(logprior_arr)


def log_posterior(theta, mN1_data, time_min, time_max, fromcsv, dataN=10, scale_factor=100*100):
    """
    The log posterior for the Raman-Rabi model, using the unbinned log-likelihood.

    Parameters:
        theta: the model parameters to use in this log-posterior calculation (array)
        mN1_data: fluoresence data for each time step (RRDataContainer)
        time_min: minimum Raman-Rabi pulse time (float)
        time_max: maximum Raman-Rabi pulse time (float)
        dataN (optional): number of experiment repititions summed (int)
        scale_factor (optional): nuclear spin signal multiplier (float)

    Returns:
        log-posterior: the value of the log-posterior for the data given the parameters in theta
    """
    return log_prior(theta) + unbinned_loglikelihood_mN1(theta, mN1_data, time_min, time_max, fromcsv, dataN=10, scale_factor=100*100)

def laserskew_log_posterior(theta, mN1_data, time_min, time_max, fromcsv, dataN=10, scale_factor=100*100):
    """
    The log posterior for the Raman-Rabi model, using the unbinned log-likelihood WITH LASER SKEW PARAMETER a.

    Parameters:
        theta: the model parameters to use in this log-posterior calculation (array)
        mN1_data: fluoresence data for each time step (RRDataContainer)
        time_min: minimum Raman-Rabi pulse time (float)
        time_max: maximum Raman-Rabi pulse time (float)
        dataN (optional): number of experiment repititions summed (int)
        scale_factor (optional): nuclear spin signal multiplier (float)

    Returns:
        log-posterior: the value of the log-posterior for the data given the parameters in theta
    """
    return log_prior(theta) + laserskew_unbinned_loglikelihood_mN1(theta, mN1_data, time_min, time_max, fromcsv, dataN=10, scale_factor=100*100)


def Walkers(mN1_data, guesses, time_min, time_max, fromcsv, dataN=10, scale_factor=100*100, nwalkers=20, nsteps=50):
    """
    This function samples the posterior using MCMC.

    Parameters:
        mN1_data: fluoresence data for each time step (RRDataContainer)
        guesses: the initial guesses for the parameters of the model (array of floats)
        time_min: minimum Raman-Rabi pulse time (float)
        time_max: maximum Raman-Rabi pulse time (float)
        fromcsv: marks whether these data were read from a CSV file (bool)
        dataN (optional): number of experiment repititions summed (int)
        scale_factor (optional): nuclear spin signal multiplier (float)
        nwalkers (optional): the number of walkers with which to sample (int)
        nsteps (optional): the number of steps each walker should take (int)

    Returns:
       sampler: the sampler object which now contains the samples taken by nwalkers
           walkers over nsteps steps
    """
    BG, Ap, Gammap, Ah, Omegah, Gammadeph = guesses
    ndim = len(guesses)
    starting_positions = [guesses + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, 
                                args=(mN1_data, time_min, time_max, fromcsv))
    sampler.run_mcmc(starting_positions, nsteps)
    return sampler

def laserskew_Walkers(mN1_data, guesses, time_min, time_max, fromcsv, dataN=10, scale_factor=100*100, nwalkers=20, nsteps=50):
    """
    This function samples the posterior using MCMC WITH LASER SKEW PARAMETER a.

    Parameters:
        mN1_data: fluoresence data for each time step (RRDataContainer)
        guesses: the initial guesses for the parameters of the model (array of floats)
        time_min: minimum Raman-Rabi pulse time (float)
        time_max: maximum Raman-Rabi pulse time (float)
        fromcsv: marks whether these data were read from a CSV file (bool)
        dataN (optional): number of experiment repititions summed (int)
        scale_factor (optional): nuclear spin signal multiplier (float)
        nwalkers (optional): the number of walkers with which to sample (int)
        nsteps (optional): the number of steps each walker should take (int)

    Returns:
       sampler: the sampler object which now contains the samples taken by nwalkers
           walkers over nsteps steps
    """
    BG, Ap, Gammap, Ah, Omegah, Gammadeph = guesses[0:6]
    a_vec = guesses[5:len(guesses)]

    ndim = len(guesses)
    starting_positions = [guesses + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, laserskew_log_posterior, 
                                args=(mN1_data, time_min, time_max, fromcsv))
    sampler.run_mcmc(starting_positions, nsteps)
    return sampler
    
