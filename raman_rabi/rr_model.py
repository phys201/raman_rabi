from raman_rabi import RRDataContainer
import numpy as np
from numpy import random
import pandas as pd
import emcee

def ideal_model(steps, time_min, time_max, BG, Ap, Gammap, Ah, Omegah, Gammadeph):
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
        likelihood (float): the value of the likelihood function with these parameters
        mu (array of floats): the values of the model given these parameters
    """
    mN1_size = mN1_data.get_df().shape[0]
    mN1_data = scale_factor*np.sum(mN1_data.get_df()) / (dataN*mN1_size)

    time, mu = ideal_model(len(mN1_data), time_min, time_max, BG, Ap, Gammap, Ah, Omegah, Gammadeph)

    #we make data into unit normal distribution, uncertainty sigma of shot noise = sqrt(mu)
    z_data = (mN1_data - mu)/np.sqrt(mu)
    likelihood = (2*np.pi)**(-len(mN1_data)/2) * np.exp(-np.sum(z_data**2)/2.)
    return likelihood, mu


#def unbinned_loglikelihood_mN1(mN1_data, time_min, time_max, BG, Ap, Gammap, Ah, Omegah, Gammadeph, fromcsv, dataN=10, scale_factor=100*100):
def unbinned_loglikelihood_mN1(theta, mN1_data, time_min, time_max, fromcsv, dataN=10, scale_factor=100*100):
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
    

def generate_test_data(theta, timesteps, samples, time_min, time_max, dataN=10, scale_factor=100*100):
    BG, Ap, Gammap, Ah, Omegah, Gammadeph = theta
    time, mu = ideal_model(timesteps, time_min, time_max, BG, Ap, Gammap, Ah, Omegah, Gammadeph)
    uncertainty = np.random.normal(scale=np.sqrt(mu), size=(samples, timesteps))
    mu_mat = np.tile(mu, (samples, 1))
    test_data = dataN*(mu_mat + uncertainty)/scale_factor
    return pd.DataFrame(test_data)


def log_prior(theta):
    return 0


def log_posterior(theta, mN1_data, time_min, time_max, fromcsv, dataN=10, scale_factor=100*100):
    return log_prior(theta) + unbinned_loglikelihood_mN1(theta, mN1_data, time_min, time_max, fromcsv, dataN=10, scale_factor=100*100)


def Walkers(mN1_data, guesses, time_min, time_max, fromcsv, dataN=10, scale_factor=100*100, nwalkers=20, nsteps=50):
    #guesses = [BG, Ap, Gammap, Ah, Omegah, Gammadeph]
    BG, Ap, Gammap, Ah, Omegah, Gammadeph = guesses
    ndim = len(guesses)
    starting_positions = [guesses + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, 
                                args=(mN1_data, time_min, time_max, fromcsv))
    sampler.run_mcmc(starting_positions, nsteps)
    return sampler
    
