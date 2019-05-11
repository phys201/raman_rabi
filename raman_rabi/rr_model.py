from raman_rabi import RRDataContainer
import numpy as np
from numpy import random
import pandas as pd
import emcee
import matplotlib.pyplot as plt

"""
This submodule provides the generative model (Model 1) for the Raman-Rabi data as well as
an MCMC sampler complete with log-prior and log-posterior functions for Model 1.

Note: theta is a list of the input parameters for the model in the following form
    'B_G','A_p', 'Gamma_p' , 'A_h', 'Omega_h', 'Gamma_deph','Skew'
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

def likelihood_mN1(mN1_data, time_min, time_max, BG, Ap, Gammap, Ah, Omegah, Gammadeph, dataN, runN, scale_factor=100*100):
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
        dataN: number of experiment repititions summed (int)
        scale_factor (optional): nuclear spin signal multiplier (float)

    Returns:
        likelihood: the value of the likelihood function with these parameters (float)
        mu: the values of the model given these parameters (array of floats)
    """
    mN1_size = mN1_data.get_df().shape[0]
    BG = BG*runN*dataN*mN1_size/scale_factor
    Ap = Ap*runN*dataN*mN1_size/scale_factor
    Ah = Ah*runN*dataN*mN1_size/scale_factor
    #mN1_data = scale_factor*np.sum(mN1_data.get_df()) / (dataN*mN1_size)
    mN1_data = runN*np.sum(mN1_data.get_df())
    time, mu = ideal_model(len(mN1_data), time_min, time_max, BG, Ap, Gammap, Ah, Omegah, Gammadeph)

    #we make data into unit normal distribution, uncertainty sigma of shot noise = sqrt(mu)
    z_data = (mN1_data - mu)/np.sqrt(mu)
    likelihood = (2*np.pi)**(-len(mN1_data)/2) * np.exp(-np.sum(z_data**2)/2.)
    return likelihood, mu*scale_factor/(runN*dataN*mN1_size)

def general_loglikelihood(theta, mN1_data, time_min, time_max, fromcsv, dataN, runN, scale_factor, withlaserskew = False):
    BG, Ap, Gammap, Ah, Omegah, Gammadeph = theta[0:6]
    BG = BG*runN*dataN/scale_factor
    Ap = Ap*runN*dataN/scale_factor
    Ah = Ah*runN*dataN/scale_factor
    if withlaserskew:
        a_vec = np.array(theta[6:len(theta)])
        a_vec.shape = (len(a_vec), 1)
    
    if fromcsv:
        mN1_data = mN1_data.get_df().values
    else:
        mN1_data = mN1_data.values
    #mN1_data = scale_factor*mN1_data/dataN
    mN1_data = runN*mN1_data #make a Poissonian distribution again
    time, mu = ideal_model(mN1_data.shape[1], time_min, time_max, BG, Ap, Gammap, Ah, Omegah, Gammadeph)
    mu_mat = np.tile(mu, (mN1_data.shape[0], 1))
    if withlaserskew:
        mu_mat = np.multiply(mu_mat,a_vec)
    z_data = (mN1_data - mu_mat)/np.sqrt(mu_mat)
    loglikelihood = np.log( (2*np.pi)**(-len(mN1_data)/2) ) - np.sum(z_data**2)/2.
    return loglikelihood


def generate_test_data(theta, timesteps, samples, time_min, time_max, dataN, runN, scale_factor=100*100, include_laserskews=False):
    """
    This function wraps the ideal model function, adds Gaussian noise to the data, and returns
    a test dataset.

    Parameters:
        theta: the model parameters to use in this log-posterior calculation (array)
        timesteps: the number of time divisions to use in the simulated data (int)
        samples: the number of samples (trials) to take (int)
        time_min: minimum Raman-Rabi pulse time (float)
        time_max: maximum Raman-Rabi pulse time (float)
        dataN: number of experiment repititions summed (int)
        scale_factor (optional): nuclear spin signal multiplier (float)
        include_laserskews (optional): does theta include laser skew parameters? (boolean)

    Returns:
        test_data: a pandas DataFrame containing the test data (DataFrame)
    """
    BG, Ap, Gammap, Ah, Omegah, Gammadeph = theta[0:6]
    BG = BG*runN*dataN/scale_factor
    Ap = Ap*runN*dataN/scale_factor
    Ah = Ah*runN*dataN/scale_factor
    if include_laserskews:
        laserskews = theta[6:]
    time, mu = ideal_model(timesteps, time_min, time_max, BG, Ap, Gammap, Ah, Omegah, Gammadeph)
    uncertainty = np.random.normal(scale=np.sqrt(mu), size=(samples, timesteps))

    mu_mat = np.tile(mu, (samples, 1))
    if include_laserskews:
        mu_mat = np.multiply(mu_mat,laserskews[:,None])
    #test_data = dataN*(mu_mat + uncertainty)/scale_factor
    test_data = (mu_mat + uncertainty)/runN
    return pd.DataFrame(test_data)


def log_prior(theta, priors=None):
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
            #print('catching uniform')
            if (param <= prior[1]) or (param >= prior[2]):
                #print('catching param outside of prior')
                return -np.inf
                #print('I dont terminate')
            else:
                logprior_arr.append(0)
        else:
            print('>>> Unknown prior specified')
            
    return np.sum(logprior_arr)


def log_posterior(theta, mN1_data, time_min, time_max, fromcsv, dataN, runN, scale_factor=100*100, priors=None):
    """
    The log posterior for the Raman-Rabi model, using the unbinned log-likelihood.

    Parameters:
        theta: the model parameters to use in this log-posterior calculation (array)
        mN1_data: fluoresence data for each time step (RRDataContainer)
        time_min: minimum Raman-Rabi pulse time (float)
        time_max: maximum Raman-Rabi pulse time (float)
        dataN: number of experiment repititions summed (int)
        scale_factor (optional): nuclear spin signal multiplier (float)
        priors (optional): an array specifying the priors to use in log_prior

    Returns:
        log-posterior: the value of the log-posterior for the data given the parameters in theta
    """

    logprior = log_prior(theta,priors)
    #print(logprior)
    if np.isnan(logprior) or not np.isfinite(logprior):
        return -np.inf
    loglikelihood = general_loglikelihood(theta, mN1_data, time_min, 
                                                            time_max, fromcsv, dataN, runN,
                                                            scale_factor=100*100)
    if np.isnan(loglikelihood) or not np.isfinite(loglikelihood) or np.isnan(logprior) or not np.isfinite(logprior) or np.isnan(logprior+loglikelihood):
        return -np.inf
    else:
        return logprior + loglikelihood

def laserskew_log_posterior(theta, mN1_data, time_min, time_max, fromcsv, dataN, runN, scale_factor=100*100, priors=None):
    """
    The log posterior for the Raman-Rabi model, using the unbinned log-likelihood WITH LASER SKEW PARAMETER a.

    Parameters:
        theta: the model parameters to use in this log-posterior calculation (array)
        mN1_data: fluoresence data for each time step (RRDataContainer)
        time_min: minimum Raman-Rabi pulse time (float)
        time_max: maximum Raman-Rabi pulse time (float)
        dataN: number of experiment repititions summed (int)
        scale_factor (optional): nuclear spin signal multiplier (float)
        priors (optional): an array specifying the priors to use in log_prior

    Returns:
        log-posterior: the value of the log-posterior for the data given the parameters in theta
    """
    #if ((theta[2] < 0.0) or (theta[5] < 0.0) or (np.any(theta[6:len(theta)] < 0.0)) or 
    #        (np.any(theta[6:len(theta)] > 1.01)) or (theta[0] < 0.0) or 
    #        (theta[1] < 0) or (theta[3] < 0)): #we do 1.01 because we start one at 1.0 and it gets confused
    #    return -np.inf
    #else:
    #    loglikelihood = laserskew_unbinned_loglikelihood_mN1(theta, mN1_data, time_min, time_max, fromcsv, dataN, scale_factor=100*100)
    #    logprior = log_prior(theta,priors)
    #    if np.isnan(loglikelihood) or not np.isfinite(loglikelihood) or np.isnan(logprior) or not np.isfinite(logprior):
    #        return -np.inf
# general_loglikelihood(theta, mN1_data, time_min, time_max, fromcsv, dataN, scale_factor, withlaserskew = True)
    logprior = log_prior(theta,priors)
    if np.isnan(logprior) or not np.isfinite(logprior):
        #print('succesful inf')
        return -np.inf
    else:
        loglikelihood = general_loglikelihood(theta, mN1_data, time_min, time_max, 
            fromcsv, dataN, runN, scale_factor=scale_factor, withlaserskew=True)
        if np.isnan(loglikelihood) or not np.isfinite(loglikelihood) or np.isnan(logprior) or not np.isfinite(logprior):  #or np.isnan(logprior+loglikelihood):
            #print(theta)
            #print(loglikelihood)
            #print('I fail')
            #print('')
            #print(loglikelihood)
            return -np.inf
        else:
            #print('I pass')
            #print('')
            #print(theta)
            #print(logprior + loglikelihood)
            #print('')
            return logprior + loglikelihood

def Walkers_Sampler(mN1_data, guesses, time_min, time_max, fromcsv, dataN, runN, gaus_var, nwalkers, nsteps, scale_factor=100*100, withlaserskew = False, priors=None):
    """
    This function samples the posterior using MCMC. It is recommended to use 1e-4 for gaus_var when withlaserskew=False,
    and 1e-3 for gaus_var when withlaserskew=True.

    Parameters:
        mN1_data: fluoresence data for each time step (RRDataContainer)
        guesses: the initial guesses for the parameters of the model (array of floats)
        time_min: minimum Raman-Rabi pulse time (float)
        time_max: maximum Raman-Rabi pulse time (float)
        fromcsv: marks whether these data were read from a CSV file (bool)
        dataN: number of experiment repititions summed (int)
        gaus_var: variance of the gaussian that defines the starting positions (float)
        nwalkers: the number of walkers with which to sample (int)
        nsteps: the number of steps each walker should take (int)
        withlaserskew (optional): marks whether to use laserskew functions or not (bool) 
        priors (optional): an array specifying the priors to use in log_prior

    Returns:
       sampler: the sampler object which now contains the samples taken by nwalkers
           walkers over nsteps steps
    """
    ndim = len(guesses)
    starting_positions = [guesses + gaus_var*np.random.randn(ndim) for i in range(nwalkers)]
    if withlaserskew == False:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, 
                                args=[mN1_data, time_min, time_max, fromcsv, dataN, runN],
                                kwargs={'priors':priors})
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, laserskew_log_posterior, 
                                args=[mN1_data, time_min, time_max, fromcsv, dataN, runN],
                                kwargs={'priors':priors})
        
    sampler.run_mcmc(starting_positions, nsteps)
    return sampler

    
def sampler_to_dataframe(sampler, withlaserskew = False):
    """
    This function transforms sampler into pandas dataframe

    Parameters:
        sampler: the object which contains the samples (emcee sampler)
        withlaserskew (optional): marks whether to use laserskew functions or not (bool)

    Returns:
       df: the pandas data frame object which now contains the samples taken by nwalkers
           walkers over nsteps steps
    """
    panel = pd.Panel(sampler.chain).transpose(2,0,1) # transpose permutes indices of the panel
    df = panel.to_frame() # transform panel to dataframe
    df.index.rename(['chain', 'step'], inplace=True)
    if withlaserskew == False:   
        df.columns = ['BG','Ap', 'Gp' , 'Ah', 'Oh', 'Gd']
    else:
        df.columns = ['BG','Ap', 'Gp' , 'Ah', 'Oh', 'Gd', 'Skew']
        
    return df

def plot_params_burnin(sampler, nwalkers, withlaserskew = False):
    """
    This function plots the parameters burnin time

    Parameters:
        sampler: the object which contains the samples (emcee sampler)
        nwalkers: the number of walkers with which to sample (int)
        withlaserskew (optional): marks whether to use laserskew functions or not (bool)

    """
    df = sampler_to_dataframe(sampler, withlaserskew)
    plt.figure()
    plt.plot(np.array([df['BG'].loc[i] for i in range(nwalkers)]).T)
    plt.title(r'Burn in for $B_G$')
    plt.show()

    plt.figure()
    plt.plot(np.array([df['Ap'].loc[i] for i in range(nwalkers)]).T)
    plt.title(r'Burn in for $A_p$')
    plt.show()

    plt.figure()
    plt.plot(np.array([df['Gp'].loc[i] for i in range(nwalkers)]).T)
    plt.title(r'Burn in for $\Gamma_p$')
    plt.show()

    plt.figure()
    plt.plot(np.array([df['Ah'].loc[i] for i in range(nwalkers)]).T)
    plt.title(r'Burn in for $A_h$')
    plt.show()

    plt.figure()
    plt.plot(np.array([df['Oh'].loc[i] for i in range(nwalkers)]).T)
    plt.title(r'Burn in for $\Omega_h$')
    plt.show()
    
    plt.figure()
    plt.plot(np.array([df['Gd'].loc[i] for i in range(nwalkers)]).T)
    plt.title(r'Burn in for $\Gamma_{deph}$')
    plt.show()
    
    if withlaserskew == True:
        plt.figure()
        plt.plot(np.array([df['Skew'].loc[i] for i in range(nwalkers)]).T)
        plt.title(r'Burn in for $Skew$')
        plt.show()
        
def get_burnin_data(sampler, burn_in_time = 200, withlaserskew = False):
    """
    This function trims the data according to the burn in time.

    Parameters:
        sampler: the object which contains the samples (emcee sampler)
        burn_in_time (optional): the number of samples to trim (int)
        withlaserskew (optional): marks whether to use laserskew functions or not (bool)
        
    Returns:
       parameter_samples: the pandas data frame object reshaped

    """
    samples = sampler.chain[:,burn_in_time:,:]
    # reshape the samples into a 1D array where the colums are
    # BG, Ap, Gammap, Ah, Omegah, Gammadeph
    if withlaserskew:
        numdim = 7
        traces = samples.reshape(-1, numdim).T
        # create a pandas DataFrame with labels
        parameter_samples = pd.DataFrame({'BG': traces[0], 'Ap': traces[1],
                        'Gp': traces[2], 'Ah': traces[3],
                        'Oh': traces[4], 'Gd':traces[5],
                            'Skew': traces[6]})
    else: 
        numdim = 6
        traces = samples.reshape(-1, numdim).T
        # create a pandas DataFrame with labels
        parameter_samples = pd.DataFrame({'BG': traces[0], 'Ap': traces[1],
                        'Gp': traces[2], 'Ah': traces[3],
                        'Oh': traces[4], 'Gd':traces[5]})
    
    return parameter_samples
    
