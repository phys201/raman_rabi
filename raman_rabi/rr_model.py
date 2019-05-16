from raman_rabi import RRDataContainer
import numpy as np
from numpy import random
import pandas as pd
import emcee
import seaborn as sns
import matplotlib.pyplot as plt

"""
This submodule provides the generative model (Model 1) for the Raman-Rabi data as well as
an MCMC sampler complete with log-prior and log-posterior functions for Model 1.

It also provides the alternative model (two kinds of incoherent decay) for the same data. It goes on to
provide parallel tempered sampling such that the logarithm of the global likelihood can breadily be
obtained for model comparison.

Note: theta is a list of the input parameters for the model in the following form
    'B_G','A_p', 'Gamma_p' , 'A_h', 'Omega_h', 'Gamma_deph','Skew'
"""

def ideal_model(steps, time_min, time_max, theta):
    """
    The generative model for the Raman-Rabi data, before adding noise. Gaussian noise
    is added by generate_test_data, below.

    Parameters:
        steps: the number of time divisions to use in the simulated data (int)
        time_min: minimum Raman-Rabi pulse time (float)
        time_max: maximum Raman-Rabi pulse time (float)
        theta: a dict containing the six parameters BG, Ap, Gp, Ah, Oh, Gd (dict)

    Returns:
        time: the time points at which the simulated readouts were taken (array of floats)
        mu: the values of the readout at each time point (array of floats)
    """
    BG, Ap, Gammap, Ah, Omegah, Gammadeph = theta[0:6]
    time = np.linspace(time_min, time_max, steps)
    mu = BG + Ap*np.exp(-Gammap*time) + Ah*np.cos(Omegah*time)*np.exp(-Gammadeph*time)
    return time, mu

def decay_model(steps, time_min, time_max, theta):
    """
    The generative model for the Raman-Rabi data, before adding noise. Gaussian noise
    is added by generate_test_data, below.

    Parameters:
        steps: the number of time divisions to use in the simulated data (int)
        time_min: minimum Raman-Rabi pulse time (float)
        time_max: maximum Raman-Rabi pulse time (float)
        theta: array of parameters, in the following order:
            BG: background fluoresence parameter (float)
            Ap1: parasitic loss strength parameter (float)
            Gammap1: parasitic loss time-scale parameter (float)
            Ap2: parasitic loss strength parameter (float)
            Gammap2: parasitic loss time-scale parameter (float)

    Returns:
        time: the time points at which the simulated readouts were taken (array of floats)
        mu: the values of the readout at each time point (array of floats)
    """
    BG, Ap1, Gammap1, Ap2, Gammap2 = theta[0:5]
    time = np.linspace(time_min, time_max, steps)
    mu = BG + Ap1*np.exp(-Gammap1*time) + Ap2*np.exp(-Gammap2*time)
    return time, mu

def likelihood_mN1(mN1_data, time_min, time_max, theta, dataN, runN, scale_factor=100*100):

    """
    The likelihood function of mN1 spin Raman-Rabi electronic-nuclear flip-flop data, assuming only shot noise from
    continuous fluoresence spectrum (Poisson distribution becomes Gaussian

    Parameters:
        mN1_data: fluoresence data for each time step (RRDataContainer)
        time_min: minimum Raman-Rabi pulse time (float)
        time_max: maximum Raman-Rabi pulse time (float)
        theta: the model parameters to use in this log-posterior calculation (array)
        dataN: number of experiment repititions summed (int)
        runN: number of runs over which the experiment was done (int)
        scale_factor (optional): nuclear spin signal multiplier (float)

    Returns:
        likelihood: the value of the likelihood function with these parameters (float)
        mu: the values of the model given these parameters (array of floats)
    """
    BG, Ap, Gammap, Ah, Omegah, Gammadeph = theta[0:6]
    mN1_size = mN1_data.get_df().shape[0]
    BG = BG*runN*dataN*mN1_size/scale_factor
    Ap = Ap*runN*dataN*mN1_size/scale_factor
    Ah = Ah*runN*dataN*mN1_size/scale_factor
    newtheta = [BG, Ap, Gammap, Ah, Omegah, Gammadeph]
    mN1_data = runN*np.sum(mN1_data.get_df())
    time, mu = ideal_model(len(mN1_data), time_min, time_max, newtheta)

    #we make data into unit normal distribution, uncertainty sigma of shot noise = sqrt(mu)
    z_data = (mN1_data - mu)/np.sqrt(mu)
    likelihood = (2*np.pi)**(-len(mN1_data)/2) * np.exp(-np.sum(z_data**2)/2.)
    return likelihood, mu*scale_factor/(runN*dataN*mN1_size)

def general_loglikelihood(theta, mN1_data, time_min, time_max, fromcsv, dataN, runN, scale_factor, withlaserskew = False):
    """
    This function calculates the log of the Bayesian likelihood function

    Parameters:
        theta: the model parameters to use in this log-posterior calculation (array)
        mN1_data: fluoresence data for each time step (RRDataContainer)
        time_min: minimum Raman-Rabi pulse time (float)
        time_max: maximum Raman-Rabi pulse time (float)
        fromcsv: marks whether these data were read from a CSV file (bool)
        dataN: number of experiment repititions summed (int)
        runN: number of runs over which the experiment was done (int)
        scale_factor (optional): nuclear spin signal multiplier (float)
        withlaserskew (optional): does theta include laser skew parameters? (boolean)

    Returns:
        test_data: a pandas DataFrame containing the test data (DataFrame)
    """
    BG, Ap, Gammap, Ah, Omegah, Gammadeph = theta[0:6]
    BG = BG*runN*dataN/scale_factor
    Ap = Ap*runN*dataN/scale_factor
    Ah = Ah*runN*dataN/scale_factor
    newtheta = [BG, Ap, Gammap, Ah, Omegah, Gammadeph]
    if withlaserskew:
        a_vec = np.array(theta[6:len(theta)])
        a_vec.shape = (len(a_vec), 1)

    if fromcsv:
        mN1_data = mN1_data.get_df().values
    else:
        mN1_data = mN1_data.values
    mN1_data = runN*mN1_data
    time, mu = ideal_model(mN1_data.shape[1], time_min, time_max, newtheta)
    mu_mat = np.tile(mu, (mN1_data.shape[0], 1))
    if withlaserskew:
        mu_mat = np.multiply(mu_mat, a_vec)
    z_data = (mN1_data - mu_mat)/np.sqrt(mu_mat)
    loglikelihood = np.log( (2*np.pi)**(-len(mN1_data)/2) ) - np.sum(np.log(np.sqrt(mu_mat))) - np.sum(z_data**2)/2.
    return loglikelihood

def decay_loglikelihood(theta, mN1_data, time_min, time_max, fromcsv, dataN, runN, scale_factor, withlaserskew = False):
    """
    This function calculates the log of the Bayesian likelihood function

    Parameters:
        theta: the model parameters to use in this log-posterior calculation (array)
        mN1_data: fluoresence data for each time step (RRDataContainer)
        time_min: minimum Raman-Rabi pulse time (float)
        time_max: maximum Raman-Rabi pulse time (float)
        fromcsv: marks whether these data were read from a CSV file (bool)
        dataN: number of experiment repititions summed (int)
        runN: number of runs over which the experiment was done (int)
        scale_factor (optional): nuclear spin signal multiplier (float)
        withlaserskew (optional): does theta include laser skew parameters? (boolean)

    Returns:
        test_data: a pandas DataFrame containing the test data (DataFrame)
    """
    BG, Ap1, Gammap1, Ap2, Gammap2 = theta[0:5]
    BG = BG*runN*dataN/scale_factor
    Ap1 = Ap1*runN*dataN/scale_factor
    Ap2 = Ap2*runN*dataN/scale_factor
    newtheta = [BG,Ap1,Gammap1,Ap2,Gammap2]
    if withlaserskew:
        a_vec = np.array(theta[5:len(theta)])
        a_vec.shape = (len(a_vec), 1)

    if fromcsv:
        mN1_data = mN1_data.get_df().values
    else:
        mN1_data = mN1_data.values
    mN1_data = runN*mN1_data
    time, mu = decay_model(mN1_data.shape[1], time_min, time_max, newtheta)
    mu_mat = np.tile(mu, (mN1_data.shape[0], 1))
    if withlaserskew:
        mu_mat = np.multiply(mu_mat, a_vec)
    z_data = (mN1_data - mu_mat)/np.sqrt(mu_mat)
    loglikelihood = np.log( (2*np.pi)**(-len(mN1_data)/2) ) - np.sum(np.log(np.sqrt(mu_mat))) - np.sum(z_data**2)/2.
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
        runN: number of runs over which the experiment was done (int)
        scale_factor (optional): nuclear spin signal multiplier (float)
        include_laserskews (optional): does theta include laser skew parameters? (boolean)

    Returns:
        test_data: a pandas DataFrame containing the test data (DataFrame)
    """
    BG, Ap, Gammap, Ah, Omegah, Gammadeph = theta[0:6]
    BG = BG*runN*dataN/scale_factor
    Ap = Ap*runN*dataN/scale_factor
    Ah = Ah*runN*dataN/scale_factor
    newtheta = [BG,Ap,Gammap,Ah,Omegah,Gammadeph]
    if include_laserskews:
        laserskews = theta[6:]
    time, mu = ideal_model(timesteps, time_min, time_max, newtheta)
    uncertainty = np.random.normal(scale=np.sqrt(mu), size=(samples, timesteps))

    mu_mat = np.tile(mu, (samples, 1))
    if include_laserskews:
        mu_mat = np.multiply(mu_mat,laserskews[:,None])
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
            if (param <= prior[1]) or (param >= prior[2]):
                return -np.inf
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
        runN: number of runs over which the experiment was done (int)
        scale_factor (optional): nuclear spin signal multiplier (float)
        priors (optional): an array specifying the priors to use in log_prior

    Returns:
        log-posterior: the value of the log-posterior for the data given the parameters in theta
    """

    logprior = log_prior(theta,priors)
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
        runN: number of runs over which the experiment was done (int)
        scale_factor (optional): nuclear spin signal multiplier (float)
        priors (optional): an array specifying the priors to use in log_prior

    Returns:
        log-posterior: the value of the log-posterior for the data given the parameters in theta
    """
    logprior = log_prior(theta,priors)
    if np.isnan(logprior) or not np.isfinite(logprior):
        return -np.inf
    else:
        loglikelihood = general_loglikelihood(theta, mN1_data, time_min, time_max, 
                fromcsv, dataN, runN, scale_factor=scale_factor, withlaserskew=True)
        if np.isnan(loglikelihood) or not np.isfinite(loglikelihood) or np.isnan(logprior) or not np.isfinite(logprior):
            return -np.inf
        else:
            return logprior + loglikelihood

def walkers_sampler(mN1_data, guesses, time_min, time_max, fromcsv, dataN, runN, gaus_var, nwalkers, nsteps, scale_factor=100*100, withlaserskew=False, priors=None):
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
        runN: number of runs over which the experiment was done (int)
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
    #print(starting_positions)
    if withlaserskew == False:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, 
                args=[mN1_data, time_min, time_max, fromcsv, dataN, runN],
                kwargs={'priors':priors})
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, laserskew_log_posterior, 
                args=[mN1_data, time_min, time_max, fromcsv, dataN, runN],
                kwargs={'priors':priors})

    sampler.run_mcmc(starting_positions, nsteps)
    print('sampler:',sampler)
    print('sampler chain:',sampler.chain)
    print('sampler flatchain:',sampler.flatchain)
    return sampler

def walkers_parallel_tempered(mN1_data, guesses, time_min, time_max, fromcsv, dataN, runN, gaus_var, nwalkers, nsteps, prior, scale_factor=100*100, withlaserskew = False):
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
        runN: number of runs over which the experiment was done (int)
        gaus_var: variance of the gaussian that defines the starting positions (float)
        nwalkers: the number of walkers with which to sample (int)
        nsteps: the number of steps each walker should take (int)
        priors: an array specifying the priors to use in log_prior
        withlaserskew (optional): marks whether to use laserskew functions or not (bool) 

    Returns:
       sampler: the sampler object which now contains the samples taken by nwalkers
           walkers over nsteps steps
    """
    ndim = len(guesses)
    # use temperature ladder specified in Gregory (see p. 330)
    betas = np.array([1.0, 0.7525, 0.505, 0.2575, 0.01])
    ntemps = len(betas)
    sampler = emcee.PTSampler(ntemps, nwalkers, ndim, general_loglikelihood, log_prior, 
            betas = betas, 
            loglargs=[mN1_data, time_min, time_max, fromcsv, dataN, runN, scale_factor], loglkwargs={'withlaserskew': withlaserskew}, logpkwargs={'priors': prior})
    starting_positions = np.tile(guesses, (ntemps,nwalkers,1)) + 1e-4*np.random.randn(ntemps, nwalkers, ndim)

    sampler.run_mcmc(starting_positions, nsteps)
    return sampler

def walkers_parallel_tempered_decay(mN1_data, guesses, time_min, time_max, fromcsv, dataN, runN, gaus_var, nwalkers, nsteps, prior, scale_factor=100*100, withlaserskew = False):
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
        runN: number of runs over which the experiment was done (int)
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
    # use temperature ladder specified in Gregory (see p. 330)
    betas = np.array([1.0, 0.7525, 0.505, 0.2575, 0.01])
    ntemps = len(betas)
    sampler = emcee.PTSampler(ntemps, nwalkers, ndim, decay_loglikelihood, log_prior, 
            betas = betas, 
            loglargs=[mN1_data, time_min, time_max, fromcsv, dataN, runN, scale_factor], loglkwargs={'withlaserskew': withlaserskew}, logpkwargs={'priors': prior})
    starting_positions = np.tile(guesses, (ntemps,nwalkers,1)) + 1e-4*np.random.randn(ntemps, nwalkers, ndim)

    sampler.run_mcmc(starting_positions, nsteps)
    return sampler

def sampler_to_dataframe(sampler, withlaserskew = False, burn_in_time = 0):
    """
    This function transforms sampler into pandas dataframe

    Parameters:
        sampler: the object which contains the samples (emcee sampler)
        withlaserskew (optional): marks whether to use laserskew functions or not (bool)
        burn_in_time (optional): the number of samples it took to burn in (int)

    Returns:(if withlaserskew = True)
       df: the pandas data frame object which now contains the samples taken by nwalkers
           walkers over nsteps steps
    Returns:(if withlaserskew = False)
       df, laserskew_samples: the pandas data frame object with laserskew
    """
    panel = pd.Panel(sampler.chain[:,burn_in_time:,:]).transpose(2,0,1) # transpose permutes indices of the panel
    df = panel.to_frame() # transform panel to dataframe
    df.index.rename(['chain', 'step'], inplace=True)
    if withlaserskew == False: 
        df.columns = ['BG','Ap', 'Gp' , 'Ah', 'Oh', 'Gd']
        return df
    else:
        df_params = df.iloc[:,0:6]
        df_params.columns = ['BG','Ap', 'Gp' , 'Ah', 'Oh', 'Gd']
        df_skew = df.iloc[:, 6:]
        return df_params, df_skew

def plot_params_burnin(sampler, nwalkers, withlaserskew = False):
    """
    This function plots the positions of walkers to help determine burn-in
    time.

    Parameters:
        sampler: the object which contains the samples (emcee sampler)
        nwalkers: the number of walkers with which to sample (int)
        withlaserskew (optional): marks whether to use laserskew functions or not (bool)

    """

    if withlaserskew == True:
        df, laserskew_df = sampler_to_dataframe(sampler, withlaserskew)
    else:
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
        plt.plot(np.array([laserskew_df.iloc[:,0][i] for i in range(nwalkers)]).T)
        plt.title(r'Burn in for $Skew$ for $A_1$')
        plt.show()

        plt.figure()
        plt.plot(np.array([laserskew_df.iloc[:,1][i] for i in range(nwalkers)]).T)
        plt.title(r'Burn in for $Skew$ for $A_2$')
        plt.show()

        plt.figure()
        plt.plot(np.array([laserskew_df.iloc[:,2][i] for i in range(nwalkers)]).T)
        plt.title(r'Burn in for $Skew$ for $A_3$')
        plt.show()

        plt.figure()
        plt.plot(np.array([laserskew_df.iloc[:,3][i] for i in range(nwalkers)]).T)
        plt.title(r'Burn in for $Skew$ for $A_4$')
        plt.show()

        plt.figure()
        plt.plot(np.array([laserskew_df.iloc[:,4][i] for i in range(nwalkers)]).T)
        plt.title(r'Burn in for $Skew$ for $A_5$')
        plt.show()

def pairplot_oscillation_params(samples,filename=None):
    """
    Produce a pairplot of the three oscillation parameters of interest
    (Omega_h, A_h, and Gamma_deph)

    Parameters:
        samples: a dataframe produced from an emcee sampler using 
            rr_model.sampler_to_dataframe (dataframe)
        filename (optional): the filename to which to save the plot produced

    Returns:
        plot: a pyplot plot shown on the screen
        file: a file with the name <filename> containing the plot
    """
    sns.set(font_scale=2)
    threevar_pairplot = sns.pairplot(samples, 
            x_vars=['Ah', 'Oh', 'Gd'], 
            y_vars=['Ah', 'Oh', 'Gd'],
            height=5, markers='.')
    threevar_pairplot = threevar_pairplot.map_offdiag(plt.scatter, s=5, alpha=0.3)
    threevar_pairplot = threevar_pairplot.map_diag(plt.hist,
            histtype='stepfilled',bins=10)
    plt.show()
    if filename:
        plt.savefig(filename)

def plot_fit_and_data(mapvals,burned_in_samples,data,steps,time_min,time_max,dataN,scale_factor=100*100,fromcsv=False):
    """
    Plot the theoretical prediction with MAP fit parameters along with un-averaged
    data

    Parameters:
        mapvals: the MAP model parameters from an MCMC calculation (array of floats)
        burned_in_samples: the MCMC sample chain from which the MAP values were computed (array of floats)
        data: the real data to plot against the prediction (RRDataContainer or Pandas DataFrame)
        steps: the number of time steps in each run in the data (int)
        time_min: the first time bin in microseconds (float)
        time_max: the last time bin in microseconds (float)
        dataN: the number of data points over which the data were averaged (int)
        scale_factor: the conversion between fluorescence readout and number of nuclei (float)
        fromcsv: boolean indicating whether the data was read in from a file

    Returns:
        plot: a pyplot plot shown on the screen
    """
    numdim = burned_in_samples.shape[2]
    traces = burned_in_samples.reshape(-1,numdim).T
    #BG_MAP, Ap_MAP, Gammap_MAP, Ah_MAP, Omegah_MAP, Gammadeph_MAP = mapvals
    time, mu = ideal_model(steps, time_min, time_max, 
        mapvals)
    laserindex = np.argmin(np.abs(traces[0, :] - np.percentile(traces[0, :], 50)))
    laserskewave = np.average(traces[6:, laserindex])
    mu = mu*laserskewave #MOST IMPORTANT PART!!!!

    #Plot over unaveraged data
    plt.figure()
    if fromcsv:
        data_length = len(data.get_df())
        for iii in range(data_length):
            if iii == 0:
                plt.scatter(time, data.get_df().values[iii, :]*scale_factor/dataN, color='C0', label='Raw Data')
            else:
                plt.scatter(time, data.get_df().values[iii, :]*scale_factor/dataN, color='C0')
    else:
        data_length = len(data.values)
        for iii in range(data_length):
            if iii == 0:
                plt.scatter(time, data.values[iii, :]*scale_factor/dataN, color='C0', label='Raw Data')
            else:
                plt.scatter(time, data.values[iii, :]*scale_factor/dataN, color='C0')
    plt.plot(time, mu, color='r', label='MCMC')
    plt.legend()
    plt.xlabel('Time [$\mu$s]', fontsize=15)
    plt.ylabel('Fluorescence [A.U.]', fontsize=15)
    #plt.savefig('mN1_rawdata.png', bbox_inches = 'tight')
    plt.show()
