from raman_rabi import RRDataContainer
import numpy as np

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
    """
    mN1_size = mN1_data.get_df().shape[0]
    mN1_data = scale_factor*np.sum(mN1_data.get_df()) / (dataN*mN1_size)

    time = np.linspace(time_min, time_max, len(mN1_data))
    mu = BG + Ap*np.exp(-Gammap*time) + Ah*np.cos(Omegah*time)*np.exp(-Gammadeph*time)

    #we make data into unit normal distribution, uncertainty sigma of shot noise = sqrt(mu)
    z_data = (mN1_data - mu)/np.sqrt(mu)
    likelihood = (2*np.pi)**(-len(mN1_data)/2) * np.exp(-np.sum(z_data**2)/2.)
    return likelihood,mu
