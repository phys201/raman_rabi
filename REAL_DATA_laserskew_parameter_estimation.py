##Todo:
#debug whatever your weird errors are
#Get relative strength estimates of each run so that you can assign good initial guesses, right now the burn in time is too small because
#you start by assuming a_i = 1

import raman_rabi
from raman_rabi import rr_model
from raman_rabi import rr_io
from raman_rabi import RRDataContainer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_length = 20

skew_values = np.ones(data_length)
#print(skew_values)

params = np.array([6.22622623e+00, 1.68718719e+01, 1.52076134e-02, 5.09509510e+00, -3.54854855e-01, 1.21121121e-01])
theta = np.concatenate( (params, skew_values), axis=0)

#import mN=+1 data
testfilepath = rr_io.get_example_data_file_path("21.07.56_Pulse Experiment_E1 Raman Pol p0 Rabi Rep Readout - -800 MHz, ND0.7 Vel1, Presel 6 (APD2, Win 3)_ID10708_image.txt")
mN1_data = RRDataContainer(testfilepath) 

# run MCMC on the test data and see if it's pretty close to the original theta
guesses = theta
numdim = len(guesses)
numwalkers = 500
numsteps = 1000
np.random.seed(0)
test_samples = rr_model.laserskew_Walkers(mN1_data, guesses,
					  0, 40, True, dataN=10,
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
print("MAP['Gammap'] is",MAP['Gammap'].values[0])
print("guesses[2] is",guesses[2])
#print("% difference is",abs((MAP['Gammap'].values[0]-guesses[2])/guesses[2]))
print("MAP['Ah'] is",MAP['Ah'].values[0])
print("guesses[3] is",guesses[3])
#print("% difference is",abs((MAP['Ah'].values[0]-guesses[3])/guesses[3]))
print("MAP['Omegah'] is",MAP['Omegah'].values[0])
print("guesses[4] is",guesses[4])
#print("% difference is",abs((MAP['Omegah'].values[0]-guesses[4])/guesses[4]))
print("MAP['Gammadeph'] is",MAP['Gammadeph'].values[0])
print("guesses[5] is",guesses[5])
#print("% difference is",abs((MAP['Gammadeph'].values[0]-guesses[5])/guesses[5]))
laserskew_columns = list(laserskew_MAP)
for column in laserskew_columns:
    print("laserskew_MAP",column,"is",laserskew_MAP[column].values[0])
    print("guesses[6+",column,"] is",guesses[6+column])
    #print("% difference is",abs((laserskew_MAP[column].values[0]-guesses[6+column])/guesses[6+column]))

BG_MCMC = MAP['BG'].values[0]
Ap_MCMC = MAP['Ap'].values[0]
Gammap_MCMC = MAP['Gammap'].values[0]
Ah_MCMC = MAP['Ah'].values[0]
Omegah_MCMC = MAP['Omegah'].values[0]
Gammadeph_MCMC = MAP['Gammadeph'].values[0]

time, mu = rr_model.ideal_model(161, 0, 40, BG_MCMC, Ap_MCMC, Gammap_MCMC, Ah_MCMC, Omegah_MCMC, Gammadeph_MCMC)
#print(mu)

plt.figure()
scale_factor = 100*100
N_value = 10
#plt.scatter(time, np.sum(mN1_data.get_df().values, axis=0)*scale_factor/N_value/data_length)
for iii in range(data_length):
    plt.scatter(time, mN1_data.get_df().values[iii, :]*scale_factor/N_value, color='b')
plt.plot(time, mu, color='k')
plt.show()
