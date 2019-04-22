import raman_rabi
from raman_rabi import testing
from raman_rabi import RRDataContainer
from raman_rabi import rr_io
from raman_rabi import rr_model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

## Step 1
#first we load a data file and test it on our new loglikelihood function
testfilename = "21.07.56_Pulse Experiment_E1 Raman Pol p0 Rabi Rep Readout - -800 MHz, ND0.7 Vel1, Presel 6 (APD2, Win 3)_ID10708_image.txt"
RRData = rr_io.load_data(rr_io.get_example_data_file_path(testfilename))

#theta are the parameters previously estimated (in an ad-hoc) manner in the paper
theta = np.array([6.10, 16.6881, 1/63.8806, 5.01886, -np.pi/8.77273, 1/8.5871])
s_loglikelihood = rr_model.unbinned_loglikelihood_mN1(theta, RRData, 0, 40, True)
print(s_loglikelihood) #this function runs our data, check mark.


## Step 2
# now we generate test data with our new generate data function. we know that the
# parameters we use to generate this data should be those that our MCMC hands us back.
# We use theta from above as our parameters.
# ALSO, here I use 500 data points of computationally generated test data for each time step instead of our usual 20. This is so that our data indeed
# converges nicely to the distribution we expect in all random draws. It's too bad that the real data set doesn't have this many repititions.....
# This is probably overkill to a degree
test_data = rr_model.generate_test_data(theta, 161, 500, 0, 40)
test_loglikelihood = rr_model.unbinned_loglikelihood_mN1(theta, test_data, 0, 40, False) #this function runs our generated test data, check mark.
print(test_loglikelihood)

## Step 3
#now we will run our new Walkers function to run MCMC on the data. We expect to have MCMC return the values of our parameters we gave in
#theta back to us.
guesses = theta
numdim = len(guesses)
numwalkers = 400 #as our computationally generated data is pretty big and this is a decent amount of walkers and steps, it takes about 3 whole minutes
#to run on my machine, and my machine is a beast. so we many want to turn it down a little in case Solomon gets tired of waiting or it crashes him (unlikely, but still)
numsteps = 1000
test_samples = rr_model.Walkers(test_data, guesses, 0, 40, False, dataN=10, scale_factor=100*100, nwalkers=numwalkers, nsteps=numsteps)
burn_in_time = 250
samples = test_samples.chain[:,burn_in_time:,:]
traces = samples.reshape(-1, numdim).T
parameter_samples = pd.DataFrame({'BG': traces[0], 'Ap': traces[1], 'Gammap': traces[2], 'Ah': traces[3], 'Omegah': traces[4], 'Gammadeph': traces[5]})
MAP = parameter_samples.quantile([0.50], axis=0)

#guesses and MAP order = [BG, Ap, Gammap, Ah, Omegah, Gammadeph]
#we print one above the other and see how similar they are
print(guesses)
print(MAP)

#make pretty plots of this shit for any two parameters if you like
#joint_kde = sns.jointplot(x='Ah', y='Omegah', data=parameter_samples, kind='kde')
#plt.show()
