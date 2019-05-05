import raman_rabi
from raman_rabi import rr_model
import numpy as np

# previously estimated parameters:
data_length = 14
# set random seed for reproducibility
np.random.seed(0)
skew_values = np.random.rand(data_length)
theta = np.concatenate( (np.array([6.10, 16.6881,
				1/63.8806, 5.01886,
				-np.pi/8.77273, 1/8.5871]),
			    skew_values), axis=0)

# generate some data
test_data = rr_model.generate_test_data(theta, 161,
					data_length, 0, 40,
					include_laserskews=True)

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
print("MAP['Gammap'] is",MAP['Gammap'].values[0])
print("guesses[2] is",guesses[2])
print("% difference is",abs((MAP['Gammap'].values[0]-guesses[2])/guesses[2]))
print("MAP['Ah'] is",MAP['Ah'].values[0])
print("guesses[3] is",guesses[3])
print("% difference is",abs((MAP['Ah'].values[0]-guesses[3])/guesses[3]))
print("MAP['Omegah'] is",MAP['Omegah'].values[0])
print("guesses[4] is",guesses[4])
print("% difference is",abs((MAP['Omegah'].values[0]-guesses[4])/guesses[4]))
print("MAP['Gammadeph'] is",MAP['Gammadeph'].values[0])
print("guesses[5] is",guesses[5])
print("% difference is",abs((MAP['Gammadeph'].values[0]-guesses[5])/guesses[5]))
laserskew_columns = list(laserskew_MAP)
for column in laserskew_columns:
    print("laserskew_MAP",column,"is",laserskew_MAP[column].values[0])
    print("guesses[6+",column,"] is",guesses[6+column])
    print("% difference is",abs((laserskew_MAP[column].values[0]-guesses[6+column])/guesses[6+column]))
