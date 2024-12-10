from matplotlib import pyplot as plt 
import seaborn as sns 
import numpy as np 
from scipy.fft import fft, fftfreq, ifft 
from scipy.stats import norm, lognorm, beta
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
import os 

sns.set_theme()
    

class SimulateData:
    def __init__(self, SNR = 0.5,
                       noise_params : list[float, float, float, float] = [200, 1, 10, 1.5],
                       stim_freq : int = 10, 
                       stim_amp : int = 6000, 
                       CAP_freq : int = 50,
                       CAP_dist : str = "uniform") -> None:
        """
        Simulate data for the project
        Inputs: 
            SNR : float - signal to noise ratio
            noise_params: np.array([float, float, float, float]) - noise parameters for power line intereference, 500 Hz noise, gaussian noise and high frequency noise 
            stim_freq: float - frequency of the stimulus signal¨
            CAP_freq: int - frequency of the CAP signal
            CAP_dist: str - distribution of the CAP signal (uniform, lognormal, normal)
        """
        self.SNR = SNR 
        self.noise_params = noise_params
        
        self.CAP_dist = CAP_dist
        self.CAP_freq = CAP_freq
        self.CAP_indices = None 

        self.stim_freq = stim_freq
        self.stim_amp = stim_amp
        self.SA_indices = None
        self.num_stims = None 
        
        self.length = 300000
        self.num_channels = 32 
        self.fs = 3*1e4
        self.duration = self.length // self.fs 
        self.num_stims = int(self.duration * self.stim_freq)

        self.signal = None 
        self.true_signal = None
        self.noise_signal = None 
    
    def base_signal(self):
        """ Initialize the base signal"""

        self.noise_signal = np.zeros((self.length, self.num_channels))

    
    def add_noise_params_pli(self):
        """Add power line interference noise to the signal"""

        pli = np.load("noise_files_sim_data/pli.npy")
        for channel in range(self.num_channels):
            self.noise_signal[:, channel] += pli * self.noise_params[0]  

    def add_noise_params_500_hz(self):
        """ Add 500 Hz noise to the signal"""

        # read 500Hz noise_params
        noise_params_500 = np.load("noise_files_sim_data/500_Hz.npy")
        n = len(noise_params_500)
        num_rep = int(np.ceil(self.length / n))

        # match length of signal 
        noise_params_500 = np.array([noise_params_500 for _ in range(num_rep)]).ravel() 
        noise_params_500 = noise_params_500[:self.length]

        # normalize noise_params such that the amplitude parameter is interpretable 
        noise_params_500 /= np.max(noise_params_500)
        for channel in range(self.num_channels):
            self.noise_signal[:, channel] += noise_params_500 * self.noise_params[1]

    def add_noise_params_high_freq(self):
        """ Add high frequency noise to the signal"""

        # read high frequency noise_params
        high_freq = np.load("noise_files_sim_data/high_freq.npy")

        # normalize noise_params such that the amplitude parameter is interpretable 
        for channel in range(self.num_channels):
            self.noise_signal[:, channel] += high_freq * self.noise_params[3]

    def add_noise_params_gauss(self):
        """ Add gaussian noise to the signal"""

        # add gaussian noise_params 
        for channel in range(self.num_channels): 
            self.noise_signal[:, channel] += np.random.normal(0, self.noise_params[2], self.length)

    def add_stim_to_all_channels(self, SA_options : np.ndarray, SA_idx : int, idx : int) -> None:
        """ Add a stimulus to all channels"""

        # construct SA and add to signals
        SA = SA_options[SA_idx] * 0.8 
        for channel in range(self.num_channels):

            idx1 = idx 
            idx2 = idx + len(SA)

            # take care of edge cases
            if idx2 >= self.length: 
                idx1 = idx2 - len(SA) - 1 
                idx2 = self.length - 1
            elif idx1 < 0:
                idx1 = 0
                idx2 = len(SA)

            self.noise_signal[idx1:idx2, channel] += SA * self.stim_amp


    def add_stim_to_single_channel(self, SA_options : np.ndarray, spacing : int, SA_idx : int, channel : int, save_indices : bool = False) -> None:
        """ Add all stimuli to a single channel"""

        # construct SA and add to signals
        SA = SA_options[SA_idx] 
        SA /= np.max(SA)

        # insert every stimulation 
        for stim in range(self.num_stims):
            if save_indices:
                offset = np.random.randint(-50, 50, 1)[0]
                idx1 = spacing * stim + offset 
                idx2 = spacing * stim + offset + len(SA)
            else:
                # SA indices have already been saved 
                idx1 = self.SA_indices[stim]
                idx2 = self.SA_indices[stim] + len(SA)

            # take care of edge cases
            if idx2 >= self.length: 
                idx1 = idx2 - len(SA) - 1 
                idx2 = self.length - 1
            elif idx1 < 0:
                idx1 = 0
                idx2 = len(SA)

            # insert SA 
            self.noise_signal[idx1:idx2, channel] += SA * self.stim_amp

            # save the indices of the stimuli
            if save_indices:
                self.SA_indices[stim] = idx1
         

    def add_all_stimuli(self) -> None:
        """ Add all stimuli to the signal"""
        
        # read the stimulus signal (one stim type should be used throughout the signal but not across channels)
        SA_options = np.load("noise_files_sim_data/SA_time.npy")

        # init variables 
        p = np.ones(self.num_channels) / self.num_channels 
        self.SA_indices = np.zeros(self.num_stims, dtype = int)
        spacing = self.length // self.num_stims

        # add stimulus to a single channel
        for channel in range(self.num_channels):
            # sample a SA 
            SA_idx = np.random.choice(self.num_channels, p = p)
            
            # insert SA 
            if channel == 0:
                self.add_stim_to_single_channel(SA_options, spacing, SA_idx, channel, save_indices=True)
            else:
                self.add_stim_to_single_channel(SA_options, spacing, SA_idx, channel)
            
            # update probability
            p[SA_idx] /= 10
            p /= np.sum(p)

        # see if extra stimuli should be added
        if np.random.rand() < 0.001:
            # draw random index and add a stimulus
            idx = np.random.randint(0, self.length, 1)[0]
            SA_idx = np.random.choice(self.num_channels, p = p)
            self.SA_indices = np.sort(np.append(self.SA_indices, idx))

            # add stimulus to signal
            self.add_stim_to_all_channels(SA_options, SA_idx, idx, apply_offset=False)

    def add_spontaneous_activity(self) -> None:
        """Function to add spontaneous spikes"""

        duration = 2 
        CAP = self.get_CAP(duration)
        num_points = int(duration * 30)
        interp = interp1d(np.arange(0, len(CAP)), CAP)

        spike_rate = 5 
        spike_probability = spike_rate / self.fs 

        for channel in range(self.num_channels):
            num_samples = int(self.fs * self.duration)

            # Generate spikes: 1 for a spike, 0 otherwise
            spikes = (np.random.rand(num_samples) < spike_probability).astype(int)

            # insert spike where there is a 1
            for i in range(num_samples):
                if spikes[i] == 1:
                    idx = i 
                    idx1 = idx 
                    idx2 = idx + num_points

                    # take care of edge cases
                    if idx2 >= self.length: 
                        idx1 = idx2 - num_points - 1 
                        idx2 = self.length - 1
                    elif idx1 < 0:
                        idx1 = 0
                        idx2 = num_points
                    
                    # insert into true signal 
                    self.true_signal[idx1:idx2, channel] += interp(np.linspace(0, len(CAP)-1, num_points)) * 2 

                    # update CAP indices 
                    stim = idx1 // (self.length // self.num_stims)
                    self.CAP_indices[stim][channel] = [self.CAP_indices[stim][channel], idx1]


    def base_CAP(self) -> np.ndarray:
        """ Create the base CAP signal"""

        # create the two parts of the CAP
        y1 = norm.pdf(np.linspace(-2, 1, 80), 0, 0.5)
        y2 = beta.pdf(np.linspace(0, 1, 100), 2, 3)

        # combine the two parts and smooth the signal
        Y = np.r_[4*y1, -0.8*y2]
        Y[len(Y)//2-10:len(Y) // 2 + 10] = gaussian_filter1d(Y[len(Y)//2-10:len(Y) // 2 + 10], 4)
        Y = gaussian_filter1d(Y, 4)

        return Y 
    

    def CAP2(self, duration : float) -> np.ndarray:
        CAP2 = np.load("noise_files_sim_data/CAP2.npy")

        # match to the duration of the CAP signal
        num_points = int(duration * 30)
        interp = interp1d(np.arange(0, len(CAP2)), CAP2)
        x_new = np.linspace(0, len(CAP2)-1, num_points)
        Y = interp(x_new)

        # vary the amplitude a bit from the specified amplitude
        Y /= np.max(Y)
        scale = np.random.randint(7, 14, 1)[0] / 10
        Y = Y * scale

        return Y 

    def CAP1(self, duration : float) -> np.ndarray:
        """ Get a CAP signal with a specified duration"""

        base_cap = self.base_CAP()
        
        # match to the duration of the CAP signal
        num_points = int(duration * 30)
        interp = interp1d(np.arange(0, len(base_cap)), base_cap)
        x_new = np.linspace(0, len(base_cap)-1, num_points)
        Y = interp(x_new)

        # vary the amplitude a bit from the specified amplitude
        Y /= np.max(Y)
        scale = np.random.randint(7, 14, 1)[0] / 10  
        Y = Y * scale 
        
        return Y 
    
    def get_CAP(self, duration : float) -> np.ndarray:
        """ Get a CAP signal with a specified duration"""

        # sample a CAP signal
        CAP = self.CAP1(duration) if np.random.rand() < 0.5 else self.CAP2(duration)

        # flip signal with 50% prob 
        CAP = -CAP if np.random.rand() < 0.5 else CAP

        return CAP
    
    def sample_uniform_indices(self, stim : int, num_CAPs : int) -> np.ndarray:
        """ Sample indices uniformly"""

        if stim == self.num_stims - 1:
            indices = np.random.randint(self.SA_indices[stim], self.length, num_CAPs)
        else: 
            indices = np.random.randint(self.SA_indices[stim], self.SA_indices[stim+1], num_CAPs)
        return indices
    
    def sample_lognormal_indices(self, stim : int, num_CAPs : int) -> np.ndarray:
        """ Sample indices from a lognormal distribution"""

        # add extra to mean since SA indices refer to where they start 
        mu = 600  
        sigma = 2*1e5
        sigma_log = np.sqrt(np.log(1 + (sigma / mu**2)))
        mu_log = np.log(mu) - 0.5 * sigma_log**2
        indices = lognorm.rvs(sigma_log, scale=np.exp(mu_log), size=num_CAPs) + self.SA_indices[stim]

        return indices.astype(int)
    
    def sample_normal_indices(self, stim : int, num_CAPs : int) -> np.ndarray:
        """ Sample indices from a normal distribution"""
        mu = 600
        sigma = 500
        indices = norm.rvs(mu, sigma, size=num_CAPs) + self.SA_indices[stim]

        return indices.astype(int)
        
    def add_CAP(self):
        """ Add CAP signals to the signal"""
        
        self.true_signal = np.zeros((self.length, self.num_channels))
        self.CAP_indices = np.zeros((self.num_stims, self.num_channels), dtype = object)
        
        # get the average length of the segments between the stimuli
        segment_length = np.mean(self.SA_indices[1:] - self.SA_indices[:-1])

        # compute the number of CAP signals to add from CAP freq and segment length
        num_CAPs = int(segment_length / 30 * self.CAP_freq / 1000) + np.random.choice([-1, 0, 1], 1)[0]

        for channel in range(self.num_channels):
            for stim in range(self.num_stims):
                # compute change of there occuring a CAP signal
                p = np.random.rand() < 0.3

                if p: 
                    if self.CAP_dist == "lognormal":
                        indices = self.sample_lognormal_indices(stim, num_CAPs)
            
                    elif self.CAP_dist == "normal":
                        indices = self.sample_normal_indices(stim, num_CAPs)
            
                    elif self.CAP_dist == "uniform":
                        indices = self.sample_uniform_indices(stim, num_CAPs)

                    for cap in range(num_CAPs):
                        # sample a CAP 
                        duration = np.random.randint(3, 8) + np.random.random()
                        CAP = self.get_CAP(duration)

                        # insert cap into signal as well as the true signal 
                        if indices[cap] + len(CAP) > self.length:
                            self.true_signal[indices[cap]:, channel] += CAP[:self.length - indices[cap]]
                        else: 
                            self.true_signal[indices[cap]:indices[cap] + len(CAP), channel] += CAP
                
                        # store the indices of the CAPs
                    self.CAP_indices[stim][channel] = indices.tolist()

            
    def construct_signal(self):
        """ Construct the signal"""

        ### Noise signal ### 
        # init signal 
        self.base_signal()

        # noise params 
        self.add_noise_params_500_hz()
        self.add_noise_params_gauss()
        self.add_noise_params_pli()
        self.add_noise_params_high_freq()

        # compute power of noise signal 
        rms_noise = np.mean(self.noise_signal**2, axis=0)

        # add stimuli
        self.add_all_stimuli(); 

        ### true signal ###
        self.add_CAP()
        self.add_spontaneous_activity()

        ### combine signals ###
        self.signal = np.zeros((self.length, self.num_channels))
        for channel in range(self.num_channels):
            pow_true_signal = np.sqrt(self.SNR * rms_noise[channel])
            self.signal[:, channel] = pow_true_signal * self.true_signal[:, channel] + self.noise_signal[:, channel]
            self.true_signal[:, channel] = pow_true_signal * self.true_signal[:, channel]
        


    def time_to_freq(self, data : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Function to convert time domain data to frequency domain data
        """
        duration = len(data) / self.fs 
        N  = int(self.fs * duration)
        yf = fft(data)
        xf = fftfreq(N, 1 / self.fs)

        return xf, yf 

    def freq_to_time(self, yf : np.ndarray) -> np.ndarray:
        """
        Function to convert frequency domain data to time domain data
        """

        ifft_data = ifft(yf)
        return ifft_data

    def plot_data(self, channel : int, xlim  : tuple[float, float] = (1, 1.1), ylim : tuple[float, float] = (-300, 300)) -> None:
        """ Plot two channels if the data"""
        num_sec = self.duration 
        x_axis = np.linspace(0, num_sec, self.length)

        fig, ax = plt.subplots(2, 1, figsize = (18, 8), sharex=True, sharey = True)
        ax[0].plot(x_axis, self.signal[:, channel], color = "darkblue")
        ax[0].set_title("Time domain signal")
        ax[0].set_xlabel("Time (s)")
        ax[0].set_ylabel("Amplitude")

        ax[1].plot(x_axis, self.true_signal[:, channel], color = "darkblue")
        ax[1].set_title("Time domain signal")
        ax[1].set_xlabel("Time (s)")
        ax[1].set_ylabel("Amplitude")

        [a.set_ylim([ylim[0], ylim[1]]) for a in ax]
        [a.set_xlim([xlim[0], xlim[1]]) for a in ax]
        plt.show()

def save_data(data : np.ndarray, name : str) -> None:
    """ Save data to file"""
    np.save(name, data)

if __name__ == "__main__":
    # change direcotry to data
    os.chdir("src/data/")
    print(os.getcwd())


    simulator = SimulateData(noise_params = [200, 1, 10, 0.3], SNR = 1, stim_freq=10, stim_amp = 6000, CAP_freq = 40, CAP_dist="lognormal")
    simulator.construct_signal()

    # save data
    # save_data(simulator.signal, "../../data/simulated/10_30_lognormal.npy")
    simulator.plot_data((0, 1))  # Plot the first two channels