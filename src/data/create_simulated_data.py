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
                       CAP_dist : str = "uniform", 
                       num_channels : int = 32,
                       seed = None) -> None:
        """
        Simulate data for the project
        Inputs: 
            SNR : float - signal to noise ratio
            noise_params: np.array([float, float, float, float]) - noise parameters for power line intereference, 500 Hz noise, gaussian noise and high frequency noise 
            stim_freq: float - frequency of the stimulus signal¨
            CAP_freq: int - frequency of the CAP signal
            CAP_dist: str - distribution of the CAP signal (uniform, lognormal, normal, None)
        """
        self.seed = seed 
        if self.seed is not None: 
            np.random.seed(self.seed)

        self.SNR = SNR 
        self.noise_params = noise_params
        
        self.CAP_dist = CAP_dist
        self.CAP_indices = None 

        self.stim_freq = stim_freq
        self.stim_amp = stim_amp
        self.SA_indices = None
        self.num_stims = None 

        self.num_channels = num_channels
        self.length = 300000
        self.fs = 3*1e4
        self.duration = self.length // self.fs 
        self.num_stims = int(self.duration * self.stim_freq)
        self.channel_varier_count = np.random.uniform(0.5, 7, self.num_channels)
        self.channel_varier_occurrence = np.random.uniform(0.1, 0.9, self.num_channels)

        self.signal = None 
        self.true_signal = None
        self.noise_signal = None 
    
    def base_signal(self):
        """ Initialize the base signal"""
        self.noise_signal = np.zeros((self.length, self.num_channels))

    
    def add_noise(self, noise_file, amplitude):
        """Generic function to add noise to the signal."""
        noise = np.load(noise_file)

        for channel in range(self.num_channels):
            self.noise_signal[:, channel] += noise * amplitude

    def add_noise_params(self):
        """Add various noise components to the signal."""
        self.add_noise("noise_files_sim_data/pli.npy", self.noise_params[0])
        self.add_noise("noise_files_sim_data/500_Hz.npy", self.noise_params[1])
        self.add_noise("noise_files_sim_data/high_freq.npy", self.noise_params[3])

        for channel in range(self.num_channels):
            self.noise_signal[:, channel] += np.random.normal(0, self.noise_params[2], self.length)

    def add_stimulus(self, SA_options, channel, spacing):
        """Add stimuli to a specific channel."""
        SA_idx = np.random.choice(len(SA_options))
        SA = SA_options[SA_idx] / np.max(SA_options[SA_idx])  # Normalize

        for stim in range(self.num_stims):
            offset = np.random.randint(-20, 20)
            idx1 = spacing * stim + offset
            idx2 = idx1 + len(SA)

            if idx1 < 0 or idx2 > self.length:
                continue  # Skip out-of-bound indices

            self.noise_signal[idx1:idx2, channel] += SA * self.stim_amp

            if channel == 0:  # Save SA indices for the first channel (common for all channels)
                # find the largest value between the two indices 
                sa_idx = np.argmax(self.noise_signal[idx1:idx2, channel]) + idx1 
                self.SA_indices[stim] = sa_idx

    def add_all_stimuli(self):
        """Add all stimuli to the signal."""
        SA_options = np.load("noise_files_sim_data/SA_time.npy")

        self.SA_indices = np.zeros(self.num_stims, dtype=int)
        spacing = self.length // self.num_stims

        for channel in range(self.num_channels):
            self.add_stimulus(SA_options, channel, spacing)

    def add_spontaneous_activity(self):
        """Add spontaneous activity to the signal."""

        # init true signal if not already done
        if self.true_signal is None:
            self.true_signal = np.zeros((self.length, self.num_channels))
        if self.CAP_indices is None:
            self.CAP_indices = np.zeros((self.num_stims, self.num_channels), dtype=object)

        for channel in range(self.num_channels):
            activity_count = np.random.choice(np.arange(int(self.duration * 5)-10, int(self.duration * 5)+10))  # Approx. 5 spontaneous activities per second
            indices = np.random.randint(0, self.length, size=activity_count)

            for idx in indices:
                duration = np.random.uniform(0, 4) + np.random.random()
                spon_act = self.get_CAP(duration) * 0.5  # Spontaneous activity scaled down
                idx1 = max(0, idx)
                idx2 = min(self.length, idx1 + len(spon_act))
                SA_slice = spon_act[:idx2 - idx1]
                self.true_signal[idx1:idx2, channel] += SA_slice

                spon_act_idx = np.argmax(self.true_signal[idx1:idx2, channel]) + idx1 

                # update CAP indices 
                stim = spon_act_idx // (self.length // self.num_stims)
                if type(self.CAP_indices[stim][channel]) == list:
                    self.CAP_indices[stim][channel].append(idx1)
                else: 
                    self.CAP_indices[stim][channel] = [idx1]
                        

    def base_CAP(self) -> np.ndarray:
        """ Create the base CAP signal"""

        # create the two parts of the CAP
        y1 = norm.pdf(np.linspace(-2, 1, 80), 0, 0.5)
        y2 = beta.pdf(np.linspace(0, 1, 100), 2, 3)

        # combine the two parts and smooth the signal
        Y = np.r_[4 * y1, -0.8 * y2]
        Y[len(Y) // 2- 10 : len(Y) // 2 + 10] = gaussian_filter1d(Y[len(Y) // 2- 10 : len(Y) // 2 + 10], 4)
        Y = gaussian_filter1d(Y, 4)

        return Y 
    

    def CAP2(self, duration : float) -> np.ndarray:
        CAP2 = np.load("noise_files_sim_data/CAP2.npy")

        # match to the duration of the CAP signal
        num_points = max(int(duration * 30), 10)
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
        

    def sample_uniform_indices(self, stim, num_CAPs):
        return self.sample_with_min_spacing(
            stim, num_CAPs, distribution_func=np.random.uniform, low=0, high=2000 # latency of a few ms 
        )

    def sample_lognormal_indices(self, stim, num_CAPs):
        mu = 1200   
        sigma = 2e5
        sigma_log = np.sqrt(np.log(1 + (sigma / mu**2)))
        mu_log = np.log(mu) - 0.5 * sigma_log**2
        return self.sample_with_min_spacing(
            stim, num_CAPs, distribution_func=lognorm.rvs, s=sigma_log, scale=np.exp(mu_log)
        )

    def sample_normal_indices(self, stim, num_CAPs):
        mu = 1500  
        sigma = 70 
        return self.sample_with_min_spacing(
            stim, num_CAPs, distribution_func=norm.rvs, loc=mu, scale=sigma
        )


    def ensure_min_spacing(self, indices, min_spacing, num_CAPs):
        """ Ensure that indices have at least min_spacing points between them """
        indices = np.sort(indices)  # Sort indices for easier checking
        
        # remove negative indices 
        indices = np.delete(indices, np.where(indices < 0)[0])

        spaced_indices = [indices[0]]
        for index in indices[1:]:
            if index - spaced_indices[-1] >= min_spacing:
                spaced_indices.append(index)
            if len(spaced_indices) >= num_CAPs:  # Stop if we have collected enough
                break
        return np.array(spaced_indices, dtype=int)

    def sample_with_min_spacing(self, stim, num_CAPs, distribution_func, **dist_params):
        """ General function to sample with spacing constraint """

        oversample_factor = 2  # Adjust this to reduce bias
        total_samples = num_CAPs * oversample_factor
        runs = 0 
        while True:
            raw_indices = distribution_func(size=total_samples, **dist_params) + self.SA_indices[stim] + 300 # Add 300 to avoid overlap with SA
            valid_indices = self.ensure_min_spacing(raw_indices.astype(int), 90, num_CAPs)
            if len(valid_indices) == num_CAPs:
                return valid_indices
            runs += 1 
            
            # if five tried just return the amount it has found that fit the requirements 
            if runs == 5: 
                return valid_indices

    def add_CAP(self):
        """ Add CAP signals to the signal"""
        
        # initalize the CAPs will occur 
        self.true_signal = np.zeros((self.length, self.num_channels))
        self.CAP_indices = np.zeros((self.num_stims, self.num_channels), dtype = object)
        
        for channel in range(self.num_channels):
            for stim in range(self.num_stims):
                # compute change of there occuring a CAP signal
                if np.random.rand() > self.channel_varier_occurrence[channel]: 
                    continue 
               
                # compute the number of CAP signals to add from CAP freq and segment length
                # num_CAPs = max(int(segment_length / 3000 * self.CAP_freq) + np.random.choice([-1, 0, 1], 1)[0], 1)
                num_CAPs = max(int(np.random.choice(np.arange(1, 8), 1)[0] * self.channel_varier_count[channel]), 1)

                if self.CAP_dist == "lognormal":
                    # indices = self.sample_lognormal_indices(stim, num_CAPs)
                    indices = self.sample_with_min_spacing(stim, num_CAPs, distribution_func=lognorm.rvs, s=2, scale=600)
        
                elif self.CAP_dist == "normal":
                    # indices = self.sample_normal_indices(stim, num_CAPs)
                    indices = self.sample_with_min_spacing(stim, num_CAPs, distribution_func=norm.rvs, loc=600, scale=500)
        
                elif self.CAP_dist == "uniform":
                    # indices = self.sample_uniform_indices(stim, num_CAPs)
                    indices = self.sample_with_min_spacing(stim, num_CAPs, distribution_func=np.random.uniform, low=0, high=1)

                for cap in range(len(indices)):
                    # sample a CAP 
                    duration = np.random.randint(3, 8) + np.random.random()
                    CAP = self.get_CAP(duration)

                    # insert cap into true signal 
                    if indices[cap] + len(CAP) >= self.length:
                        n = self.true_signal[indices[cap]:, channel].shape[0]
                        self.true_signal[indices[cap]:, channel] += CAP[:n]
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
        self.add_noise_params()

        # compute power of noise signal 
        rms_noise = np.mean(self.noise_signal**2, axis=0)

        # add stimuli
        if self.CAP_dist is not None: 
            self.add_all_stimuli(); 

        ### true signal ###
        if self.CAP_dist is not None:
            self.add_CAP()

        self.add_spontaneous_activity()
                
        ### combine signals ###
        self.signal = np.zeros((self.length, self.num_channels))
        for channel in range(self.num_channels):
            pow_true_signal = np.sqrt(self.SNR * rms_noise[channel])
            self.signal[:, channel] = pow_true_signal * self.true_signal[:, channel] + self.noise_signal[:, channel]
            self.true_signal[:, channel] = pow_true_signal * self.true_signal[:, channel]
        


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