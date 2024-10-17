import numpy as np 
from scipy.stats import lognorm, norm, beta
from scipy.ndimage import gaussian_filter1d
from scipy.fft import fft, fftfreq, ifft 
from matplotlib import pyplot as plt 
import seaborn as sns
import os 

os.chdir("../data")

sns.set_theme()

class SimulateData:
    def __init__(self, noise : list[float, float, float], stim_freq : int, CAP_amp : float) -> None:
        """
        Simulate data for the project
        Inputs: 
            noise: np.array([float, float, float]) - noise parameters for power line intereference, 500 Hz noise and random gaussian noise
            stim_freq: float - frequency of the stimulus signal
        """
        self.noise = noise
        self.stim_freq = stim_freq
        self.CAP_amp = CAP_amp
        
        self.length = 300300
        self.num_channels = 2 
        self.fs = 3*1e4
        self.duration = self.length // self.fs 
        self.num_stims = int(self.duration * self.stim_freq)
    
    def base_signal(self):
        self.signal = np.empty((self.length, self.num_channels))
        # for i in range(self.num_channels):
        #     # mix different sinusoids to create the base signal
        #     sin1 = np.sin(2 * np.pi * 350 * np.linspace(0, self.duration, self.length))
        #     sin2 = np.cos(2 * np.pi * 400 * np.linspace(0, self.duration, self.length))
        #     sin3 = np.sin(2 * np.pi * 320 * np.linspace(0, self.duration, self.length))
        #     self.signal[:, i] = 40 * (np.random.rand() * sin1 + np.random.rand() * sin2 + np.random.rand() * sin3)
    
    def add_noise_pli(self):
        """Add power line interference noise to the signal"""
        spike = (lognorm.pdf(np.linspace(0, 10, 100), 1, 0, 1) + norm.pdf(np.linspace(-1, 1, 100), 0, 0.07)) / 2 
        hertz_20_ms = int(20 * self.fs // 1000)
        num_spikes = int(self.length / hertz_20_ms) 
        spike = spike / np.max(spike) * self.noise[0]

        for channel in range(self.num_channels):
            for i in range(num_spikes-1):
                spike_ = spike # + norm.rvs(0, 20, len(spike))
                self.signal[hertz_20_ms * i: (hertz_20_ms * i + len(spike)), channel] += spike_
                self.signal[hertz_20_ms * i- hertz_20_ms //2 : (hertz_20_ms * i + len(spike) - hertz_20_ms // 2), channel] -= spike_


    def add_noise_500_hz(self):
        # read 500Hz noise
        noise_500 = np.load("noise_files_sim_data/500_Hz.npy")
        n = len(noise_500)
        num_rep = int(np.ceil(self.length / n))

        # match length of signal 
        noise_500 = np.array([noise_500 for _ in range(num_rep)]).ravel() 
        noise_500 = noise_500[:self.length]

        # normalize noise such that the amplitude parameter is interpretable 
        noise_500 /= np.max(noise_500)
        for channel in range(self.num_channels):
            self.signal[:, channel] += noise_500 * self.noise[1]

    def add_noise_gauss(self):
        # add gaussian noise 
        for channel in range(self.num_channels): 
            self.signal[:, channel] += np.random.normal(0, self.noise[2], self.length)

    def add_noise_high_freq(self):
        # add noise around 125000
        for channel in range(self.num_channels):
            for c, hz in enumerate(np.arange(12300, 12600, 30)):
                coef = c+1 if c < 5 else 10-c
                coef /= 3 
                self.signal[:, channel] += coef * np.sin(2 * np.pi * hz * np.linspace(0, self.duration, self.length))

    def add_stimuli(self):
        # read the stimulus signal (one stim type should be used for all channels but not for same timestamp)
        SA_options = np.load("noise_files_sim_data/SA_time.npy")

        # init probability 
        p = np.ones(self.num_channels) / self.num_channels 
        
        self.SA_indices = np.zeros(self.num_stims, dtype = int)
        spacing = self.length // self.num_stims
        for i in range(self.num_stims):
            SA_idx = np.random.choice(self.num_channels, p = p)

            # construct SA and add to signals
            SA = SA_options[SA_idx]
            for channel in range(self.num_channels):
                self.signal[spacing*i:i * spacing + len(SA), channel] += SA # add random offset here later to vary the position of the stimulus 
                self.SA_indices[i] = i * spacing + len(SA)

            # update probability
            p[SA_idx] /= 10 
            p /= np.sum(p)

    def add_CAP(self):
        y1 = norm.pdf(np.linspace(-2, 1, 80), 0, 0.5)
        y2 = beta.pdf(np.linspace(0, 1, 100), 2, 3)
        Y = np.r_[4*y1, -0.8*y2]
        Y[len(Y)//2-10:len(Y) // 2 + 10] = gaussian_filter1d(Y[len(Y)//2-10:len(Y) // 2 + 10], 4)
        Y = gaussian_filter1d(Y, 4)
        Y /= np.max(Y)
        Y *= self.CAP_amp

        segment_length = self.SA_indices[1] - self.SA_indices[0]
        for channel in range(self.num_channels):
            for i in range(self.num_stims):
                try:    
                    self.signal[self.SA_indices[i]:self.SA_indices[i] + 180, channel] += Y
                    self.signal[self.SA_indices[i] + segment_length // 6 :self.SA_indices[i] + 180 + segment_length // 6, channel] += Y
                    self.signal[self.SA_indices[i] + 2*segment_length // 6:self.SA_indices[i] + 180 + 2*segment_length // 6, channel] += Y
                    self.signal[self.SA_indices[i] + 3*segment_length // 6:self.SA_indices[i] + 180 + 3*segment_length // 6, channel] += Y
                    self.signal[self.SA_indices[i] + 4*segment_length // 6:self.SA_indices[i] + 180 + 4*segment_length // 6, channel] += Y
                except: 
                    pass

    def construct_signal(self):
        # init signal 
        self.base_signal()

        # add stimuli and CAP
        self.add_stimuli(); # self.SA_indices = np.arange(0, 300300, 3000)
        self.add_CAP()

        # noise 
        self.add_noise_pli()
        self.add_noise_500_hz()
        self.add_noise_gauss()
        self.add_noise_high_freq()   


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



if __name__ == "__main__":
    simulator = SimulateData([200, 1, 10], stim_freq=10, CAP_amp=30)
    simulator.construct_signal()
    simulator.plot_data((0, 1))
