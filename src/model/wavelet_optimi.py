from src.data.preprocess_utils import bin_data
from src.model.wavelet_model import * 
from tqdm import tqdm 
import pickle 

# helper function 
CAP_length = lambda x: len(x) if type(x) == list else 0


def get_data(): 

    # read the data
    with open("src/visualizations/simulated_data.pkl", "rb") as f:
        config = pickle.load(f)
    f.close()

    filtered_signal = config['filtered_signal']
    CAP_indices = config['CAP_indices']

    # get all bins 
    all_segments = np.zeros((100*32, 2400))

    for channel in range(32):
        peaks, _ = find_peaks(filtered_signal[:, channel], height = 300, distance = 2900)
        bins = bin_data(filtered_signal[:, channel], peaks)
        all_segments[channel*100:(channel+1)*100, :] = bins

    y = np.array([[CAP_length(CAP_indices[i][channel]) for i in range(100)] for channel in range(32)]).ravel()

    return all_segments, y


def run_wavelet_model(X, y, ratio): 

    # allocate memory for the counts
    all_est_counts = np.zeros(len(y))

    # loop over all channels
    for segment in tqdm(range(X.shape[0])):
        # apply wavelet transform
        coefficients, _ = pywt.cwt(X[segment], scales=np.arange(1, 128), wavelet='cgau1', sampling_period=1/30000)
        
        # get accepted coefficients
        spike_indicators, _ = get_accepted_coefficients(coefficients, scales=np.arange(1, 128), ratio = ratio)

        # merge and parse the spikes
        TE = parse(spike_indicators, fs=30, width=(3, 9)) # from how the simulated data is constructed 
        
        # save the number of estimates caps 
        all_est_counts[segment] = len(TE)

    return all_est_counts

def grid_search(ratios : list = [0, 0.1, 0.2, 0.3, 0.4, 0.5]):
    # get data 
    X, y = get_data()

    # allcate memory for the results
    results = np.zeros(len(ratios))

    # loop over ratios 
    for i, ratio in tqdm(enumerate(ratios)): 
        estimated = run_wavelet_model(X, y, ratio)

        # get the accuracy
        results[ratios.index(ratio)] = np.sqrt(np.mean((y - estimated)**2))

    return results

if __name__ == '__main__': 
    results = grid_search()
    print(results)


    # [1.25324579 0.92110667 0.94157448 0.97082439 1.00140526 1.025     ]

