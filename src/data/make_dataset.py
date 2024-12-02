# This script reads the preprocessed data, and bins it into segments without the stimulation artefacts 

import pickle 
from src.data.dataset_utils import *
from tqdm import tqdm 


# 2D and 3D 
problem_channels = {
    "2D": {
        "Eye 1": {"Stim": [5, 6, 18, 21, 25], "TTX": []},
        "Eye 2": {"Stim": [6, 11, 21], "TTX": []},
        "Eye 3": {"Stim": [6, 11, 21], "TTX": []},
        "Eye 4": {"Stim": [18, 22, 24, 26, 28, 30], "TTX": list(range(32))},
        "Eye 5": {"Stim": [9], "TTX": []},
        "Eye 6": {"Stim": [5, 6], "TTX": []},
    },
    "3D": {
        "Eye 1": {"Stim": [], "TTX": []},
        "Eye 2": {"Stim": [], "TTX": []},
        "Eye 3": {"Stim": [], "TTX": []},
        "Eye 4": {"Stim": [], "TTX": []},
        "Eye 5": {"Stim": [], "TTX": []},
        "Eye 6": {"Stim": [], "TTX": []},
    }
}

# ramp 
problem_channels = {
        "Eye 1": {"num_stims": [(4, 101), (10, 101)], "channel6": [1, 2, 3, 4, 5, 6, 7], "channel7": [], "extra_stim" : [(9, 6.11)]},
        "Eye 2": {"num_stims": [(2, 101)], "channel6": list(range(32)), "channel7": [], "extra_stim" : [(2, 9.67, 5, 6.65)]},
        "Eye 3": {"num_stims": [], "channel6": [], "channel7": [], "extra_stim" : []},
        "Eye 4": {"num_stims": [], "channel6": [], "channel7": [], "extra_stim" : []},
        "Eye 5": {"num_stims": [], "channel6": [], "channel7": [], "extra_stim" : []},
        "Eye 6": {"num_stims": [], "channel6": [], "channel7": [], "extra_stim" : []},
        } 



def pre_process() -> None:
    """Function bins all data according to locations of stimulation artefacts"""
    
    # initialize dictionaries for stim, TTX and spon
    d_stim = {}
    d_ttx = {}
    d_spon = {}

    # loop over eyes 
    for eye in tqdm(range(6)):
        ############# Bad fix for now #############
        if (eye == 3): 
                continue 
        ###########################################

        # add eye to dictionaries 
        d_stim = add_to_dictionary(eye, d_stim)
        d_ttx = add_to_dictionary(eye, d_ttx)
        d_spon = add_to_dictionary(eye, d_spon)

        # read 2D data 
        stim_2, ttx_2, spon_2 = read_data(eye, "2D")
        if len(ttx_2) > 300300:
            ttx_2[:300300]

        # read 3D data 
        stim_3, ttx_3, spon_3 = read_data(eye, "3D")

        # get a spike for ttx and stim
        if eye == 0: 
            spike = get_spike(stim_2[:, 0])

        # loop over channels
        for channel in range(32):
            print(f"Channel {channel+1}/32")

            # bin 2D data 
            segments_stim = bin_data(stim_2[:, channel], spike = spike)
            segments_ttx = bin_data(ttx_2[:, channel], spike = spike)
            segments_spon = bin_spon(spon_2[:, channel])

            # save to dictionary 
            d_stim['2D']['Eye ' + str(eye+1)][channel] = segments_stim
            d_ttx['2D']['Eye ' + str(eye+1)][channel] = segments_ttx
            d_spon['2D']['Eye ' + str(eye+1)][channel] = segments_spon

            # bin 3D data 
            segments_stim = bin_data(stim_3[:, channel], spike = spike)
            segments_ttx = bin_data(ttx_3[:, channel], spike = spike)
            segments_spon = bin_spon(spon_3[:, channel])

            # save to dictionary 
            d_stim['3D']['Eye ' + str(eye+1)][channel] = segments_stim
            d_ttx['3D']['Eye ' + str(eye+1)][channel] = segments_ttx
            d_spon['3D']['Eye ' + str(eye+1)][channel] = segments_spon

    # save results 
    with open('data/processed/stimulation.pkl', 'wb') as f: pickle.dump(d_stim, f); f.close()
    with open('data/processed/spontaneous.pkl', 'wb') as f: pickle.dump(d_spon, f); f.close()
    with open('data/processed/ttx.pkl', 'wb') as f: pickle.dump(d_ttx, f); f.close()


if __name__ == "__main__":
    pre_process()