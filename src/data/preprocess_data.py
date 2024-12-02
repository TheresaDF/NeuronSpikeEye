from utils import *
from data_utils import * 
from tqdm import tqdm
import numpy as np
from scipy.stats import lognorm 
import os
import glob



def pre_process() -> None:
    
    # initialize dictionaries for stim, TTX and spon
    d_stim = {}
    d_ttx = {}
    d_spon = {}

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

    # loop over data 
    for electrode in ["2D", "3D"]:
        for eye in range(6):
            # update dictionaries 
            d_stim = add_to_dictionary(eye, d_stim)
            d_ttx = add_to_dictionary(eye, d_ttx)
            d_spon = add_to_dictionary(eye, d_spon)

            # read data 
            data_stim, data_ttx, data_spon = read_data(eye, electrode)
            
            for channel in range(32):
                print(f"Channel {channel+1}/32")
    
                # preprocess data 
                data_stim_processed = filter(data_stim[:, channel])
                data_ttx_processed = filter(data_ttx[:, channel])

                # bin data 
                segments_spon = bin_spon(data_spon[:, channel])

                # save to dictionary 
                d_stim[electrode]['Eye ' + str(eye+1)][channel] = data_stim_processed
                d_ttx[electrode]['Eye ' + str(eye+1)][channel] = data_ttx_processed
                d_spon[electrode]['Eye ' + str(eye+1)][channel] = segments_spon



                        
def pre_process_ramp() -> None: 
    problem_channels = {
            "Eye 1": {"num_stims": [(4, 101), (10, 101)], "channel6": [1, 2, 3, 4, 5, 6, 7], "channel7": [], "extra_stim" : [(9, 6.11)]},
            "Eye 2": {"num_stims": [(2, 101)], "channel6": list(range(32)), "channel7": [], "extra_stim" : [(2, 9.67, 5, 6.65)]},
            "Eye 3": {"num_stims": [], "channel6": [], "channel7": [], "extra_stim" : []},
            "Eye 4": {"num_stims": [], "channel6": [], "channel7": [], "extra_stim" : []},
            "Eye 5": {"num_stims": [], "channel6": [], "channel7": [], "extra_stim" : []},
            "Eye 6": {"num_stims": [], "channel6": [], "channel7": [], "extra_stim" : []},
            } 


