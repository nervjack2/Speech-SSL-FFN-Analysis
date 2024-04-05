import librosa 
import argparse
import glob
import amfm_decompy.pYAAPT as pYAAPT
import amfm_decompy.basic_tools as basic
import numpy as np
import json 
from tqdm import tqdm

def extract(data_pth, save_pth, mode):
    wav_pths = glob.glob(data_pth+"**/*.wav", recursive=True)
    keys = {}
    for p in tqdm(wav_pths):
        name = p.split('/')[-1].split('.')[0]
        # Extract the pitch
        signal = basic.SignalObj(p)
        pitch = pYAAPT.yaapt(signal, **{'f0_min' : 71.0, 'f0_max' : 800.0, 'frame_space' : 20}) # calculate the pitch track
        f0 = pitch.samp_values.astype(np.float64)
        keys[name] = list(f0) 

    with open(save_pth, 'w') as fp:
        json.dump(keys, fp)

def discretize(data_pth, save_pth, mode):
    with open(data_pth, 'r') as fp:
        data = json.load(fp)
    values = []
    for value in data.values():
        for v in value:
            if v != 0:
                values.append(v)
    print(values)
    if 'extreme' not in mode:
        percent = [33,66]
        t = [np.percentile(values, percent[i]) for i in range(2)] 
    else:
        percent = [10,90]
        t = [np.percentile(values, percent[i]) for i in range(2)] 
    keys = {}
    for key, value in tqdm(data.items()):
        new_value = []
        for v in value:
            if v == 0:
                new_value.append(-1)
                continue
            if v < t[0]:
                new_value.append(0)
            elif v > t[0] and v <= t[1]:
                new_value.append(1)
            elif v > t[1]:
                new_value.append(2)
        keys[key] = new_value

    with open(save_pth, 'w') as fp:
        json.dump(keys, fp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-pth', help='Dataset directory in extract mode. JSON path in discretize mode')
    parser.add_argument('-s', '--save-pth', help='Result save path')
    parser.add_argument('-m', '--mode', help='Mode', choices=['extract', 'discretize', 'discretize-extreme'])
    args = parser.parse_args()
    if args.mode == 'extract':
        extract(**vars(args))
    elif 'discretize' in args.mode:
        discretize(**vars(args))