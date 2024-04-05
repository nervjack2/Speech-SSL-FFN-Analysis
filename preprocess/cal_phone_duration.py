import sys 
import argparse
import json
from tqdm import tqdm
import numpy as np 

def get_phone_dur(phone):
    d = 0 
    dur = []
    pre_x = phone[0]
    for idx, x in enumerate(phone):
        if x != pre_x:
            if pre_x != 0: # Remove silence
                dur.append(d)
            else:
                dur.append(-1)
            d = 0
        pre_x = x 
        d += 1 
    return dur

def main(mfa_json, save_pth, mode):
    # Load phoneme force align result 
    with open(mfa_json, 'r') as fp:
        mfa = json.load(fp)
    print(f"Load MFA result from {len(mfa)} utterances")
    
    dur_dict = {}
    n_dur_dict = {}
    for key, phone in mfa.items():
        dur = get_phone_dur(phone)
        dur_dict[key] = dur
        for d in dur:
            n_dur_dict[d] = n_dur_dict.get(d,0)+1 

    total = sum(n_dur_dict.values())
    n_dur_dict = {k: v/total for k,v in n_dur_dict.items()}
    sort_keys = sorted(n_dur_dict.keys(), key=lambda x: x)
    n_dur_dict = {x: n_dur_dict[x]  for x in sort_keys}

    avg = sum([k*v for k, v in n_dur_dict.items() if k!=-1])
    print(f"Average phone duration is {avg}")

    # Calculate percentile 
    value = []
    for key, vv in tqdm(dur_dict.items()):
        for v in vv:
            if v != -1:
                value.append(v)

    if mode == 'normal':
        percent = [33,66]
    elif mode == 'extreme':
        percent = [10,90]

    threshold = [np.percentile(value, percent[i]) for i in range(2)]
    print(f"Set threshold = {threshold}")

    count = [0,0,0]
    group_dur_dict = {}
    for k, v in tqdm(dur_dict.items()):
        new_v = []
        for x in v: 
            if x == -1:
                new_v.append(-1)
                continue
            if x <= threshold[0]:
                new_v.append(0)
                count[0] += 1 
            elif x > threshold[0] and x <= threshold[1]:
                new_v.append(1)
                count[1] += 1 
            elif x > threshold[1]:
                new_v.append(2)
                count[2] += 1
        group_dur_dict[k] = new_v

    print(f"Probability = {[x/sum(count) for x in count]}")
    # Normal: [0.4528829946251948, 0.31544381897401647, 0.23167318640078874]
    # Extreme: [0.2216656595532657, 0.6839307105980134, 0.09440362984872097]
   
    with open(save_pth, 'w') as fp:
        json.dump(group_dur_dict, fp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--mfa-json', help='Force aligning result')
    parser.add_argument('-s', '--save-pth', help='Result save path')
    parser.add_argument('-m', '--mode', help='Mode', choices=['normal','extreme'])
    args = parser.parse_args()
    main(**vars(args))