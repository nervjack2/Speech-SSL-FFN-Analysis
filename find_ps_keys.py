import argparse 
import json 
import numpy as np 
import os
import pickle
import matplotlib.pyplot as plt
from tools import sort_by_same_phone, sort_voiced_unvoiced, find_ps_keys

group_name = {
    'phone-type': ['vowels', 'voiced-consonants', 'unvoiced-consonants'],
    'gender': ['male', 'female'],
    'pitch': ['<129.03Hz', '129.03-179.78Hz', '>179.78Hz'],
    'duration': ['<60ms', '60-100ms', '>100ms']
}

def find_keys_gender(pkl_pth, sigma, phone_idx):
    pkl_pth = os.path.join(pkl_pth)
    with open(pkl_pth, 'rb') as fp:
        data = pickle.load(fp)
    n_layer = len(data)
    keys_layer = {}
    for idx in range(n_layer):
        v_datas = []
        NPHONE = data[idx].shape[0] // 2
        for i in range(2):
            sample_idx = [NPHONE*i+x for x in phone_idx]
            v_datas.append(data[idx][sample_idx,:]) 
        n_group = len(v_datas)
        keys_group = {}
        for g_idx in range(n_group):
            n_phone, D = v_datas[g_idx].shape
            random_baseline = round(D*0.01)/D
            num_dim = [0 for i in range(n_phone)]
            for i in range(n_phone):
                num_dim_meaningful = np.sum(v_datas[g_idx][i] > random_baseline)
                num_dim[i] = num_dim_meaningful
            indices = [[i for i in range(D)] for i in range(n_phone)]
            for i in range(n_phone):
                indices[i] = sorted(indices[i], key=lambda x: v_datas[g_idx][i][x], reverse=True)[:num_dim[i]]
            keys = {}
            for i in range(n_phone):
                nd = len(indices[i])
                for j in range(nd):
                    keys[indices[i][j]] = keys.get(indices[i][j], 0)+1 
            for k, v in keys.items():
                keys[k] = v/n_phone
   
            n_match = np.sum(np.array(list(keys.values())) >= sigma)
            print(f"There are {n_match} detected keys for group {group_name['gender'][g_idx]} of gender in layer {idx}.")
            # Sort the index of keys by the matching probability of specific group
            indices = sorted(keys.keys(), key=lambda x: keys[x], reverse=True)[:n_match]
            keys_group[group_name['gender'][g_idx]] = indices
        keys_layer[idx+1] = keys_group
    return keys_layer

def find_gender_ps_keys(keys):
    layer_keys = {}
    for l in keys.keys():
        # Calculating property-specific keys
        ps_keys = []
        for g1 in keys[l].keys():
            for index1 in keys[l][g1]:
                flag = True 
                for g2 in keys[l].keys():
                    if g1 == g2:
                        continue
                    if index1 in keys[l][g2]:
                        flag = False 
                        break 
                if flag:
                    ps_keys.append(index1)
        print(f"There are {len(ps_keys)} property-specific keys for gender in {l} layer.")    
        layer_keys[l] = ps_keys
    return layer_keys

def main(pkl_pth, save_pth, phone_label_pth, sigma):
    with open(phone_label_pth, 'r') as fp:
        phone_label = json.load(fp)
    sort_phone = sorted(phone_label, key=lambda x: phone_label[x][1], reverse=True)
    sort_phone_same = sort_by_same_phone(sort_phone)
    sort_phone_unvoiced, num_type = sort_voiced_unvoiced(sort_phone_same)

    phone_name = sort_phone_unvoiced
    phone_idx = [phone_label[x][0] for x in phone_name]
    keys = find_keys_gender(pkl_pth, sigma, phone_idx)
    ps_keys = find_gender_ps_keys(keys)
    with open(save_pth, 'w') as fp:
        json.dump(ps_keys, fp, indent=3)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pkl-pth', help='Data .pkl path')
    parser.add_argument('-s', '--save-pth', help='Save path')
    parser.add_argument('-l', '--phone-label-pth', help='Phoneme lable path')
    parser.add_argument('--sigma', default=0.8, type=float)
    args = parser.parse_args()
    main(**vars(args))