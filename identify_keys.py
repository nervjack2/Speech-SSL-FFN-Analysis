import argparse 
import json 
import numpy as np 
import os
import pickle
import matplotlib.pyplot as plt
from tools import sort_by_same_phone, sort_voiced_unvoiced, find_ps_keys
from matplotlib_venn import venn2, venn3

def find_keys(pkl_dir, sigma, n):
    properties = ['cluster', 'ivector']
    n_group = [n, 2]
    cluster_per_group = [1, n]
    results = {}
    for p_idx, p in enumerate(properties):
        pkl_pth = os.path.join(pkl_dir, p+'.pkl')
        with open(pkl_pth, 'rb') as fp:
            data = pickle.load(fp)
        print(data[0].shape)
        n_layer = len(data)
        # n_cluster = len(data)//n_group[p_idx]
        keys_layer = {}
        for idx in range(n_layer):
            l_data = data[idx]
            v_datas = []
            n_cluster = cluster_per_group[p_idx]     
            for g_idx in range(n_group[p_idx]):
                v_datas.append(l_data[n_cluster*g_idx:n_cluster*(g_idx+1)])
            keys_group = {}
            for g_idx in range(n_group[p_idx]):
                _, D = v_datas[g_idx].shape
                random_baseline = round(D*0.01)/D
                num_dim = [0 for i in range(n_cluster)]
                for i in range(n_cluster):
                    num_dim_meaningful = np.sum(v_datas[g_idx][i] > random_baseline)
                    num_dim[i] = num_dim_meaningful
                indices = [[i for i in range(D)] for i in range(n_cluster)]
                for i in range(n_cluster):
                    indices[i] = sorted(indices[i], key=lambda x: v_datas[g_idx][i][x], reverse=True)[:num_dim[i]]
                keys = {}
                for i in range(n_cluster):
                    nd = len(indices[i])
                    for j in range(nd):
                        keys[indices[i][j]] = keys.get(indices[i][j], 0)+1 
                n_keys = len(keys)
                # Match probability for a specific group 
                for k, v in keys.items():
                    keys[k] = v/n_cluster
                n_match = np.sum(np.array(list(keys.values())) >= sigma)
                print(f"There are {n_match} detected keys for group {g_idx} of property {p} in layer {idx}.")
                # Sort the index of keys by the matching probability of specific group
                indices = sorted(keys.keys(), key=lambda x: keys[x], reverse=True)[:n_match]
                keys_group[g_idx] = indices
            keys_layer[idx+1] = keys_group
        results[p] = keys_layer
    return results

def draw_layer_n(x, save_pth):
    data = find_ps_keys(x)
    layer_n_ps_keys = {p: [len(data[p][l]) for l in data[p].keys()]  for p in data.keys()}
    color = {
        'cluster': 'C0', 
        'ivector': 'C1'
    }
    label = {
        'cluster': 'cluster',
        'ivector': 'ivector'
    }
    for k, v in layer_n_ps_keys.items():
        plt.plot(range(1, 12+1), v, label=label[k], color=color[k], marker='o')
    plt.legend()
    plt.xlabel('Layer')
    plt.ylabel('Num. property neurons')
    plt.savefig(save_pth, bbox_inches='tight', dpi=200)


def main(pkl_dir, save_pth, mode, sigma, nc):
    if mode == 'layer-n-compare':
        keys = find_keys(pkl_dir, sigma, nc)
        draw_layer_n(keys, save_pth)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--pkl-dir', help='Data .pkl directory')
    parser.add_argument('-s', '--save-pth', help='Save path')
    parser.add_argument('-m', '--mode', help='Drawing figure mode', choices=['layer-n-compare'])
    parser.add_argument('--sigma', default=1.0, type=float)
    parser.add_argument('--nc', default=100, type=int)
    args = parser.parse_args()
    main(**vars(args))