import argparse
import json
import numpy as np 
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tools import sort_by_same_phone, sort_voiced_unvoiced
from sklearn.manifold import MDS

def generate_distinct_colors(n_colors):
    colors = []
    # Generate colors across the hue range in HSV color space
    for i in np.linspace(0, 1, n_colors, endpoint=False):
        # Fix saturation and value, vary the hue
        hsv = (i, 0.7, 0.9)  # Adjust saturation and value for better brightness and colorfulness
        rgb = mcolors.hsv_to_rgb(hsv)
        # Convert the RGB from float (0,1) to hex format for general usability
        hex_color = mcolors.to_hex(rgb)
        colors.append(hex_color)
    return colors

def draw_mds_speaker(data, save_pth, layer, n_spk, n_spk_to_visualize):
    v_data = data[layer-1]
    N_cluster = v_data.shape[0]//n_spk
    N_data = v_data.shape[0]
    D = v_data.shape[-1]
    random_baseline = round(D*0.01)/D
    # See how many phones have matching probability over random_baseline
    num_dim = [0 for i in range(N_data)]
    for i in range(N_data):
        num_dim_meaningful = np.sum(v_data[i] > random_baseline)
        num_dim[i] = num_dim_meaningful
    # Sort index by its matching probability
    indices = [[i for i in range(D)] for i in range(N_data)]
    for i in range(N_data):
        indices[i] = sorted(indices[i], key=lambda x: v_data[i][x], reverse=True)[:num_dim[i]]
    keys = {}
    idx = 0
    key_visualize_idx = []
    for i in range(N_data):
        type_v_idx = []
        nd = len(indices[i])
        for j in range(nd):
            if indices[i][j] not in keys:
                keys[indices[i][j]] = idx 
                idx += 1 
            type_v_idx.append(keys[indices[i][j]])
        key_visualize_idx.append(type_v_idx)
    print(f'{len(keys)} activate keys over {N_cluster} clusters')
    v_data = np.zeros((N_data, len(keys)))
    for i in range(N_data):
        v_data[i][key_visualize_idx[i]] = 1 
    # Multiscale scaling
    mds = MDS(n_components=2, random_state=0)
    v_data_2d = mds.fit_transform(v_data[:n_spk_to_visualize*N_cluster,:])
    color = generate_distinct_colors(n_spk_to_visualize)
    label = [f"Speaker {i+1}" for i in range(n_spk_to_visualize)]

    for idx in range(n_spk_to_visualize):
        plt.scatter(v_data_2d[idx*N_cluster:(idx+1)*N_cluster,0], v_data_2d[idx*N_cluster:(idx+1)*N_cluster,1], c=color[idx], label=label[idx])
    
    plt.legend(loc='upper right', fontsize=6)
    plt.axis('off')
    plt.savefig(save_pth, bbox_inches='tight', dpi=200)


def main(pkl_pth, save_pth, mode, layer_n, n_spk, n_spk_to_visualize):
    with open(pkl_pth, 'rb') as fp:
        data = pickle.load(fp) 

    if mode == 'mds-speaker':
        draw_mds_speaker(data, save_pth, layer_n, n_spk, n_spk_to_visualize)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--pkl-pth', help='Data .pkl path')
    parser.add_argument('-s', '--save-pth', help='Save path')
    parser.add_argument('-m', '--mode', help='Drawing figure mode', 
                    choices=['mds-speaker'])
    parser.add_argument('-l', '--layer-n', help='Layer index', type=int)
    parser.add_argument('-n', '--n-spk', help='Number of speaker', type=int)
    parser.add_argument('--n-spk-to-visualize', help='Number of speaker to visualize', type=int)
    args = parser.parse_args()
    main(**vars(args))