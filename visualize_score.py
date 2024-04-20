import argparse
import json
import os
import pickle
import numpy as np 
import matplotlib.pyplot as plt
from tools import sort_by_same_phone, sort_voiced_unvoiced, get_DBI, get_silhouette_score, find_ps_keys
from sklearn.manifold import MDS

def get_layer_score(pkl_dir, phone_idx, split_idx):
    properties = ['phone-type', 'gender', 'pitch', 'duration']
    n_cluster = [3, 2, 3, 3]
    n_phone_group = [1, 2, 3, 3]
    results = {}
    for p_idx, p in enumerate(properties):
        pkl_pth = os.path.join(pkl_dir, p+'.pkl')
        with open(pkl_pth, 'rb') as fp:
            data = pickle.load(fp)
        n_layer = len(data)
        score_layer = []
        for idx in range(n_layer):
            NPHONE = data[idx].shape[0]//n_phone_group[p_idx]
            v_datas = []
            for i in range(n_phone_group[p_idx]):
                sample_idx = [NPHONE*i+x for x in phone_idx]
                v_datas.append(data[idx][sample_idx,:])
            v_data = np.stack(v_datas, axis=0)
            v_data = v_data.reshape(-1, v_data.shape[-1])
            n_type, D = v_data.shape
            random_baseline = round(D*0.01)/D
            # See how many phones have matching probability over random_baseline
            num_dim = [0 for i in range(n_type)]
            for i in range(n_type):
                num_dim_meaningful = np.sum(v_data[i] > random_baseline)
                num_dim[i] = num_dim_meaningful

            indices = [[i for i in range(D)] for i in range(n_type)]
            for i in range(n_type):
                indices[i] = sorted(indices[i], key=lambda x: v_data[i][x], reverse=True)[:num_dim[i]]
            keys = {}
            idx = 0
            key_visualize_idx = []
            for i in range(n_type):
                type_v_idx = []
                nd = len(indices[i])
                for j in range(nd):
                    if indices[i][j] not in keys:
                        keys[indices[i][j]] = idx 
                        idx += 1 
                    type_v_idx.append(keys[indices[i][j]])
                key_visualize_idx.append(type_v_idx)
            print(f'{len(keys)} activate keys over {n_type} type on {p} property.')
            v_data = np.zeros((n_type, len(keys)))
            for i in range(n_type):
                v_data[i][key_visualize_idx[i]] = 1 
            # Multiscale scaling
            mds = MDS(n_components=2, random_state=0)
            v_data_2d = mds.fit_transform(v_data) # n_phone x 2 

            if p == 'phone-type': 
                s_idx = split_idx
            else:
                s_idx = None 

            score = get_silhouette_score(v_data_2d, n_cluster[p_idx], s_idx=s_idx)
            score_layer.append(score)

        results[p] = score_layer

    return results

def draw_score_layer_compare(pkl_dir, save_pth, phone_idx, split_idx):
    results = get_layer_score(pkl_dir, phone_idx, split_idx)
    color = {
        'phone-type': 'red',
        'gender': 'blue',
        'pitch': 'green',
        'duration': 'black'
    }
    labels = {
        'phone-type': 'phoneme',
        'gender': 'gender',
        'pitch': 'pitch',
        'duration': 'duration' 
    }
    for k, v in results.items():
        plt.plot(range(1,13), v, label=labels[k], c=color[k], marker='o')
    plt.xticks(ticks=range(1,13))
    plt.xlabel('Layer')
    plt.ylabel('Silhouette score')
    plt.legend()
    plt.savefig(save_pth, bbox_inches='tight', dpi=200)

def draw_row_pruning_score(pkl_dir, save_pth, phone_idx, split_idx, layer_idx):
    properties = ['phone-type', 'gender', 'pitch', 'duration']
    # row = [512, 1024, 1536, 2048, 2560, 2688, 2816, 2944, 3072]
    row = [2944, 3072]
    ticks = []
    pkl_dir_list = []
    for r in row:
        pkl_dir_list.append(os.path.join(pkl_dir, f'phone-uniform-{r}'))
        ticks.append(f't{r}')
        if r != 3072:
            pkl_dir_list.append(os.path.join(pkl_dir, f'phone-uniform-pruned-{r}'))
            ticks.append(f'p{r}')

    n_cluster = [3, 2, 3, 3]
    n_phone_group = [1, 2, 3, 3]

    results = {}
    for p_idx, p in enumerate(properties):
        score_property = []
        N_property = []
        for r_idx, pkl_dir in enumerate(pkl_dir_list):
            # Load selected .npy file path
            pkl_pth = os.path.join(pkl_dir, p+'.pkl')
            with open(pkl_pth, 'rb') as fp:
                data = pickle.load(fp)
            # Split selected data 
            NPHONE = data[layer_idx-1].shape[0]//n_phone_group[p_idx]
            v_datas = []
            for i in range(n_phone_group[p_idx]):
                sample_idx = [NPHONE*i+x for x in phone_idx]
                v_datas.append(data[layer_idx-1][sample_idx,:])
            v_data = np.stack(v_datas, axis=0)
            v_data = v_data.reshape(-1, v_data.shape[-1]) 
            n_type, D = v_data.shape
            random_baseline = round(D*0.01)/D
            # See how many phones have matching probability over random_baseline
            num_dim = [0 for i in range(n_type)]
            for i in range(n_type):
                num_dim_meaningful = np.sum(v_data[i] > random_baseline)
                num_dim[i] = num_dim_meaningful
            # Sort index by its matching probability
            indices = [[i for i in range(D)] for i in range(n_type)]
            for i in range(n_type):
                indices[i] = sorted(indices[i], key=lambda x: v_data[i][x], reverse=True)[:num_dim[i]]
            keys = {}
            idx = 0
            key_visualize_idx = []
            for i in range(n_type):
                type_v_idx = []
                nd = len(indices[i])
                for j in range(nd):
                    if indices[i][j] not in keys:
                        keys[indices[i][j]] = idx 
                        idx += 1 
                    type_v_idx.append(keys[indices[i][j]])
                key_visualize_idx.append(type_v_idx)
            print(f'{len(keys)} activate keys over {n_type} type on {p} property.')
            # Create keys activated vector 
            v_data = np.zeros((n_type, len(keys)))
            for i in range(n_type):
                v_data[i][key_visualize_idx[i]] = 1 
            # Multiscale scaling
            mds = MDS(n_components=2, random_state=0)
            v_data_2d = mds.fit_transform(v_data) # n_phone x 2 

            if p == 'phone-type': 
                s_idx = split_idx
            else:
                s_idx = None 

            score = get_silhouette_score(v_data_2d, n_cluster[p_idx], s_idx=s_idx)
            score_property.append(score)

        results[p] = score_property

    color = {
        'phone-type': 'red',
        'gender': 'blue',
        'pitch': 'green',
        'duration': 'black'
    }
    for k, v in results.items():
        plt.plot(range(len(ticks)), v, label=k, c=color[k], marker='o')

    plt.xticks(ticks=range(len(ticks)), labels=ticks, rotation=90)
    plt.xlabel('Rows')
    plt.ylabel('Silhouette score')
    plt.legend()
    plt.savefig(save_pth, bbox_inches='tight', dpi=200)

def draw_models_comapre(pkl_dir, save_pth, phone_idx, split_idx):
    models_list = ['melhubert_base', 'hubert_base', 'wav2vec2_base', 'wavlm_base']
    properties = ['phone-type', 'gender', 'pitch']
    model_score = {x: [] for x in models_list}
    
    for model in models_list:
        pkl_model_dir = os.path.join(pkl_dir, f"pkl_{model}_dev_clean_merge")
        results = get_layer_score(pkl_model_dir, phone_idx, split_idx)
        for p in properties:
            max_score = max(results[p])
            model_score[model].append(max_score)
    color = {
        'melhubert_base': 'red',
        'hubert_base': 'blue',
        'wav2vec2_base': 'green',
        'wavlm_base': 'black'
    }
    label = {
        'melhubert_base': 'MelHuBERT',
        'hubert_base': 'HuBERT',
        'wav2vec2_base': 'Wav2vec 2.0',
        'wavlm_base': 'WavLM'
    }

    x = np.arange(3)  # the label locations
    width = 0.08  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained', figsize=(5,4))

    for m in models_list:
        offset = width * multiplier
        ax.bar(x*0.5 + offset, model_score[m], width, label=label[m], color=color[m])
        multiplier += 1

    xticks = ['phoneme', 'gender', 'pitch']
    ax.set_ylabel('Silhouette score')
    ax.set_xticks(x*0.5 + width*1.5, xticks)
    
    ax.legend(loc='upper left')
    plt.savefig(save_pth, dpi=200, bbox_inches='tight')


def main(pkl_dir, save_pth, mode, phone_label_pth):
    with open(phone_label_pth, 'r') as fp:
        phone_label = json.load(fp)
    sort_phone = sorted(phone_label, key=lambda x: phone_label[x][1], reverse=True)
    sort_phone_same = sort_by_same_phone(sort_phone)
    sort_phone_unvoiced, num_type = sort_voiced_unvoiced(sort_phone_same)
    split_idx = [0]+[sum(num_type[:i+1]) for i in range(len(num_type))]
    print('Split index =', split_idx)
    if mode == 'layer-compare':
        ## Hyperparameters 
        ## ===============
        phone_name = sort_phone_unvoiced
        phone_idx = [phone_label[x][0] for x in phone_name]
        ## ===============
        draw_score_layer_compare(pkl_dir, save_pth, phone_idx, split_idx)

    elif mode == 'row-pruning-score':
        ## Hyperparameters 
        ## ===============
        layer_idx = 12
        phone_name = sort_phone_unvoiced
        phone_idx = [phone_label[x][0] for x in phone_name]
        ## ===============
        draw_row_pruning_score(pkl_dir, save_pth, phone_idx, split_idx, layer_idx)
    
    elif mode == 'models-compare':
        ## Hyperparameters 
        ## ===============
        phone_name = sort_phone_unvoiced
        phone_idx = [phone_label[x][0] for x in phone_name]
        ## ===============
        draw_models_comapre(pkl_dir, save_pth, phone_idx, split_idx)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--pkl-dir', 
        help='Data .pkl dir. Should contain four .pkl file, including phone type, gender, pitch, and duration.')
    parser.add_argument('-s', '--save-pth', help='Save path')
    parser.add_argument('-m', '--mode', help='Drawing figure mode', 
                    choices=['layer-compare', 'row-pruning-score', 'models-compare'])

    parser.add_argument('-p', '--phone-label-pth', help='Phoneme lable path')
    args = parser.parse_args()
    main(**vars(args))