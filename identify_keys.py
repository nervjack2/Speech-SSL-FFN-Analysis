import argparse 
import json 
import numpy as np 
import os
import matplotlib.pyplot as plt
from tools import sort_by_same_phone, sort_voiced_unvoiced, find_ps_keys


group_name = {
    'phone-type': ['vowels', 'voiced-consonants', 'unvoiced-consonants'],
    'gender': ['male', 'female'],
    'pitch': ['<129.03Hz', '129.03-179.78Hz', '>179.78Hz'],
    'duration': ['<60ms', '60-100ms', '>100ms']
}

def find_keys(npy_dir, save_pth, phone_label_pth, mode, sigma, phone_idx, s_idx):
    properties = ['phone-type', 'gender', 'pitch', 'duration']
    n_phone_group = [1, 2, 3, 3]
    results = {}
    for p_idx, p in enumerate(properties):
        npy_pth = os.path.join(npy_dir, p+'.npy')
        data = np.load(npy_pth)
        n_layer, _, D = data.shape
        random_baseline = round(D*0.01)/D
        keys_layer = {}
        for idx in range(n_layer):
            v_datas = []
            for i in range(n_phone_group[p_idx]):
                sample_idx = [70*i+x for x in phone_idx]
                v_datas.append(data[idx,sample_idx,:]) 

            if p == 'phone-type':
                new_v_datas = []
                for i in range(3):
                    new_v_datas.append(v_datas[0][s_idx[i]:s_idx[i+1],:])
                v_datas = new_v_datas

            n_group = len(v_datas)
            # See the common activated keys for a specfic group (etc. Male, Female)
            keys_group = {}
            for g_idx in range(n_group):
                n_phone, n_dim = v_datas[g_idx].shape
                num_dim = [0 for i in range(n_phone)]
                for i in range(n_phone):
                    num_dim_meaningful = np.sum(v_datas[g_idx][i] > random_baseline)
                    num_dim[i] = num_dim_meaningful
                indices = [[i for i in range(n_dim)] for i in range(n_phone)]
                for i in range(n_phone):
                    indices[i] = sorted(indices[i], key=lambda x: v_datas[g_idx][i][x], reverse=True)[:num_dim[i]]
                keys = {}
                for i in range(n_phone):
                    nd = len(indices[i])
                    for j in range(nd):
                        keys[indices[i][j]] = keys.get(indices[i][j], 0)+1 
                n_keys = len(keys)
                # Match probability for a specific group 
                for k, v in keys.items():
                    keys[k] = v/n_phone
   
                n_match = np.sum(np.array(list(keys.values())) >= sigma)
                print(f"There are {n_match} detected keys for group {g_idx} of property {p} in layer {idx}.")
                # Sort the index of keys by the matching probability of specific group
                indices = sorted(keys.keys(), key=lambda x: keys[x], reverse=True)[:n_match]
                keys_group[group_name[p][g_idx]] = indices
            keys_layer[idx+1] = keys_group
        results[p] = keys_layer

    return results

def draw_layer_n(x, save_pth):
    data = find_ps_keys(x)
    layer_n_ps_keys = {p: [len(data[p][l]) for l in data[p].keys()]  for p in data.keys()}
    
    color = {
        'phone-type': 'red',
        'gender': 'blue',
        'pitch': 'green',
        'duration': 'black'
    }
    for k, v in layer_n_ps_keys.items():
        plt.plot(range(1, 12+1), v, label=k, color=color[k], marker='o')
    plt.legend()
    plt.xlabel('Layer')
    plt.ylabel('Num. property-activated keys')
    plt.savefig(save_pth, bbox_inches='tight', dpi=200)

def draw_venn_ps_keys(x, save_pth, layer_idx):
    # ====================
    # Note: We do not visualize the result of duration, since FFD does not show good capability to caputre duration information.
    # ====================
    from matplotlib_venn import venn3
    data = find_ps_keys(x)
    p_name = ['phone-type', 'gender', 'pitch']
    l_data = [data[p][layer_idx] for p in p_name]
    n_p = len(p_name)
    base_s = '000'
    set_sizes = {}
    # One set 
    for i in range(n_p):
        k = base_s[:i]+'1'+base_s[i+1:]
        set_sizes[k] = len(l_data[i])
    # Two set 
    for i in range(n_p):
        base_s_i = base_s[:i]+'1'+base_s[i+1:]
        for j in range(i+1, n_p):
            k = base_s_i[:j]+'1'+base_s_i[j+1:]
            n = set(l_data[i]) & set(l_data[j])
            set_sizes[k] = len(n) 
    # Three set 
    for i in range(n_p):
        base_s_i = base_s[:i]+'1'+base_s[i+1:]
        for j in range(i+1, n_p):
            base_s_j = base_s_i[:j]+'1'+base_s_i[j+1:]
            for l in range(j+1, n_p):
                k = base_s_j[:l]+'1'+base_s_j[l+1:]
                n = set(l_data[i])&set(l_data[j])&set(l_data[l])
                set_sizes[k] = len(n) 
    # Four set 
    k = '111'
    x = set([])
    for i in range(n_p):
        x &= set(l_data[i])
    set_sizes[k] = len(x)    
    venn = venn3(subsets=set_sizes, set_labels=p_name)
    plt.savefig(save_pth, bbox_inches='tight', dpi=200)

def draw_row_pruning_n_ps_keys(rows_keys, save_pth, layer_idx, row):
    properties = ['phone-type', 'gender', 'pitch', 'duration']
    ticks = []
    for r in row:
        ticks.append(f't{r}')
        if r != 3072:
            ticks.append(f'p{r}')
    row_ps_keys = []
    for x in rows_keys:
        row_ps_keys.append(find_ps_keys(x))
    v_data = {}
    for p in properties:
        v_data[p] = []
        for r_ps_keys in row_ps_keys:
            v_data[p].append(len(r_ps_keys[p][layer_idx]))
    color = {
        'phone-type': 'red',
        'gender': 'blue',
        'pitch': 'green',
        'duration': 'black'
    }
    for k, v in v_data.items():
        plt.plot(range(len(ticks)), v, label=k, color=color[k], marker='o')
    plt.xticks(ticks=range(len(ticks)), labels=ticks, rotation=90)
    plt.xlabel('Rows')
    plt.ylabel('Num. Property-Specific Keys')
    plt.legend()
    plt.savefig(save_pth, bbox_inches='tight', dpi=200)

def main(npy_dir, save_pth, phone_label_pth, mode):
    with open(phone_label_pth, 'r') as fp:
        phone_label = json.load(fp)
    sort_phone = sorted(phone_label, key=lambda x: phone_label[x][1], reverse=True)
    sort_phone_same = sort_by_same_phone(sort_phone)
    sort_phone_unvoiced, num_type = sort_voiced_unvoiced(sort_phone_same)
    split_idx = [0]+[sum(num_type[:i+1]) for i in range(len(num_type))]
    n_phone = len(sort_phone)

    if mode == 'layer-n-compare':
        # Hyperparameters
        # =================
        sigma = 0.8
        # =================
        phone_name = sort_phone_unvoiced
        phone_idx = [phone_label[x][0] for x in phone_name]
        keys = find_keys(npy_dir, save_pth, phone_label_pth, mode, sigma, phone_idx, split_idx)
        draw_layer_n(keys, save_pth)
    elif mode == 'venn-ps-keys':
        # Hyperparameters
        # ================
        layer_idx = 1
        sigma = 0.8
        # ================
        phone_name = sort_phone_unvoiced
        phone_idx = [phone_label[x][0] for x in phone_name]
        keys = find_keys(npy_dir, save_pth, phone_label_pth, mode, sigma, phone_idx, split_idx)
        draw_venn_ps_keys(keys, save_pth, layer_idx)
    elif mode == 'row-pruning-n-ps-keys':
        # Hyperparameters
        # ================
        layer_idx = 12
        sigma = 0.8
        # ================
        phone_name = sort_phone_unvoiced
        phone_idx = [phone_label[x][0] for x in phone_name]
        npy_dir_paths = []
        row = [512, 1024, 1536, 2048, 2560, 2688, 2816, 2944, 3072]
        for r in row:
            npy_dir_paths.append(os.path.join(npy_dir, f"phone-uniform-{r}"))
            if r != 3072:
                npy_dir_paths.append(os.path.join(npy_dir, f"phone-uniform-pruned-{r}"))
        rows_keys = []
        for idx, npy_dir in enumerate(npy_dir_paths):
            keys = find_keys(npy_dir, save_pth, phone_label_pth, mode, sigma, phone_idx, split_idx)
            rows_keys.append(keys)
        draw_row_pruning_n_ps_keys(rows_keys, save_pth, layer_idx, row)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--npy-dir', help='Data .npy directory')
    parser.add_argument('-s', '--save-pth', help='Save path')
    parser.add_argument('-p', '--phone-label-pth', help='Phoneme lable path')
    parser.add_argument('-m', '--mode', help='Drawing figure mode', choices=['layer-n-compare', 'venn-ps-keys', 'row-pruning-n-ps-keys'])
    args = parser.parse_args()
    main(**vars(args))