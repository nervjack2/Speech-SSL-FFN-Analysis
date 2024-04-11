import argparse
import json
import numpy as np 
import pickle
import matplotlib.pyplot as plt
from tools import sort_by_same_phone, sort_voiced_unvoiced
from sklearn.manifold import MDS

def draw_intra_phone(data, save_pth, phone_idx, phone_name, layer):
    _, D = data[layer-1].shape
    random_baseline = round(D*0.01)/D
    v_data = data[layer-1][phone_idx,:]
    v_data = -np.sort(-v_data, axis=-1)
    n_dim = v_data.shape[-1]
    n_phone = len(phone_idx)
    fig, axs = plt.subplots(1, n_phone, figsize=(10,4))
    for i in range(n_phone):
        axs[i].bar(range(n_dim), v_data[i])
        axs[i].title.set_text(phone_name[i])
        axs[i].axhline(y=random_baseline, color='r', linestyle='-', linewidth=1)
    plt.savefig(save_pth, bbox_inches='tight', dpi=200)

def draw_keys_activated_vector_demo(data, save_pth, phone_idx, phone_name, layer, per_phone, num_type):
    v_data = data[layer-1][phone_idx,:]
    n_phone = len(phone_idx)
    D = v_data.shape[-1]
    # Sort index by its matching probability
    indices = [[i for i in range(D)] for i in range(n_phone)]
    for i in range(n_phone):
        indices[i] = sorted(indices[i], key=lambda x: v_data[i][x], reverse=True)[:per_phone]
    keys = {}
    idx = 0
    key_visualize_idx = []
    for i in range(n_phone):
        type_v_idx = []
        for j in range(per_phone):
            if indices[i][j] not in keys:
                keys[indices[i][j]] = idx 
                idx += 1 
            type_v_idx.append(keys[indices[i][j]])
        key_visualize_idx.append(type_v_idx)
    print(f'{len(keys)} activate keys over {n_phone} phoneme')
    v_data = np.zeros((n_phone, len(keys)))
    for i in range(n_phone):
        v_data[i][key_visualize_idx[i]] = 1 
    
    plt.imshow(v_data, cmap='gray')
    plt.xlabel('keys')
    plt.ylabel('phone')
    plt.yticks(range(n_phone), phone_name)
    acc = 0
    for i in range(len(num_type)-1):
        plt.axhline(acc+num_type[i]-0.5, c='red')
        acc += num_type[i]
    plt.savefig(save_pth, bbox_inches='tight', dpi=200)

def draw_mds_phone_type(data, save_pth, phone_idx, phone_name, layer, num_type):
    v_data = data[layer-1][phone_idx,:]
    n_phone = len(phone_idx)
    D = v_data.shape[-1]
    random_baseline = round(D*0.01)/D
    # See how many phones have matching probability over random_baseline
    num_dim = [0 for i in range(n_phone)]
    for i in range(n_phone):
        num_dim_meaningful = np.sum(v_data[i] > random_baseline)
        num_dim[i] = num_dim_meaningful
    # Sort index by its matching probability
    indices = [[i for i in range(D)] for i in range(n_phone)]
    for i in range(n_phone):
        indices[i] = sorted(indices[i], key=lambda x: v_data[i][x], reverse=True)[:num_dim[i]]
    keys = {}
    idx = 0
    key_visualize_idx = []
    for i in range(n_phone):
        type_v_idx = []
        nd = len(indices[i])
        for j in range(nd):
            if indices[i][j] not in keys:
                keys[indices[i][j]] = idx 
                idx += 1 
            type_v_idx.append(keys[indices[i][j]])
        key_visualize_idx.append(type_v_idx)
    print(f'{len(keys)} activate keys over {n_phone} phoneme')
    v_data = np.zeros((n_phone, len(keys)))
    for i in range(n_phone):
        v_data[i][key_visualize_idx[i]] = 1 
    # Multiscale scaling
    mds = MDS(n_components=2, random_state=0, normalized_stress='auto')
    v_data_2d = mds.fit_transform(v_data)
    color = ['red', 'blue', 'green']
    label = ['vowels', 'voiced-consonants', 'unvoiced-consonants']
    acc = 0 
    for idx, n in enumerate(num_type):
        plt.scatter(v_data_2d[acc:acc+n,0], v_data_2d[acc:acc+n,1], c=color[idx], label=label[idx])
        acc += n 
    for idx, name in enumerate(phone_name):
        plt.annotate(name, (v_data_2d[idx,0],v_data_2d[idx,1]))
    plt.legend(loc='upper right', fontsize=6)
    plt.axis('off')
    plt.title(f'Layer {layer}')
    plt.savefig(save_pth, bbox_inches='tight', dpi=200)

def draw_mds_gender(data, save_pth, phone_name, layer, num_type, phone_idx):
    NPHONE = data[layer-1].shape[0]//2
    v_datas = []
    for i in range(2):
        idx = [NPHONE*i+x for x in phone_idx]
        v_datas.append(data[layer-1][idx,:])
    v_data = np.stack(v_datas, axis=0)
    v_data = v_data.reshape(-1, v_data.shape[-1])
    n_phone = len(phone_name)
    n_data = 2*n_phone
    D = v_data.shape[-1]
    random_baseline = round(D*0.01)/D
    # See how many phones have matching probability over random_baseline
    num_dim = [0 for i in range(n_data)]
    for i in range(n_data):
        num_dim_meaningful = np.sum(v_data[i] > random_baseline)
        num_dim[i] = num_dim_meaningful
    # Sort index by its matching probability
    indices = [[i for i in range(D)] for i in range(n_data)]
    for i in range(n_data):
        indices[i] = sorted(indices[i], key=lambda x: v_data[i][x], reverse=True)[:num_dim[i]]
    keys = {}
    idx = 0
    key_visualize_idx = []
    for i in range(n_data):
        type_v_idx = []
        nd = len(indices[i])
        for j in range(nd):
            if indices[i][j] not in keys:
                keys[indices[i][j]] = idx 
                idx += 1 
            type_v_idx.append(keys[indices[i][j]])
        key_visualize_idx.append(type_v_idx)
    print(f'{len(keys)} activate keys over {n_phone} phoneme')
    v_data = np.zeros((n_data, len(keys)))
    for i in range(n_data):
        v_data[i][key_visualize_idx[i]] = 1 
    # Multiscale scaling
    mds = MDS(n_components=2, random_state=0, normalized_stress='auto')
    v_data_2d = mds.fit_transform(v_data)
    color = ['red', 'blue']
    label = ['male', 'female']

    for idx in range(2):
        plt.scatter(v_data_2d[idx*n_phone:(idx+1)*n_phone,0], v_data_2d[idx*n_phone:(idx+1)*n_phone,1], c=color[idx], label=label[idx])

    for idx, name in enumerate(phone_name):
        plt.annotate(name, (v_data_2d[idx,0],v_data_2d[idx,1]))
        plt.annotate(name, (v_data_2d[n_phone+idx,0],v_data_2d[n_phone+idx,1]))
    
    plt.legend(loc='upper right', fontsize=6)
    plt.axis('off')
    plt.title(f'Layer {layer}')
    plt.savefig(save_pth, bbox_inches='tight', dpi=200)

def draw_mds_duration(data, save_pth, phone_name, layer, num_type, label, phone_idx):
    NPHONE = data[layer-1].shape[0]//3
    v_datas = []
    for i in range(3):
        idx = [NPHONE*i+x for x in phone_idx]
        v_datas.append(data[layer-1][idx,:])
    v_data = np.stack(v_datas, axis=0)
    v_data = v_data.reshape(-1, v_data.shape[-1])
    n_phone = len(phone_name)
    n_data = 3*n_phone
    D = v_data.shape[-1]
    random_baseline = round(D*0.01)/D
    # See how many phones have matching probability over random_baseline
    num_dim = [0 for i in range(n_data)]
    for i in range(n_data):
        num_dim_meaningful = np.sum(v_data[i] > random_baseline)
        num_dim[i] = num_dim_meaningful
    # Sort index by its matching probability
    indices = [[i for i in range(D)] for i in range(n_data)]
    for i in range(n_data):
        indices[i] = sorted(indices[i], key=lambda x: v_data[i][x], reverse=True)[:num_dim[i]]
    keys = {}
    idx = 0
    key_visualize_idx = []
    for i in range(n_data):
        type_v_idx = []
        nd = len(indices[i])
        for j in range(nd):
            if indices[i][j] not in keys:
                keys[indices[i][j]] = idx 
                idx += 1 
            type_v_idx.append(keys[indices[i][j]])
        key_visualize_idx.append(type_v_idx)
    print(f'{len(keys)} activate keys over {n_phone} phoneme')
    v_data = np.zeros((n_data, len(keys)))
    for i in range(n_data):
        v_data[i][key_visualize_idx[i]] = 1 
    # Multiscale scaling
    mds = MDS(n_components=2, random_state=0, normalized_stress='auto')
    v_data_2d = mds.fit_transform(v_data)
    color = ['red', 'blue', 'green']
    for idx in range(3):
        plt.scatter(v_data_2d[idx*n_phone:(idx+1)*n_phone,0], v_data_2d[idx*n_phone:(idx+1)*n_phone,1], c=color[idx], label=label[idx])
    for idx, name in enumerate(phone_name):
        plt.annotate(name, (v_data_2d[idx,0],v_data_2d[idx,1]))
        plt.annotate(name, (v_data_2d[n_phone+idx,0],v_data_2d[n_phone+idx,1]))
        plt.annotate(name, (v_data_2d[n_phone*2+idx,0],v_data_2d[n_phone*2+idx,1]))
    plt.legend(loc='upper right', fontsize=6)
    plt.axis('off')
    plt.title(f'Layer {layer}')
    plt.savefig(save_pth, bbox_inches='tight', dpi=200)

def draw_mds_pitch(data, save_pth, phone_name, layer, num_type, label, phone_idx):
    NPHONE = data[layer-1].shape[0]//3
    v_datas = []
    for i in range(3):
        idx = [NPHONE*i+x for x in phone_idx]
        v_datas.append(data[layer-1][idx,:])
    v_data = np.stack(v_datas, axis=0)
    v_data = v_data.reshape(-1, v_data.shape[-1])
    n_phone = len(phone_name)
    n_data = 3*n_phone
    D = v_data.shape[-1]
    random_baseline = round(D*0.01)/D
    # See how many phones have matching probability over random_baseline
    num_dim = [0 for i in range(n_data)]
    for i in range(n_data):
        num_dim_meaningful = np.sum(v_data[i] > random_baseline)
        num_dim[i] = num_dim_meaningful
    # Sort index by its matching probability
    indices = [[i for i in range(D)] for i in range(n_data)]
    for i in range(n_data):
        indices[i] = sorted(indices[i], key=lambda x: v_data[i][x], reverse=True)[:num_dim[i]]
    keys = {}
    idx = 0
    key_visualize_idx = []
    for i in range(n_data):
        type_v_idx = []
        nd = len(indices[i])
        for j in range(nd):
            if indices[i][j] not in keys:
                keys[indices[i][j]] = idx 
                idx += 1 
            type_v_idx.append(keys[indices[i][j]])
        key_visualize_idx.append(type_v_idx)
    print(f'{len(keys)} activate keys over {n_phone} phoneme')
    v_data = np.zeros((n_data, len(keys)))
    for i in range(n_data):
        v_data[i][key_visualize_idx[i]] = 1 
    # Multiscale scaling
    mds = MDS(n_components=2, random_state=0, normalized_stress='auto')
    v_data_2d = mds.fit_transform(v_data)
    color = ['red', 'blue', 'green', 'yellow']
    for idx in range(3):
        plt.scatter(v_data_2d[idx*n_phone:(idx+1)*n_phone,0], v_data_2d[idx*n_phone:(idx+1)*n_phone,1], c=color[idx], label=label[idx])
    for idx, name in enumerate(phone_name):
        plt.annotate(name, (v_data_2d[idx,0],v_data_2d[idx,1]))
        plt.annotate(name, (v_data_2d[n_phone+idx,0],v_data_2d[n_phone+idx,1]))
        plt.annotate(name, (v_data_2d[n_phone*2+idx,0],v_data_2d[n_phone*2+idx,1]))
    plt.legend(loc='upper right', fontsize=6)
    plt.axis('off')
    plt.title(f'Layer {layer}')
    plt.savefig(save_pth, bbox_inches='tight', dpi=200)

def main(pkl_pth, save_pth, phone_label_pth, mode, layer_n):
    with open(pkl_pth, 'rb') as fp:
        data = pickle.load(fp) 

    with open(phone_label_pth, 'r') as fp:
        phone_label = json.load(fp)
    sort_phone = sorted(phone_label, key=lambda x: phone_label[x][1], reverse=True)
    sort_phone_same = sort_by_same_phone(sort_phone)
    sort_phone_unvoiced, num_type = sort_voiced_unvoiced(sort_phone_same)
    print(sort_phone_unvoiced, num_type)

    n_phone = len(sort_phone)

    if mode == 'intra-phone':
        # Hyperparameters
        # =================
        layer = layer_n
        # =================
        phone_name = [sort_phone[0], sort_phone[n_phone//2], sort_phone[-5]]
        print(f"Visualize phone {' '.join(phone_name)} with occurance {' '.join([str(phone_label[n][1]) for n in phone_name])} respectively")
        phone_idx = [phone_label[x][0] for x in phone_name]
        draw_intra_phone(data, save_pth, phone_idx, phone_name, layer)
    elif mode == 'mds-phone-type':
        # Hyperparameters
        # =================
        layer = layer_n
        # =================
        phone_name = sort_phone_unvoiced
        phone_idx = [phone_label[x][0] for x in phone_name]
        draw_mds_phone_type(data, save_pth, phone_idx, phone_name, layer, num_type)
    elif mode == 'mds-gender':
        # Hyperparameters
        # =================
        layer = layer_n
        # =================
        phone_name = sort_phone_unvoiced
        phone_idx = [phone_label[x][0] for x in phone_name]
        draw_mds_gender(data, save_pth, phone_name, layer, num_type, phone_idx)
    elif mode == 'mds-duration':
        # Hyperparameters
        # =================
        layer = layer_n
        # Extreme 40ms 140ms
        # label = ['<40', '40-140', '>140']
        # Normal 60ms 100ms
        label = ['<60', '60-100', '>100']
        # =================
        phone_name = sort_phone_unvoiced
        phone_idx = [phone_label[x][0] for x in phone_name]
        draw_mds_duration(data, save_pth, phone_name, layer, num_type, label, phone_idx)
    elif mode == 'mds-pitch':
        # Hyperparameters
        # =================
        layer = layer_n
        # Extreme 95.80838323353294, 222.91946630147046
        # label = ['<95.81', '95.81-222.92', '>222.92']
        # Normal 129.03225806451613, 179.77528089887642
        label = ['<129.03', '129.03-179.78', '>179.78']
        # =================
        phone_name = sort_phone_unvoiced
        phone_idx = [phone_label[x][0] for x in phone_name]
        draw_mds_pitch(data, save_pth, phone_name, layer, num_type, label, phone_idx)
    elif mode == 'keys-activated-vector-demo': 
        # Hyperparameters
        # =================
        layer = layer_n
        per_phone = 5
        # phone_name = ['AH0','AH1','AH2','AO1','AO2','AO0','D','B','G','T','S','F']
        phone_name = sort_phone_unvoiced
        # num_type = [6,3,3]
        # =================
        phone_idx = [phone_label[x][0] for x in phone_name]
        draw_keys_activated_vector_demo(data, save_pth, phone_idx, phone_name, layer, per_phone, num_type)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--pkl-pth', help='Data .pkl path')
    parser.add_argument('-s', '--save-pth', help='Save path')
    parser.add_argument('-p', '--phone-label-pth', help='Phoneme lable path')
    parser.add_argument('-m', '--mode', help='Drawing figure mode', 
                    choices=['intra-phone', 
                            'mds-phone-type', 'mds-gender',
                            'mds-duration', 'mds-pitch',
                            'keys-activated-vector-demo'])
    parser.add_argument('-l', '--layer-n', help='Layer index', type=int)
    args = parser.parse_args()
    main(**vars(args))