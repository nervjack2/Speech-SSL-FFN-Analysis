import argparse
import torch 
import pickle
import numpy as np 
from melhubert.model import MelHuBERTModel, MelHuBERTConfig 


def find_keys(pkl_pth, sigma, n_ivec_cluster):
    with open(pkl_pth, 'rb') as fp:
        data = pickle.load(fp)
    print(data[0].shape)
    n_layer = len(data)
    n_cluster = len(data[0])//n_ivec_cluster
    keys_layer = {}
    for idx in range(n_layer):
        l_data = data[idx]
        v_datas = []     
        for g_idx in range(n_ivec_cluster):
            v_datas.append(l_data[n_cluster*g_idx:n_cluster*(g_idx+1)])
        keys_group = {}
        for g_idx in range(n_ivec_cluster):
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
            # print(f"There are {n_match} detected keys for group {g_idx} in layer {idx}.")
            # Sort the index of keys by the matching probability of specific group
            indices = sorted(keys.keys(), key=lambda x: keys[x], reverse=True)[:n_match]
            keys_group[g_idx] = indices
        keys_layer[idx+1] = keys_group
    return keys_layer

def find_ps_keys(x):
    layer_keys = {}
    for l in x.keys():
        ps_keys = []
        for g1 in x[l].keys():
            keys = set(x[l][g1])
            for g2 in x[l].keys():
                if g1 == g2:
                    continue 
                keys = keys - set(x[l][g2])
            ps_keys += list(keys)
        print(f"There are {len(ps_keys)} property neurons in {l} layer.")    
        layer_keys[l] = ps_keys 
    return layer_keys

def erase(upstream_model, erase_layer, erase_index):
    encoder = upstream_model.encoder
    total_ffn_dim = upstream_model.model_config.encoder_ffn_embed_dim
    total_ffn_dim = [total_ffn_dim for i in range(upstream_model.model_config.encoder_layers)] if type(total_ffn_dim)==int else total_ffn_dim

    for layer_idx in erase_layer:
        erase_layer_ffn(encoder.layers[layer_idx-1].fc2, erase_index[layer_idx], layer_idx, total_ffn_dim[layer_idx-1])
        print(f"[PertubationTools] - Erase speaker information from feed-forward memory. Total of {len(erase_index[layer_idx])} related neurons.")

def erase_layer_ffn(fc2, to_erase, layer_idx, total_ffn_dim):
    new_fc2_weight = []
    for i in range(total_ffn_dim):
        if i not in to_erase:
            new_fc2_weight.append(
               fc2.weight[:,i].unsqueeze(1)
            )
        else: 
            new_fc2_weight.append(
                torch.zeros(768,1).to('cuda')
            )
    new_fc2_weight = torch.cat(new_fc2_weight, dim=1).detach()
    fc2.weight = torch.nn.Parameter(new_fc2_weight)
    return

def main(model_pth, pkl_pth, erase_layer, save_model_pth, n_ivec_cluster, sigma):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Load upstream model 
    all_states = torch.load(model_pth, map_location="cpu")
    if "melhubert" in all_states["Upstream_Config"]:
        upstream_config = all_states["Upstream_Config"]["melhubert"]
    else:
        upstream_config = all_states["Upstream_Config"]["hubert"]
    config = MelHuBERTConfig(upstream_config)
    upstream_model = MelHuBERTModel(config).to(device)
    state_dict = all_states["model"]
    upstream_model.load_state_dict(state_dict)
    upstream_model.eval() 
    # Find property neurons for each ivector group
    keys_layer = find_keys(pkl_pth, sigma, n_ivec_cluster)
    property_neurons_spk = find_ps_keys(keys_layer)
    # Erase corresponding values slot
    erase_layer = list(map(int, erase_layer.strip().split(',')))
    erase(upstream_model, erase_layer, property_neurons_spk)
    all_states["model"] = upstream_model.state_dict()
    torch.save(all_states, save_model_pth)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model-pth', help='Model path')
    parser.add_argument('-k', '--pkl-pth', help='Kmeans clustering result')  
    parser.add_argument('-e', '--erase-layer', help='Erase layers')
    parser.add_argument('-s', '--save-model-pth', help='Save model path')
    parser.add_argument('-n', '--n-ivec-cluster', help='Number of ivector clusters', type=int)
    parser.add_argument('--sigma', help='Threshold within each group', default=0.8, type=float)
    args = parser.parse_args()
    main(**vars(args))