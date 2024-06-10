import argparse
import torch 
import glob
import json
import pickle
import numpy as np 
from tqdm import tqdm
from melhubert.model import MelHuBERTModel, MelHuBERTConfig 
from data import DataProcessor
from tools import get_monophone_end, get_monophone_mid, get_monophone_start

def main(model_pth, km_pth, save_pth, mean_std_pth, data_pth, ivector_pth, tsv_pth):
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
    # Load mean std 
    dp = DataProcessor(mean_std_pth, device, fp=20)
    # Wav paths 
    wav_pths = list(glob.glob(data_pth+"**/*.flac", recursive=True))

    NLAYER = upstream_config['encoder_layers']
    D = upstream_config['encoder_ffn_embed_dim']
    D = D if type(D) == list else [D for i in range(NLAYER)]

    # Load tsv path
    key_tsv_list = [] 
    with open(tsv_pth, 'r') as fp:
        fp.readline()
        for x in fp:
            pth, _ = x.split(' ')
            key = pth.split('/')[-1].split('.')[0]
            key_tsv_list.append(key)

    km_dict = {}
    max_c = 0
    with open(km_pth, 'r') as fp:
        for key, x in zip(key_tsv_list, fp):
            x = list(map(int, x.split(' ')))
            km_dict[key] = x 
            max_c = max(max_c, max(x)) 

    N_cluster = max_c+1
    print(f"Load MFA result from {len(km_dict)} utterances")

    ivector_dict = {}
    with open(ivector_pth, 'r') as fp:
        N_ivector_cluster = 0
        for key, x in zip(key_tsv_list, fp):
            ivector_dict[key] = int(x)
            N_ivector_cluster = max(N_ivector_cluster, int(x))
        N_ivector_cluster += 1 

    N = N_cluster * N_ivector_cluster

    record = [torch.zeros((N, D[i])) for i in range(12)] 
    record_n = [[0 for i in range(N)] for i in range(12)]
                
    for pth in tqdm(wav_pths): 
        key = pth.split('/')[-1].split('.')[0]
        clusters = km_dict[key]
        g = ivector_dict[key] 
        # Forward models to get FFC layer results 
        mel_input, pad_mask = dp.prepare_data(pth)
    
        with torch.no_grad():
            out = upstream_model(mel_input, pad_mask, get_hidden=True, no_pred=True)
        fc_results = out[7]
        for layer_idx, (fc1, fc2) in enumerate(fc_results):
            check_keys = fc1.squeeze(1)
            tau = round(D[layer_idx] * 0.01)
            assert all(D[layer_idx] == len(keys) for keys in check_keys)
            topk_indices = torch.topk(torch.abs(check_keys), tau, dim=1).indices.cpu()
            for k, indices in enumerate(topk_indices):
                c = clusters[k]
                record[layer_idx][g*N_cluster+c, indices] += 1
                record_n[layer_idx][g*N_cluster+c] += 1

    for idx in range(12):
        for pidx in range(N):
            if record_n[idx][pidx] != 0:
                record[idx][pidx,:] /= record_n[idx][pidx]
        record[idx] = np.array(record[idx])

    with open(save_pth, 'wb') as fp:
        pickle.dump(record, fp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model-pth', help='Model path')
    parser.add_argument('-k', '--km-pth', help='Kmeans clustering result')  
    parser.add_argument('-s', '--save-pth', help='Result save path')
    parser.add_argument('-x', '--mean-std-pth', help='Mean std path')
    parser.add_argument('-d', '--data-pth', help='Dataset directory')
    parser.add_argument('-i', '--ivector-pth', help='Ivector path')
    parser.add_argument('-v', '--tsv-pth', help='.tsv path')
    args = parser.parse_args()
    main(**vars(args))
