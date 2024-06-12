import argparse
import torch 
import glob
import json
import pickle
import numpy as np 
import torchaudio
from tqdm import tqdm
from melhubert.model import MelHuBERTModel, MelHuBERTConfig 
from data import DataProcessor
from tools import get_monophone_end, get_monophone_mid, get_monophone_start
from s3prl.hub import *

def main(model_name, km_pth, save_pth, data_pth, ivector_pth, tsv_pth):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Load upstream model 
    upstream_model = eval(model_name)().to(device)
    upstream_model.eval()
    # Wav paths 
    wav_pths = list(glob.glob(data_pth+"**/*.flac", recursive=True))

    NLAYER = 12
    D = 3072
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
        wav, sr = torchaudio.load(pth)
        input_wavs = [wav.view(-1).to(device)]
    
        with torch.no_grad():
            out = upstream_model(input_wavs)
            fc_results = out["fc_results"]

        for layer_idx, (fc1, fc2) in enumerate(fc_results):
            check_keys = fc1.squeeze(1)
            tau = round(D[layer_idx] * 0.01)
            assert all(D[layer_idx] == len(keys) for keys in check_keys)
            topk_indices = torch.topk(torch.abs(check_keys), tau, dim=1).indices.cpu()
            if len(clusters) != len(topk_indices):
                min_len = min(len(clusters),len(topk_indices))
                topk_indices = topk_indices[:min_len,:]
                clusters = clusters[:min_len]
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
    parser.add_argument('-m', '--model-name', help='Model name')
    parser.add_argument('-k', '--km-pth', help='Kmeans clustering result')  
    parser.add_argument('-s', '--save-pth', help='Result save path')
    parser.add_argument('-d', '--data-pth', help='Dataset directory')
    parser.add_argument('-i', '--ivector-pth', help='Ivector path')
    parser.add_argument('-v', '--tsv-pth', help='.tsv path')
    args = parser.parse_args()
    main(**vars(args))
