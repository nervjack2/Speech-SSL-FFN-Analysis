import argparse
import torch 
import torch.nn.functional as F
import glob
import json
import pickle
import os 
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
from melhubert.model import MelHuBERTModel, MelHuBERTConfig 
from data import DataProcessor

def main(model_pth, cluster_pth, save_pth, fp, mean_std_pth, data_pth, select):
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

    NLAYER = upstream_config['encoder_layers']
    # Create 3072 values (fc2 multiply by output projection)
    params = dict(upstream_model.named_parameters())
    final_proj_w = params['final_proj.weight'].permute(1,0)
    final_proj_b = params['final_proj.bias']

    layers_values = []
    for i in range(NLAYER):
        w = params[f'encoder.layers.{i}.fc2.weight'].permute(1,0)
        b = params[f'encoder.layers.{i}.fc2.bias']
        layers_values.append((w,b))

    if '.json' in cluster_pth:
        cluster_mode = 'json'
        # Load cluster alignment 
        with open(cluster_pth, 'r') as f:
            cluster_align = json.load(f)
    else:
        cluster_mode = 'npy'

    # Load mean std 
    dp = DataProcessor(mean_std_pth, device, fp)
    # Wav paths 
    if 'train' in data_pth:
        import random
        wav_pths = list(glob.glob(data_pth+"**/*.flac", recursive=True))
        data_len = len(wav_pths)
        wav_pths = random.sample(wav_pths, int(data_len*0.1))
    else:
        wav_pths = list(glob.glob(data_pth+"**/*.flac", recursive=True))

    n_correct = np.zeros((NLAYER))
    n_total = np.zeros((NLAYER))

    labels = []
    embs = [[] for i in range(12)]
    labels_count = [0 for i in range(512)]

    for pth in tqdm(wav_pths): 
        key = pth.split('/')[-1].split('.')[0]
        mel_input, pad_mask = dp.prepare_data(pth)

        if cluster_mode == 'json':
            cluster = cluster_align[key] # T 
        elif cluster_mode == 'npy':
            cluster = np.load(os.path.join(cluster_pth, key+'.npy'))
        
        for c in cluster:
            labels.append(c)
            labels_count[c] += 1 

        with torch.no_grad():
            out = upstream_model(mel_input, pad_mask, get_hidden=True, no_pred=True)
        fc_results = out[7]

        for layer_idx, (fc1, _) in enumerate(fc_results):
            check_keys = fc1.squeeze(1) # T x D
            T, D = check_keys.shape
            tau = round(D*0.01)
            for t in range(T):
                keys = torch.abs(check_keys[t]) # D 
                if select == 'ratio':
                    _, topk_indices = torch.topk(keys, tau)
                elif select == 'top1':
                    topk_indices = torch.argmax(keys).unsqueeze(0)
                weights = check_keys[t][topk_indices] # tau
                values = layers_values[layer_idx][0][topk_indices,:] # tau x H
                emb = torch.matmul(weights, values) # H
                embs[layer_idx].append(np.array(emb.cpu().detach()))
    
    save_data = {}
    save_data['labels_count'] = labels_count
    save_data['labels'] = labels
    save_data['embs'] = embs
    
    with open(save_pth, 'wb') as fp:
        pickle.dump(save_data, fp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model-pth', help='Model path')
    parser.add_argument('-c', '--cluster-pth', help='Cluster path. Either .json or a directory')  
    parser.add_argument('-s', '--save-pth', help='Result save path')
    parser.add_argument('-p', '--fp', type=int, help='Frame period', choices=[10, 20])
    parser.add_argument('-x', '--mean-std-pth', help='Mean std path')
    parser.add_argument('-d', '--data-pth', help='Dataset directory')
    parser.add_argument('--select', help='Way to select weights', choices=['top1', 'ratio'])
    args = parser.parse_args()
    main(**vars(args))
