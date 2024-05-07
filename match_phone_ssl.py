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

def main(model_pth, mfa_json, save_pth, fp, mean_std_pth, data_pth, phone_type, extra_class, tsv_pth):
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
    dp = DataProcessor(mean_std_pth, device, fp)
    # Wav paths 
    wav_pths = list(glob.glob(data_pth+"**/*.flac", recursive=True))
    
    # =====Change to unsupervised clusters==========
    # Load phoneme force align result 
    with open(mfa_json, 'r') as fp:
        mfa = json.load(fp)
    print(f"Load MFA result from {len(mfa)} utterances")
    # =====Change to unsupervised clusters==========

    NLAYER = upstream_config['encoder_layers']
    D = upstream_config['encoder_ffn_embed_dim']
    D = D if type(D) == list else [D for i in range(NLAYER)]
    N_phone = 41

    # Load tsv path
    key_tsv_list = [] 
    with open(tsv_pth, 'r') as fp:
        fp.readline()
        for x in fp:
            pth, _ = x.split(' ')
            key = pth.split('/')[-1].split('.')[0]
            key_tsv_list.append(key)

    N_list = {
        'ivector-2c': N_phone*2
    }
    N = N_list[extra_class]

    if extra_class == 'ivector-2c':
        record = [torch.zeros((N, D[i])) for i in range(12)] 
        record_n = [[0 for i in range(N*2)] for i in range(12)]
        ivector_dict = {}
        with open('./info/ivector-2c.km') as fp:
            for key, x in zip(key_tsv_list, fp):
                ivector_dict[key] = int(x)
                
    for pth in tqdm(wav_pths): 
        key = pth.split('/')[-1].split('.')[0]
        phoneme = mfa[key]
        if extra_class == 'ivector-2c':
            g = ivector_dict[key] 
        if phone_type == 'end-phone':
            check_idx = get_monophone_end(phoneme)
        elif phone_type == 'mid-phone':
            check_idx = get_monophone_mid(phoneme)
        elif phone_type == 'start-phone':
            check_idx = get_monophone_start(phoneme)
        check_phone = [phoneme[idx] for idx in check_idx]
        # Forward models to get FFC layer results 
        mel_input, pad_mask = dp.prepare_data(pth)
    
        with torch.no_grad():
            out = upstream_model(mel_input, pad_mask, get_hidden=True, no_pred=True)
        fc_results = out[7]
        for layer_idx, (fc1, fc2) in enumerate(fc_results):
            check_keys = fc1.squeeze(1)[check_idx,:]
            tau = round(D[layer_idx]*0.01)
            for k in range(len(check_idx)):
                keys = torch.abs(check_keys[k]) # D 
                assert D[layer_idx] == len(keys)
                p = check_phone[k] 
                _, topk_indices = torch.topk(keys, tau)
                topk_indices = topk_indices.cpu()
                if extra_class == 'ivector-2c':
                    record[layer_idx][g*N_phone+p, topk_indices] += 1 
                    record_n[layer_idx][g*N_phone+p] += 1 

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
    parser.add_argument('-f', '--mfa-json', help='Force aligning result')  
    parser.add_argument('-s', '--save-pth', help='Result save path')
    parser.add_argument('-p', '--fp', type=int, help='Frame period', choices=[10, 20])
    parser.add_argument('-x', '--mean-std-pth', help='Mean std path')
    parser.add_argument('-d', '--data-pth', help='Dataset directory')
    parser.add_argument('-t', '--phone-type', help='Phone type', choices=['end-phone', 'mid-phone', 'start-phone'])
    parser.add_argument('-c', '--extra-class', choices=['ivector-2c'])
    parser.add_argument('-v', '--tsv-pth', help='.tsv path')
    args = parser.parse_args()
    main(**vars(args))
