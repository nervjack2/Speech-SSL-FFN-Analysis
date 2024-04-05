import argparse
import torch 
import glob
import json
import numpy as np 
from tqdm import tqdm
from melhubert.model import MelHuBERTModel, MelHuBERTConfig 
from data import DataProcessor
from tools import get_monophone_end, get_monophone_mid, get_monophone_start

def main(model_pth, mfa_json, save_pth, fp, mean_std_pth, data_pth, phone_type, extra_class):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Load upstream model 
    all_states = torch.load(model_pth, map_location="cpu")
    if "melhubert" in all_states["Upstream_Config"]:
        upstream_config = all_states["Upstream_Config"]["melhubert"]
    else:
        upstream_config = all_states["Upstream_Config"]["hubert"]
    print(upstream_config)
    D = upstream_config['encoder_ffn_embed_dim']
    tau = round(D*0.01)
    upstream_config = MelHuBERTConfig(upstream_config)
    upstream_model = MelHuBERTModel(upstream_config).to(device)
    state_dict = all_states["model"]
    upstream_model.load_state_dict(state_dict)
    upstream_model.eval() 
    # Load mean std 
    dp = DataProcessor(mean_std_pth, device, fp)
    # Wav paths 
    wav_pths = list(glob.glob(data_pth+"**/*.flac", recursive=True))
    # Load phoneme force align result 
    with open(mfa_json, 'r') as fp:
        mfa = json.load(fp)
    print(f"Load MFA result from {len(mfa)} utterances")

    if extra_class == 'phone-type':
        record = [torch.zeros((70, D)) for i in range(12)] 
        record_n = [[0 for i in range(70)] for i in range(12)]
    elif extra_class == 'gender':
        record = [torch.zeros((70*2, D)) for i in range(12)] 
        record_n = [[0 for i in range(70*2)] for i in range(12)]
        with open('./info/libri-test-spk-gender.json', 'r') as fp:
            gender_dict = json.load(fp)
    elif extra_class == 'duration':
        record = [torch.zeros((70*3, D)) for i in range(12)] 
        record_n = [[0 for i in range(70*3)] for i in range(12)]
        with open('./info/phone-duration.json', 'r') as fp:
            duration_dict = json.load(fp)
    elif extra_class == 'pitch':
        record = [torch.zeros((70*3, D)) for i in range(12)] 
        record_n = [[0 for i in range(70*3)] for i in range(12)]
        with open('./info/pitch-discrete.json', 'r') as fp:
            pitch_dict = json.load(fp)

    for pth in tqdm(wav_pths): 
        key = pth.split('/')[-1].split('.')[0]
        phoneme = mfa[key]
        if extra_class == 'gender':
            gender = key.split('-')[0]
            g = 0 if gender_dict[gender] == 'M' else 1 
        elif extra_class == 'duration':
            dur = duration_dict[key]
        elif extra_class == 'pitch':
            pitch = pitch_dict[key]
            gap = len(phoneme)-len(pitch)
            if gap > 0:
                pitch = [-1]*gap + pitch
            elif gap < 0:
                pitch = pitch[gap:]
        if phone_type == 'end-phone':
            check_idx = get_monophone_end(phoneme)
        elif phone_type == 'mid-phone':
            check_idx = get_monophone_mid(phoneme)
        elif phone_type == 'start-phone':
            check_idx = get_monophone_start(phoneme)
        check_phone = [phoneme[idx] for idx in check_idx]
        if extra_class == 'pitch':
            check_pitch = [pitch[idx] for idx in check_idx]
        # Forward models to get FFC layer results 
        mel_input, pad_mask = dp.prepare_data(pth)
        with torch.no_grad():
            out = upstream_model(mel_input, pad_mask, get_hidden=True, no_pred=True)
        fc_results = out[7]

        for layer_idx, (fc1, fc2) in enumerate(fc_results):
            check_keys = fc1.squeeze(1)[check_idx,:]
            for k in range(len(check_idx)):
                keys = check_keys[k] # D 
                p = check_phone[k] 
                _, topk_indices = torch.topk(keys, tau)
                topk_indices = topk_indices.cpu()
                if extra_class == 'phone-type':
                    record[layer_idx][p, topk_indices] += 1 
                    record_n[layer_idx][p] += 1 
                elif extra_class == 'gender':
                    record[layer_idx][g*70+p, topk_indices] += 1 
                    record_n[layer_idx][g*70+p] += 1 
                elif extra_class == 'duration':
                    d = dur[k]
                    if d == -1:
                        continue 
                    record[layer_idx][d*70+p, topk_indices] += 1 
                    record_n[layer_idx][d*70+p] += 1 
                elif extra_class == 'pitch':
                    pc = check_pitch[k]
                    if pc == -1:
                        continue
                    record[layer_idx][pc*70+p, topk_indices] += 1 
                    record_n[layer_idx][pc*70+p] += 1 

    if extra_class == 'phone-type':
        N = 70
    elif extra_class == 'gender':
        N = 70*2
    elif extra_class == 'duration':
        N = 70*3
    elif extra_class == 'pitch':
        N = 70*3
        
    for idx in range(12):
        for pidx in range(N):
            if record_n[idx][pidx] != 0:
                record[idx][pidx,:] /= record_n[idx][pidx]
    
    result = torch.stack(record)
    result = np.array(result)
    np.save(save_pth, result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model-pth', help='Model path')
    parser.add_argument('-f', '--mfa-json', help='Force aligning result')  
    parser.add_argument('-s', '--save-pth', help='Result save path')
    parser.add_argument('-p', '--fp', type=int, help='Frame period', choices=[10, 20])
    parser.add_argument('-x', '--mean-std-pth', help='Mean std path')
    parser.add_argument('-d', '--data-pth', help='Dataset directory')
    parser.add_argument('-t', '--phone-type', help='Phone type', choices=['end-phone', 'mid-phone', 'start-phone'])
    parser.add_argument('-c', '--extra-class', choices=['phone-type', 'gender', 'pitch', 'duration'])
    args = parser.parse_args()
    main(**vars(args))