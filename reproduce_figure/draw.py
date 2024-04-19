import argparse
import numpy as np 
import json 
import matplotlib.pyplot as plt 
from matplotlib_venn import venn3

def mds_phoneme(phone_name):
    v_data_2d = np.load('data/mds-phoneme-2d.npy')
    color = ['red', 'blue', 'green']
    label = ['vowels', 'voiced-consonants', 'unvoiced-consonants']
    num_type = [15, 15, 9]
    acc = 0 
    for idx, n in enumerate(num_type):
        plt.scatter(v_data_2d[acc:acc+n,0], v_data_2d[acc:acc+n,1], c=color[idx], label=label[idx])
        acc += n 
    for idx, name in enumerate(phone_name):
        plt.annotate(name, (v_data_2d[idx,0],v_data_2d[idx,1]))
    plt.legend(loc='upper right', fontsize=6)
    plt.axis('off')
    plt.title(f'Layer 8')
    plt.savefig('fig/phone.png', bbox_inches='tight', dpi=200)
    plt.clf()

def mds_gender(phone_name):
    n_phone = len(phone_name)
    v_data_2d = np.load('data/mds-gender-2d.npy')
    color = ['red', 'blue']
    label = ['male', 'female']
    for idx in range(2):
        plt.scatter(v_data_2d[idx*n_phone:(idx+1)*n_phone,0], v_data_2d[idx*n_phone:(idx+1)*n_phone,1], c=color[idx], label=label[idx])
    for idx, name in enumerate(phone_name):
        plt.annotate(name, (v_data_2d[idx,0],v_data_2d[idx,1]))
        plt.annotate(name, (v_data_2d[n_phone+idx,0],v_data_2d[n_phone+idx,1]))
    plt.legend(loc='upper right', fontsize=6)
    plt.axis('off')
    plt.title(f'Layer 1')
    plt.savefig('fig/gender.png', bbox_inches='tight', dpi=200)
    plt.clf()

def mds_pitch(phone_name):
    n_phone = len(phone_name)
    v_data_2d = np.load('data/mds-pitch-2d.npy')
    color = ['red', 'blue', 'green']
    label = ['<129.03Hz', '129.03-179.78Hz', '>179.78Hz']
    for idx in range(3):
        plt.scatter(v_data_2d[idx*n_phone:(idx+1)*n_phone,0], v_data_2d[idx*n_phone:(idx+1)*n_phone,1], c=color[idx], label=label[idx])
    for idx, name in enumerate(phone_name):
        plt.annotate(name, (v_data_2d[idx,0],v_data_2d[idx,1]))
        plt.annotate(name, (v_data_2d[n_phone+idx,0],v_data_2d[n_phone+idx,1]))
        plt.annotate(name, (v_data_2d[n_phone*2+idx,0],v_data_2d[n_phone*2+idx,1]))
    plt.legend(loc='upper right', fontsize=6)
    plt.axis('off')
    plt.title(f'Layer 1')
    plt.savefig('fig/pitch.png', bbox_inches='tight', dpi=200)
    plt.clf()

def mds_duration(phone_name):
    n_phone = len(phone_name)
    v_data_2d = np.load('data/mds-duration-2d.npy')
    color = ['red', 'blue', 'green']
    label = ['<60ms', '60-100ms', '>100ms']
    for idx in range(3):
        plt.scatter(v_data_2d[idx*n_phone:(idx+1)*n_phone,0], v_data_2d[idx*n_phone:(idx+1)*n_phone,1], c=color[idx], label=label[idx])
    for idx, name in enumerate(phone_name):
        plt.annotate(name, (v_data_2d[idx,0],v_data_2d[idx,1]))
        plt.annotate(name, (v_data_2d[n_phone+idx,0],v_data_2d[n_phone+idx,1]))
        plt.annotate(name, (v_data_2d[n_phone*2+idx,0],v_data_2d[n_phone*2+idx,1]))
    plt.legend(loc='upper right', fontsize=6)
    plt.axis('off')
    plt.title(f'Layer 6')
    plt.savefig('fig/duration.png', bbox_inches='tight', dpi=200)
    plt.clf()

def mds_results():
    phone_name = ['AH', 'IH', 'IY', 'EH', 'ER', 'AE', 'AY', 'EY', 'AO', 'AA', 'OW', 'UW', 'AW', 'UH', 'OY', 'N', 'D', 'R', 'L', 'DH', 'M', 'Z', 'W', 'V', 'B', 'NG', 'G', 'Y', 'JH', 'ZH', 'T', 'S', 'K', 'HH', 'F', 'P', 'SH', 'TH', 'CH']
    properties = ['phoneme', 'gender', 'pitch', 'duration']
    for p in properties:
        eval(f"mds_{p}")(phone_name)

def layer_compare():
    with open('data/layer_score.json', 'r') as fp:
        results = json.load(fp)
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
    plt.savefig('fig/layer-compare.png', bbox_inches='tight', dpi=200)

def model_compare():
    models_list = ['melhubert_base', 'hubert_base', 'wav2vec2_base', 'wavlm_base']
    properties = ['phone-type', 'gender', 'pitch']
    with open('data/model_score.json', 'r') as fp:
        model_score = json.load(fp)
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
    plt.savefig('fig/models-compare.png', dpi=200, bbox_inches='tight')

def layer_n_ps_compare():
    with open('data/layer_n_ps_keys.json', 'r') as fp:
        layer_n_ps_keys = json.load(fp)
    color = {
        'phone-type': 'red',
        'gender': 'blue',
        'pitch': 'green',
        'duration': 'black'
    }
    label = {
        'phone-type': 'phoneme',
        'gender': 'gender',
        'pitch': 'pitch',
        'duration': 'duration'
    }
    for k, v in layer_n_ps_keys.items():
        plt.plot(range(1, 12+1), v, label=label[k], color=color[k], marker='o')
    plt.legend()
    plt.xlabel('Layer')
    plt.ylabel('Num. property-specific keys')
    plt.savefig('fig/layer-n-compare.png', bbox_inches='tight', dpi=200)

def venn_ps_keys():
    p_name = ['phoneme', 'gender', 'pitch']
    with open('data/venn_set_sizes_layer_1.json', 'r') as fp:
        set_sizes = json.load(fp)
    venn = venn3(subsets=set_sizes, set_labels=p_name)
    plt.savefig('fig/venn-ps-keys-layer-1.png', bbox_inches='tight', dpi=200)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', 
        choices=['mds_results', 'layer_compare', 
                'model_compare', 'layer_n_ps_compare',
                'venn_ps_keys']
            ,help='Mode of drawing figure')
    args = parser.parse_args()
    eval(args.mode)()