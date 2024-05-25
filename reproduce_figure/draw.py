import argparse
import numpy as np 
import json 
import seaborn as sns
import matplotlib.pyplot as plt 
from matplotlib_venn import venn3

sns.set()

RED = '#FF5555'
RED2 = '#FF7E7E'
RED3 = '#FFB5B5'
BLUE = '#2662FF'
GREEN = '#1BEB40'
YELLOW = '#F0F71D'

def mds_phoneme(phone_name):
    v_data_2d = np.load('data/mds-phoneme-2d.npy')
    color = [RED, BLUE, GREEN]
    label = ['vowels', 'voiced-consonants', 'unvoiced-consonants']
    num_type = [15, 15, 9]
    acc = 0 
    for idx, n in enumerate(num_type):
        plt.scatter(v_data_2d[acc:acc+n,0], v_data_2d[acc:acc+n,1], c=color[idx], label=label[idx])
        acc += n 
    for idx, name in enumerate(phone_name):
        plt.annotate(name, (v_data_2d[idx,0],v_data_2d[idx,1]), fontsize=15)
    plt.legend(loc='upper right', fontsize=13, ncol=1, framealpha=0.8, columnspacing=0.0)
    plt.axis('off')
    # plt.title(f'Layer 8')
    plt.savefig('fig/phone.png', bbox_inches='tight', dpi=200)
    plt.clf()

def mds_gender(phone_name):
    n_phone = len(phone_name)
    v_data_2d = np.load('data/mds-gender-2d.npy')
    color = [RED, BLUE]
    label = ['male', 'female']
    for idx in range(2):
        plt.scatter(v_data_2d[idx*n_phone:(idx+1)*n_phone,0], v_data_2d[idx*n_phone:(idx+1)*n_phone,1], c=color[idx], label=label[idx])
    for idx, name in enumerate(phone_name):
        plt.annotate(name, (v_data_2d[idx,0],v_data_2d[idx,1]), fontsize=15)
        plt.annotate(name, (v_data_2d[n_phone+idx,0],v_data_2d[n_phone+idx,1]), fontsize=15)
    plt.legend(loc='upper right', fontsize=13, ncol=2, framealpha=0.8, columnspacing=0.0)
    plt.axis('off')
    # plt.title(f'Layer 1')
    plt.savefig('fig/gender.png', bbox_inches='tight', dpi=200)
    plt.clf()

def mds_pitch(phone_name):
    n_phone = len(phone_name)
    v_data_2d = np.load('data/mds-pitch-2d.npy')
    color = [RED, BLUE, GREEN]
    label = ['<129.03Hz', '129.03-179.78Hz', '>179.78Hz']
    for idx in range(3):
        plt.scatter(v_data_2d[idx*n_phone:(idx+1)*n_phone,0], v_data_2d[idx*n_phone:(idx+1)*n_phone,1], c=color[idx], label=label[idx])
    for idx, name in enumerate(phone_name):
        plt.annotate(name, (v_data_2d[idx,0],v_data_2d[idx,1]), fontsize=15)
        plt.annotate(name, (v_data_2d[n_phone+idx,0],v_data_2d[n_phone+idx,1]), fontsize=15)
        plt.annotate(name, (v_data_2d[n_phone*2+idx,0],v_data_2d[n_phone*2+idx,1]), fontsize=15)
    plt.legend(loc='upper right', fontsize=13, ncol=2, framealpha=0.8, columnspacing=0.0)
    plt.axis('off')
    # plt.title(f'Layer 1')
    plt.savefig('fig/pitch.png', bbox_inches='tight', dpi=200)
    plt.clf()

def mds_duration(phone_name):
    n_phone = len(phone_name)
    v_data_2d = np.load('data/mds-duration-2d.npy')
    color = [RED, BLUE, GREEN]
    label = ['<60ms', '60-100ms', '>100ms']
    for idx in range(3):
        plt.scatter(v_data_2d[idx*n_phone:(idx+1)*n_phone,0], v_data_2d[idx*n_phone:(idx+1)*n_phone,1], c=color[idx], label=label[idx])
    for idx, name in enumerate(phone_name):
        plt.annotate(name, (v_data_2d[idx,0],v_data_2d[idx,1]), fontsize=15)
        plt.annotate(name, (v_data_2d[n_phone+idx,0],v_data_2d[n_phone+idx,1]), fontsize=15)
        plt.annotate(name, (v_data_2d[n_phone*2+idx,0],v_data_2d[n_phone*2+idx,1]), fontsize=15)
    plt.legend(loc='upper right', fontsize=13, ncol=2, framealpha=0.8, columnspacing=0.0)
    plt.axis('off')
    # plt.title(f'Layer 6')
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
        'phone-type': RED,
        'gender': BLUE,
        'pitch': GREEN,
        'duration': YELLOW
    }
    labels = {
        'phone-type': 'phoneme',
        'gender': 'gender',
        'pitch': 'pitch',
        'duration': 'duration' 
    }
    for k, v in results.items():
        if k == 'duration':
            continue
        plt.plot(range(1,13), v, label=labels[k], c=color[k], marker='o')
    plt.xticks(ticks=range(1,13))
    plt.xlabel('Layer')
    plt.ylabel('Silhouette score')
    plt.legend(fontsize=13, ncol=2, framealpha=0.9)
    plt.savefig('fig/layer-compare.png', bbox_inches='tight', dpi=200)

def model_compare():
    models_list = ['hubert_base', 'wav2vec2_base', 'wavlm_base', 'melhubert_base', 'pr', 'sid']
    properties = ['phone-type', 'gender', 'pitch']
    with open('data/model_score.json', 'r') as fp:
        model_score = json.load(fp)
    color = {
        'melhubert_base': RED,
        'hubert_base': BLUE,
        'wav2vec2_base': GREEN,
        'wavlm_base': YELLOW,
        'pr': RED2,
        'sid': RED3,
    }
    label = {
        'melhubert_base': 'MelHuBERT',
        'hubert_base': 'HuBERT',
        'wav2vec2_base': 'Wav2vec 2.0',
        'wavlm_base': 'WavLM',
        'pr': 'MelHuBERT-PR',
        'sid': 'MelHuBERT-SID',
    }
    x = np.arange(3)  # the label locations
    width = 0.06  # the width of the bars
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
        'phone-type': RED,
        'gender': BLUE,
        'pitch': GREEN,
        'duration': YELLOW
    }
    label = {
        'phone-type': 'phoneme',
        'gender': 'gender',
        'pitch': 'pitch',
        'duration': 'duration'
    }
    for k, v in layer_n_ps_keys.items():
        if k == 'duration':
            continue
        plt.plot(range(1, 12+1), v, label=label[k], color=color[k], marker='o')
    plt.legend()
    plt.xlabel('Layer')
    plt.ylabel('Num. property neurons')
    plt.savefig('fig/layer-n-compare.png', bbox_inches='tight', dpi=200)

def venn_ps_keys():
    p_name = ['phoneme', 'gender', 'pitch']
    with open('data/venn_set_sizes_layer_1.json', 'r') as fp:
        set_sizes = json.load(fp)
    venn = venn3(subsets=set_sizes, set_labels=p_name)
    plt.savefig('fig/venn-ps-keys-layer-1.png', bbox_inches='tight', dpi=200)

# def row_pruning_cmp_n_ps_keys():
#     properties = ['phone-type', 'gender']
#     data_type = ['regular', 'all-128', 'all-all']
#     data_pth = [f'data/{x}-row-pruning-n-ps-keys.json' for x in data_type]
#     rows = {
#         'regular': [512, 1024, 1536, 2048, 2560, 2688, 2816, 2944, 3072],
#         'all-128': [512, 1024, 1536, 2048, 2560, 2688, 2816, 2944, 3072],
#         'all-all': [598, 3072]
#     }
#     # Calculate density 
#     D = 3072 
#     density = {}
#     for k, v in rows.items():
#         density[k] = [i/D for i in v]

#     datas = {}
#     for k, pth in zip(data_type, data_pth):
#         with open(pth, 'r') as fp:
#             d = json.load(fp)
#         datas[k] = d

#     colors = {
#         'regular': 'red', 
#         'all-128': 'blue', 
#         'all-all': 'green'
#     }
#     labels = {
#         'regular': 'regular', 
#         'all-128': 'proposed-128', 
#         'all-all': 'proposed-all'
#     }

#     for dt in data_type:
#         gender_ps = datas[dt]['gender'][::2]
#         plt.plot(density[dt], gender_ps, color=colors[dt], marker='o', label=labels[dt])
#         # phone_ps = datas[dt]['phone-type'][::2]
#         # plt.plot(density[dt], phone_ps, color=colors[dt], marker='o', alpha=0.2)
#     plt.legend()
#     plt.xlabel('Density')
#     plt.ylabel('Num. property-specific keys')
#     plt.savefig('fig/row-pruning-cmp-n-ps-keys.png', bbox_inches='tight', dpi=200)

# def row_pruning_cmp_score():
#     properties = ['phone-type', 'gender']
#     data_type = ['regular', 'all-128', 'all-all']
#     data_pth = [f'data/{x}-row-pruning-score.json' for x in data_type]
#     rows = {
#         'regular': [512, 1024, 1536, 2048, 2560, 2688, 2816, 2944, 3072],
#         'all-128': [512, 1024, 1536, 2048, 2560, 2688, 2816, 2944, 3072],
#         'all-all': [598, 3072]
#     }
#     # Calculate density 
#     D = 3072 
#     density = {}
#     for k, v in rows.items():
#         density[k] = [i/D for i in v]

#     datas = {}
#     for k, pth in zip(data_type, data_pth):
#         with open(pth, 'r') as fp:
#             d = json.load(fp)
#         datas[k] = d

#     colors = {
#         'regular': 'red', 
#         'all-128': 'blue', 
#         'all-all': 'green'
#     }
#     labels = {
#         'regular': 'regular', 
#         'all-128': 'proposed-128', 
#         'all-all': 'proposed-all'
#     }

#     for dt in data_type:
#         gender_score = datas[dt]['gender'][::2]
#         plt.plot(density[dt], gender_score, color=colors[dt], marker='o', label=labels[dt])
#         # phone_score = datas[dt]['phone-type'][::2]
#         # plt.plot(density[dt], phone_score, color=colors[dt], marker='o', alpha=0.2)
#     plt.legend()
#     plt.xlabel('Density')
#     plt.ylabel('Silhouette score')
#     plt.savefig('fig/row-pruning-cmp-score.png', bbox_inches='tight', dpi=200)

def row_pruning_regular_n_ps_keys():
    properties = ['phone-type', 'gender', 'pitch', 'duration']
    data_pth = 'data/regular-row-pruning-n-ps-keys.json'
    # Setup x-ticks
    row = [512, 1024, 1536, 2048, 2560, 2688, 2816, 2944, 3072]
    ticks = []
    for r in row:
        ticks.append(f't{r}')
        if r != 3072:
            ticks.append(f'p{r}')

    with open(data_pth, 'r') as fp:
        v_data = json.load(fp)

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
    for k, v in v_data.items():
        plt.plot(range(len(ticks)), v, label=labels[k], color=color[k], marker='o')
        plt.xticks(ticks=range(len(ticks)), labels=ticks, rotation=90)
    plt.xlabel('Rows')
    plt.ylabel('Num. Property-Specific Keys')
    plt.legend()
    plt.savefig('fig/row-pruning-regular-n-ps-keys.png', bbox_inches='tight', dpi=200)

def row_pruning_pr():
    methods = ['regular-all', 'proposed-all']
    per = {
        'regular-128': [12.28, 8.40, 7.42, 7.23, 7.34, 8.17],
        'regular-all': [12.03, 8.17],
        'protect-128': [10.80, 8.99, 7.98, 7.42, 7.14, 8.17],
        'protect-all': [10.66, 8.17],
        'each-all': [10.95, 8.17],
    }
    rows = {
        'regular-128': [512, 1024, 1536, 2048, 2560, 3072],
        'regular-all': [577, 3072],
        'protect-128': [512, 1024, 1536, 2048, 2560, 3072],
        'protect-all': [577, 3072],
        'each-all': [434, 3072],
    }
    # Calculate density 
    D = 3072 
    density = {}
    for k, v in rows.items():
        density[k] = [i/D for i in v]

    colors = {
        'regular-all': RED,
        'protect-all': BLUE,
    }
    labels = {
        'regular': 'regular',
        'protect-all': 'protect'
    }

    for m in methods:
        plt.plot(density[m], per[m], label=labels[m], color=colors[m], marker='o')
    plt.legend()
    plt.axhline(per['regular'][-1], linestyle='--', color='black')
    plt.ylim(7, 22)
    plt.xlabel('Density')
    plt.ylabel('PER(%)')
    plt.savefig('fig/row-pruning-pr.png', bbox_inches='tight', dpi=200)

def row_pruning_sid():
    methods = ['regular-all', 'proposed-all']
    acc = {
        'regular-128': [51.04, 54.93, 59.77, 60.35, 62.63, 63.96],
        'regular-all': [50.84, 63.96],
        'protect-128': [54.10, 55.41, 59.21, 60.98, 62.71, 63.96],
        'protect-all': [58.89, 63.96],
        'each-all': [59.37, 63.96],
    }
    rows = {
        'regular-128': [512, 1024, 1536, 2048, 2560, 3072],
        'regular-all': [577, 3072],
        'protect-128': [512, 1024, 1536, 2048, 2560, 3072],
        'protect-all': [577, 3072],
        'each-all': [434, 3072],
    }
    # Calculate density 
    D = 3072 
    density = {}
    for k, v in rows.items():
        density[k] = [i/D for i in v]
    colors = {
        'regular-all': RED,
        'protect-all': BLUE,
    }
    labels = {
        'regular': 'regular',
        'protect-all': 'protect'
    }
    for m in methods:
        plt.plot(density[m], [100-i for i in acc[m]], label=labels[m], color=colors[m], marker='o')
    plt.legend()
    plt.axhline(100-acc['regular'][-1], linestyle='--', color='black')
    plt.ylim(35, 50)
    plt.xlabel('Density')
    plt.ylabel('ERR(%)')
    plt.savefig('fig/row-pruning-sid.png', bbox_inches='tight', dpi=200)

def row_pruning_bar():
    PR = {
        'topline': 2.42,
        'proposed': 10.77,
        'baseline': 11.52
    }
    SID = {
        'topline': 81.99,
        'proposed': 73.74,
        'baseline': 69.54
    }
    # Convert SID accuracy to error rate
    SID_error = {k: 100 - v for k, v in SID.items()}
    methods = list(PR.keys())
    # Plotting
    fig, ax1 = plt.subplots(figsize=(8, 6))
    # Bar width
    bar_width = 0.18
    # Index for the bar locations for PR and SID
    index_PR = np.arange(len(methods)) * 0.3  
    index_SID = np.arange(len(methods)) * 0.3 + max(index_PR) + 0.5
    # Bar plots for PR
    bars_PR = ax1.bar(index_PR, PR.values(), bar_width, label='PR', color='C0')
    ax1.set_ylabel('PER (%)')
    ax1.set_xticks(np.concatenate([index_PR, index_SID]))
    ax1.set_xticklabels(['PR ' + m for m in methods] + ['SID ' + m for m in methods])  # Explicit labels for clarity
    # Create a second y-axis for SID error rates
    ax2 = ax1.twinx()
    bars_SID = ax2.bar(index_SID, SID_error.values(), bar_width, label='SID Error Rate', color='C1')
    ax2.set_ylabel('SID ERR (%)')
    # Legend
    plt.savefig('fig/task-specific-bar.png', bbox_inches='tight', dpi=200)

def ssl_pruning():
    row_pruning_pr()
    row_pruning_sid()
    row_pruning_bar()

def match_prob():
    v_data = np.load('data/match_prob.npy')
    n_dim = v_data.shape[-1]
    n_phone = 3
    random_baseline = round(3072*0.01)/3072
    phone_name = ['AH', 'F', 'UH']
    fig, axs = plt.subplots(1, n_phone, figsize=(10,4))
    for i in range(n_phone):
        axs[i].bar(range(n_dim), v_data[i])
        axs[i].title.set_text(phone_name[i])
        axs[i].axhline(y=random_baseline, color='r', linestyle='-', linewidth=1)
    plt.savefig('fig/match_prob.png', bbox_inches='tight', dpi=200)

def generate_colors(N):
    cmap = plt.get_cmap('gist_rainbow')
    return [cmap(1.*i/N) for i in range(N)]

# def values_tsne():
#     v_data_2d = np.load('data/values-8th-layer.npy')
#     n_cluster = 5
#     n_sample = 600
#     color = generate_colors(n_cluster)
#     sum_ = 0
#     for idx in range(n_cluster):
#         cluster_emb = v_data_2d[sum_:sum_+n_sample,:]
#         plt.scatter(cluster_emb[:,0], cluster_emb[:,1], color=color[idx], label='Cluster '+str(idx))
#         sum_ += n_sample
#     plt.axis('off')
#     plt.title(f'Layer 8')
#     plt.savefig('fig/values-tsne.png', bbox_inches='tight', dpi=200)

def task_specific_pr():
    methods = ['regular', 'proposed']
    per = {
        # 'regular': [15.93, 11.52, 9.35, 8.08, 7.03, 6.17, 2.42],
        # 'proposed': [12.04, 10.77, 8.74, 7.64, 6.93, 6.56, 2.42]
        'regular': [15.10, 11.11, 8.88, 7.62, 6.82, 6.54, 4.22, 3.32, 2.85, 2.66, 2.42],
        'proposed': [13.24, 10.96, 8.67, 7.36, 6.62, 6.45, 4.14, 3.28, 2.84, 2.60, 2.42]
    }
    rows = {
        # 'regular': [28, 128, 228, 328, 428, 528, 3072],
        # 'proposed': [77, 135, 228, 328, 428, 528, 3072],
        'regular': [12, 112, 212, 312, 412, 512, 1024, 1536, 2048, 2560, 3072],
        'proposed': [32, 112, 212, 312, 412, 512, 1024, 1536, 2048, 2560, 3072],
    }
    # Calculate density 
    D = 3072 
    density = {}
    for k, v in rows.items():
        density[k] = [i/D for i in v]
    colors = {
        'regular': 'C0',
        'proposed': 'C1',
    }
    labels = {
        'regular': 'regular',
        'proposed': 'proposed',
    }
    for m in methods:
        plt.plot(density[m], per[m], label=labels[m], color=colors[m], marker='o')
    plt.legend()
    plt.axhline(per['regular'][-1], linestyle='--', color='black')
    plt.ylim(2, 16)
    plt.xlabel('Density')
    plt.ylabel('PER(%)')
    plt.savefig('fig/task-specific-pr.png', bbox_inches='tight', dpi=200)
    plt.clf()

def task_specific_sid():
    methods = ['regular', 'proposed']
    acc = {
        # 'regular': [69.54, 81.99],
        # 'proposed': [73.74, 81.99],
        'regular': [72.40, 74.68, 77.33, 79.11, 80.64, 81.99],
        'proposed': [74.10, 76.70, 78.43, 80.07, 81.60, 81.99]
    }
    rows = {
        # 'regular': [169, 3072],
        # 'proposed': [169, 3072],
        'regular': [512, 1024, 1536, 2048, 2560, 3072],
        'proposed': [512, 1024, 1536, 2048, 2560, 3072]
    }
    # Calculate density 
    D = 3072 
    density = {}
    for k, v in rows.items():
        density[k] = [i/D for i in v]
    colors = {
        'regular': 'C0',
        'proposed': 'C1',
    }
    labels = {
        'regular': 'regular',
        'proposed': 'proposed',
    }
    for m in methods:
        plt.plot(density[m], [100-i for i in acc[m]], label=labels[m], color=colors[m], marker='o')
    plt.legend()
    plt.axhline(100-acc['regular'][-1], linestyle='--', color='black')
    plt.ylim(17, 32)
    plt.xlabel('Density')
    plt.ylabel('ERR(%)')
    plt.savefig('fig/task-specific-sid.png', bbox_inches='tight', dpi=200)
    plt.clf()

def task_specific_bar():
    # Data for the performance of methods
    PR = { # about 135 rows remained
        'unpruned': 2.42,
        'proposed': 10.77,
        'baseline': 11.52
    }
    SID = { # about 169 rows remained
        'unpruned': 81.99,
        'proposed': 73.74,
        'baseline': 69.54
    }
    # Convert SID accuracy to error rate
    SID_error = {k: 100 - v for k, v in SID.items()}
    methods = list(PR.keys())
    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))
    # Bar width
    bar_width = 0.18
    # Index for the bar locations for PR and SID
    index_PR = np.arange(len(methods)) * 0.3  
    index_SID = np.arange(len(methods)) * 0.3 + max(index_PR) + 0.4
    # Bar plots for PR
    bars_PR = ax1.bar(index_PR, PR.values(), bar_width, label='PR', color='C0')
    ax1.set_ylabel('PER (%)')
    ax1.set_xticks(np.concatenate([index_PR, index_SID]))
    ax1.set_xticklabels([f'PR {methods[0]}']+[m for m in methods[1:]] + [f'SID {methods[0]}'] + [m for m in methods[1:]], fontsize=14)  # Explicit labels for clarity
    ax1.grid(zorder=0)
    # Create a second y-axis for SID error rates
    ax2 = ax1.twinx()
    bars_SID = ax2.bar(index_SID, SID_error.values(), bar_width, label='SID Error Rate', color='C1')
    ax2.set_ylabel('SID ERR (%)')
    ax2.grid(visible=False)
    # Legend
    plt.savefig('fig/task-specific-bar.png', bbox_inches='tight', dpi=200)

def task_specific_pruning():
    task_specific_pr()
    task_specific_sid()
    task_specific_bar()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', 
        choices=['mds_results', 'layer_compare', 
                'model_compare', 'layer_n_ps_compare',
                'venn_ps_keys', 'row_pruning_regular_n_ps_keys',
                'row_pruning_pr', 'row_pruning_sid', 'match_prob',
                'task_specific_pruning']
            ,help='Mode of drawing figure')
    args = parser.parse_args()
    eval(args.mode)()