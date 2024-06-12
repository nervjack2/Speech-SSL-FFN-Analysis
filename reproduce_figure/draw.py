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
    phone_name_pitch = ['AH', 'IH', 'IY', 'EH', 'ER', 'AE', 'AY', 'EY', 'AO', 'AA', 'OW', 'UW', 'AW', 'UH', 'OY', 'N', 'D', 'R', 'L', 'DH', 'M', 'Z', 'W', 'V', 'B', 'NG', 'G', 'Y', 'JH', 'ZH']
    properties = ['phoneme', 'gender', 'pitch', 'duration']
    for p in properties:
        if p != 'pitch':
            eval(f"mds_{p}")(phone_name)
        else:
            eval(f"mds_{p}")(phone_name_pitch)

def layer_compare():
    with open('data/layer_score.json', 'r') as fp:
        results = json.load(fp)
    color = {
        'phone-type': 'C0',
        'gender': 'C1',
        'pitch': 'C2',
        'duration': 'C3'
    }
    labels = {
        'phone-type': 'phoneme',
        'gender': 'gender',
        'pitch': 'pitch',
        'duration': 'duration' 
    }
    plt.figure(figsize=(6,4))
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
        'hubert_base': 'C0',
        'wav2vec2_base': 'C1',
        'wavlm_base': 'C2',
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
    fig, ax = plt.subplots(layout='constrained', figsize=(6,3))
    for m in models_list:
        offset = width * multiplier
        ax.bar(x*0.5 + offset, model_score[m], width, label=label[m], color=color[m])
        multiplier += 1
    xticks = ['phoneme', 'gender', 'pitch']
    ax.set_ylabel('Silhouette score', fontsize=15)
    ax.set_xticks(x*0.5 + width*2.5, xticks, fontsize=13)
    ax.legend(loc='upper left')
    plt.savefig('fig/models-compare.png', dpi=200, bbox_inches='tight')

def layer_n_ps_compare():
    with open('data/layer_n_ps_keys.json', 'r') as fp:
        layer_n_ps_keys = json.load(fp)
    color = {
        'phone-type': 'C0',
        'gender': 'C1',
        'pitch': 'C2',
        'duration': 'C3'
    }
    label = {
        'phone-type': 'phoneme',
        'gender': 'gender',
        'pitch': 'pitch',
        'duration': 'duration'
    }
    plt.figure(figsize=(6,4))
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
    venn.get_patch_by_id('100').set_color('C0')  
    venn.get_patch_by_id('010').set_color('C1')
    venn.get_patch_by_id('001').set_color('C2') 
    plt.savefig('fig/venn-ps-keys-layer-1.png', bbox_inches='tight', dpi=200)

def row_pruning_pr():
    methods = ['regular-all', 'protect-all']
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
        'regular-all': 'regular',
        'protect-all': 'protect'
    }

    for m in methods:
        plt.plot(density[m], per[m], label=labels[m], color=colors[m], marker='o')
    plt.legend()
    plt.axhline(per['regular-all'][-1], linestyle='--', color='black')
    plt.ylim(7, 22)
    plt.xlabel('Density')
    plt.ylabel('PER(%)')
    plt.savefig('fig/row-pruning-pr.png', bbox_inches='tight', dpi=200)

def row_pruning_sid():
    methods = ['regular-all', 'protect-all']
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
        'regular-all': 'regular',
        'protect-all': 'protect'
    }
    for m in methods:
        plt.plot(density[m], [100-i for i in acc[m]], label=labels[m], color=colors[m], marker='o')
    plt.legend()
    plt.axhline(100-acc['regular-all'][-1], linestyle='--', color='black')
    plt.ylim(35, 50)
    plt.xlabel('Density')
    plt.ylabel('ERR(%)')
    plt.savefig('fig/row-pruning-sid.png', bbox_inches='tight', dpi=200)

def row_pruning_bar():
    PR = {
        'unpruned': 8.17,
        'protect ': 10.66,
        'baseline': 12.03,
    }
    SID = {
        'unpruned': 63.96,
        'protect': 58.89,
        'baseline': 50.84,
    }
    logF0 = {
        'unpruned': 0.296,
        'protect': 0.272,
        'baseline': 0.282,
    }

    # Convert SID accuracy to error rate
    SID_error = {k: 100 - v for k, v in SID.items()}
    methods = list(PR.keys())
    # Plotting
    fig, ax1 = plt.subplots(figsize=(12, 6))
    # Bar width
    bar_width = 0.18
    # Index for the bar locations for PR and SID
    index_PR = np.arange(len(methods)) * 0.3  
    index_SID = np.arange(len(methods)) * 0.3 + max(index_PR) + 0.4
    index_f0 = np.arange(len(methods)) * 0.3 + max(index_SID) + 0.4
    # Bar plots for PR
    bars_PR = ax1.bar(index_PR, PR.values(), bar_width, label='PR', color='C0')
    ax1.set_ylabel('PER (%)', c='C0')
    ax1.set_xticks(np.concatenate([index_PR, index_SID, index_f0]))
    ax1.tick_params(axis='y', colors='C0')
    ax1.set_ylim(6, 13)
    ax1.set_xticklabels(methods*3, fontsize=14)  # Explicit labels for clarity
    ax1.legend(fontsize=15, loc='upper left', framealpha=0.8)
    # Create a second y-axis for SID error rates
    ax2 = ax1.twinx()
    bars_SID = ax2.bar(index_SID, SID_error.values(), bar_width, label='SID', color='C1')
    ax2.set_ylabel('SID ERR (%)', c='C1')
    ax2.grid(visible=False)
    ax2.set_ylim(34, 50)
    ax2.tick_params(axis='y', colors='C1')
    ax2.legend(fontsize=15, loc='upper left', bbox_to_anchor=(0,0.9), framealpha=0.8)
    # Create a third y-axis for log f0 reconstruction mean square error 
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 40))
    ax1.spines['bottom'].set_patch_line()
    bars_f0 = ax3.bar(index_f0, logF0.values(), bar_width, label='log F0', color='C2')
    ax3.set_ylabel('MSE', c='C2')
    ax3.tick_params(axis='y', colors='C2')
    ax3.set_ylim(0.265, 0.300)
    ax3.grid(visible=False)
    ax3.legend(fontsize=15, loc='upper left', bbox_to_anchor=(0,0.8), framealpha=0.5)

    # Function to attach a text label above each bar displaying its height
    def autolabel(rects, ax, round_p=1):
        for rect in rects:
            height = round(rect.get_height(), round_p)
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
                        
    autolabel(bars_PR, ax1)
    autolabel(bars_SID, ax2)
    autolabel(bars_f0, ax3, round_p=3)
    plt.savefig('fig/pruning-bar.png', bbox_inches='tight', dpi=200)

def ssl_pruning():
    # row_pruning_pr()
    # row_pruning_sid()
    row_pruning_bar()

def match_prob():
    v_data = np.load('data/match_prob.npy')
    n_dim = v_data.shape[-1]
    random_baseline = round(3072*0.01)/3072
    plt.figure(figsize=(6,4))
    plt.bar(range(n_dim), v_data[0], color='C0', linewidth=0)
    plt.axhline(y=random_baseline, color=RED, linestyle='-', linewidth=2)
    plt.text(n_dim*0.7, random_baseline+0.02, 'random baseline', color=RED, fontweight='bold')
    plt.savefig('fig/match_prob.png', bbox_inches='tight', dpi=200)

def generate_colors(N):
    cmap = plt.get_cmap('gist_rainbow')
    return [cmap(1.*i/N) for i in range(N)]

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
    ax1.set_ylabel('PER (%)', c='C0')
    ax1.set_xticks(np.concatenate([index_PR, index_SID]))
    ax1.tick_params(axis='y', colors='C0')
    ax1.set_xticklabels(methods + methods, fontsize=14)  # Explicit labels for clarity
    ax1.grid(zorder=0)
    ax1.legend(fontsize=15, loc='upper left', framealpha=0.8)
    # Create a second y-axis for SID error rates
    ax2 = ax1.twinx()
    bars_SID = ax2.bar(index_SID, SID_error.values(), bar_width, label='SID', color='C1')
    ax2.set_ylabel('SID ERR (%)', c='C1')
    ax2.tick_params(axis='y', colors='C1')
    ax2.grid(visible=False)
    ax2.set_ylim(12, 32)
    ax2.legend(fontsize=15, loc='upper left', bbox_to_anchor=(0,0.9), framealpha=0.8)
    # Function to attach a text label above each bar displaying its height
    def autolabel(rects, ax):
        for rect in rects:
            height = round(rect.get_height(), 1)
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    autolabel(bars_PR, ax1)
    autolabel(bars_SID, ax2)
    plt.savefig('fig/task-specific-bar.png', bbox_inches='tight', dpi=200)

def task_specific_pruning():
    # task_specific_pr()
    # task_specific_sid()
    task_specific_bar()

def erase_gender_info():
    # Data setup
    groups = ['Erase Male', 'Erase Female']
    male_delta_errors = [18.58, 2.24]
    female_delta_errors = [4.1, 22.43]

    # Set up for bars
    y = [i*0.45 for i in range(len(groups))]  # label locations
    height = 0.18  # height of the bars

    fig, ax = plt.subplots(figsize=(8, 4))
    rects1 = ax.barh([p - height for p in y], male_delta_errors, height, label='Male', color='C0')
    rects2 = ax.barh(y, female_delta_errors, height, label='Female', color='C1')

    # Add some text for labels, title and custom y-axis tick labels, etc.
    ax.set_xlabel('Δ SID Error Rate (%)', fontsize=15)
    ax.set_yticks([p  - height/2 for p in y])
    ax.set_yticklabels(groups, fontsize=15)
    ax.legend(fontsize=15)

    # Function to attach a text label next to each bar displaying its width
    def autolabel(rects):
        for rect in rects:
            width = rect.get_width()
            ax.annotate('{}'.format(width),
                        xy=(width, rect.get_y() + rect.get_height() / 2),
                        xytext=(3, 0),  # 3 points horizontal offset
                        textcoords="offset points",
                        ha='left', va='center')

    # Attach labels to the bars
    autolabel(rects1)
    autolabel(rects2)
    plt.savefig('fig/erase-gender.png', bbox_inches='tight', dpi=200)
   
def erase_phone_type():
    # Data setup
    groups = ['Erase Vowels', 'Erase Voiced-Con.', 'Erase Unvoiced-Con.']
    vowels_delta_errors = [0.88, 0.05, 0.00]
    voiced_con_delta_errors = [-0.08, 0.25, -0.02]
    unvoiced_con_delta_errors = [-0.24, 0.04, 0.54]

    # Set up for bars
    y = [i*0.60 for i in range(len(groups))]  # label locations
    height = 0.18  # height of the bars

    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.barh([p - height*2 for p in y], vowels_delta_errors, height, label='Vowels', color='C0')
    rects2 = ax.barh([p - height for p in y], voiced_con_delta_errors, height, label='Voiced-Con.', color='C1')
    rects3 = ax.barh(y, unvoiced_con_delta_errors, height, label='Unvoiced-Con.', color='C2')

    # Add some text for labels, title and custom y-axis tick labels, etc.
    ax.set_xlabel('Δ Phoneme CLS Error Rate (%)', fontsize=15)
    ax.set_yticks([p  - height/2 for p in y])
    ax.set_yticklabels(groups, fontsize=15)
    ax.legend(fontsize=15)

    # Function to attach a text label next to each bar displaying its width
    def autolabel(rects):
        for rect in rects:
            width = rect.get_width()
            ax.annotate('{}'.format(width),
                        xy=(width, rect.get_y() + rect.get_height() / 2),
                        xytext=(3, 0),  # 3 points horizontal offset
                        textcoords="offset points",
                        ha='left', va='center')

    # Attach labels to the bars
    autolabel(rects1)
    autolabel(rects2)
    plt.savefig('fig/erase-phoneme-type.png', bbox_inches='tight', dpi=200)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', 
        choices=['mds_results', 'layer_compare', 
                'model_compare', 'layer_n_ps_compare',
                'venn_ps_keys', 'match_prob',
                'task_specific_pruning', 'ssl_pruning', 
                'erase_gender_info', 'erase_phone_type']
            ,help='Mode of drawing figure')
    args = parser.parse_args()
    eval(args.mode)()