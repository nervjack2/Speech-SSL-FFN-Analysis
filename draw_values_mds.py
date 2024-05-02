import argparse
import pickle
import os 
import random
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import MDS, TSNE
from sklearn.decomposition import PCA

def generate_colors(N):
    cmap = plt.get_cmap('gist_rainbow')
    return [cmap(1.*i/N) for i in range(N)]

def draw_value(embs, labels, v_idx, v_count, save_dir, n_sample, reduce_method, layer):
    n_cluster = len(v_idx)
    v_emb = [[] for i in range(n_cluster)]
    for i, (e, l) in enumerate(zip(embs, labels)):
        for idx in range(n_cluster):
            if l == v_idx[idx]:
                v_emb[idx].append(e)
        print(f"{i}/{len(embs)}", end='')
    v_emb_cat = []
    for idx in range(n_cluster):
        e = random.choices(v_emb[idx], k=n_sample)
        v_emb_cat += e
    v_data = np.stack(v_emb_cat, axis=0)
    print("Start doing dimension reduction")
    if reduce_method == 'mds':
        mds = MDS(n_components=2, random_state=0)
        v_data_2d = mds.fit_transform(v_data)
    elif reduce_method == 'tsne':
        tsne = TSNE(n_components=2, random_state=0)
        v_data_2d = tsne.fit_transform(v_data)
    elif reduce_method == 'pca':
        pca = PCA(n_components=2, random_state=0)
        v_data_2d = pca.fit_transform(v_data)
    color = generate_colors(n_cluster)
    sum_ = 0
    for idx in range(n_cluster):
        cluster_emb = v_data_2d[sum_:sum_+n_sample,:]
        plt.scatter(cluster_emb[:,0], cluster_emb[:,1], color=color[idx], label='Cluster '+str(idx))
        sum_ += n_sample
    plt.axis('off')
    plt.title(f'Layer {layer}')
    save_pth = os.path.join(save_dir, f'layer-{layer}.png')
    plt.savefig(save_pth, bbox_inches='tight', dpi=200)
    plt.clf()

def main(value_pkl, reduce_method, n_cluster, n_sample, n_layer, save_dir):
    with open(value_pkl, 'rb') as fp:
        data = pickle.load(fp)
    # Raw data 
    labels_count = data['labels_count']
    labels = data['labels']
    embs = data['embs']
    # Select first N most frequently appear clusters
    v_idx = np.argsort(np.negative(labels_count))[:n_cluster]
    v_count = [labels_count[idx] for idx in v_idx]
    print('Index of clusters: ', v_idx)
    print('Number of occurance of clusters: ', v_count)
  
    draw_value(embs[n_layer], labels, v_idx, v_count, save_dir, n_sample, reduce_method, layer=n_layer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--value-pkl', help='Value pkl path')
    parser.add_argument('--reduce-method', help='Dimension reducing method')
    parser.add_argument('--n-cluster', help='Number of cluster for visualize', type=int)
    parser.add_argument('--n-sample', help='Number of visualized samples for each cluster', type=int)    
    parser.add_argument('--n-layer', help='Number of layer', type=int)
    parser.add_argument('--save-dir', help='Save directory', type=str)
    args = parser.parse_args()
    main(**vars(args))
