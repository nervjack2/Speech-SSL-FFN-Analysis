import argparse
import torch 
import torch.nn.functional as F
import glob
import json
import pickle
import os 
import matplotlib.pyplot as plt
from tqdm import tqdm

def main(value_pkl, reduce_dim):
    with open(value_pkl, 'rb') as fp:
        data = pickle.load(fp)
    print(data.keys())
    # v_idx = np.argsort(np.negative(labels_count))[:3]
    # v_count = [labels_count[idx] for idx in v_idx]
    # print(v_idx, v_count)
    # for layer_idx in range(8,9):
    #     v_emb = [[] for i in range(3)]
    #     for e, l in zip(embs[layer_idx], labels):
    #         if l == v_idx[0]:
    #             v_emb[0].append(e)
    #         elif l == v_idx[1]:
    #             v_emb[1].append(e)
    #         elif l == v_idx[2]:
    #             v_emb[2].append(e)
    #     v_data = np.stack(v_emb[0]+v_emb[1]+v_emb[2], axis=0)
    #     # Multiscale scaling
    #     if dim_reduce == 'mds':
    #         mds = MDS(n_components=2, random_state=0)
    #         v_data_2d = mds.fit_transform(v_data)
    #     elif dim_reduce == 'tsne':

    #     color = ['red', 'blue', 'green']
    #     sum_ = 0
    #     for idx in range(3):
    #         plt.scatter(v_data_2d[sum_:sum_+v_count[idx],0], v_data_2d[sum_:sum_+v_count[idx],1], c=color[idx], label='Cluster '+str(idx))
    #         sum_ += v_count[idx]
    #     plt.axis('off')
    #     plt.title(f'Layer {layer_idx}')
    #     save_pth = os.path.join(save_dir, f'layer-{layer_idx}.png')
    #     plt.savefig(save_pth, bbox_inches='tight', dpi=200)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--value-pkl', help='Value pkl path')
    parser.add_argument('--reduce-dim', help='Dimension reducing method')  
    args = parser.parse_args()
    main(**vars(args))
