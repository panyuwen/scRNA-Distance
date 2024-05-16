import torch
import scanpy as sc
import os
import pickle
import time
import pandas as pd
import numpy as np
import yaml
import argparse
import sys
import socket
from tqdm import tqdm

def median_heuristic(x, y, adjustment_factor, device):
    combined = torch.cat([x, y], dim=0)
    dists = torch.cdist(combined, combined, p=2)
    
    triu_indices = torch.triu_indices(dists.size(0), dists.size(1), offset=1).to(device)
    upper_triu_dists = dists[triu_indices[0], triu_indices[1]]
    
    median_dist = upper_triu_dists.median()
    
    return median_dist / adjustment_factor


def gaussian_kernel_sum(x, y, sigma):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    
    x = x.unsqueeze(1)  # Shape: (x_size, 1, dim)
    y = y.unsqueeze(0)  # Shape: (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    
    kernel_matrix = torch.exp(-1.0 * torch.sum((tiled_x - tiled_y) ** 2, dim=2) / (2 * sigma ** 2))
    
    return kernel_matrix


def gaussian_kernel_mean(x, y, sigma):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    
    x = x.unsqueeze(1)  # Shape: (x_size, 1, dim)
    y = y.unsqueeze(0)  # Shape: (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    
    kernel_matrix = torch.exp(-1.0 * torch.mean((tiled_x - tiled_y) ** 2, dim=2) / (2 * sigma ** 2))
    
    return kernel_matrix


def mmd_square(x, y, sigma, gaussian_kernel_type):
    if gaussian_kernel_type == 'mean':
        x_kernel = gaussian_kernel_mean(x, x, sigma)
        y_kernel = gaussian_kernel_mean(y, y, sigma)
        xy_kernel = gaussian_kernel_mean(x, y, sigma)
    else:
        x_kernel = gaussian_kernel_sum(x, x, sigma)
        y_kernel = gaussian_kernel_sum(y, y, sigma)
        xy_kernel = gaussian_kernel_sum(x, y, sigma)
        
    m, n = x.size(0), y.size(0)
    
    xx_sum = x_kernel.sum() / (m * m)
    yy_sum = y_kernel.sum() / (n * n)
    xy_sum = xy_kernel.sum() / (m * n)
    
    return xx_sum + yy_sum - 2 * xy_sum


def one_pair(adata1, adata2, gaussian_kernel_type='sum', min_cells=1, adjustment_factor=2.0, resamplesize=15, n_iter=1000):
    x = torch.tensor(adata1.X.toarray())  ## cell x gene
    y = torch.tensor(adata2.X.toarray())  ## cell x gene

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, y = x.to(device), y.to(device)
    
    if ((x.shape[0] < 12) and (y.shape[0] < 12)) or ((x.shape[0] < 7) or (y.shape[0] < 7)):
        return [], 0, 0, 0
    
    if x.shape[0] < resamplesize:
        resamplesize1 = int(x.shape[0] * 0.9)
    else:
        resamplesize1 = resamplesize
    
    if y.shape[0] < resamplesize:
        resamplesize2 = int(y.shape[0] * 0.9)
    else:
        resamplesize2 = resamplesize

    
    mmdlist, genecount = [], []
    for _ in range(n_iter):
        indices_x = torch.randperm(len(x))[:resamplesize1].to(device)
        x_subsamples = x[indices_x]
        
        indices_y = torch.randperm(len(y))[:resamplesize2].to(device)
        y_subsamples = y[indices_y]

        if gaussian_kernel_type == 'mean':
            gene2remain = ((x_subsamples > 0).sum(axis=0) >= min_cells) | ((y_subsamples > 0).sum(axis=0) >= min_cells)
            x_subsamples = x_subsamples[:,gene2remain]
            y_subsamples = y_subsamples[:,gene2remain]
            genecount.append(gene2remain.sum().item())
        else:
            genecount.append(x_subsamples.shape[1])

        # x_subsamples, y_subsamples = x_subsamples.to(device), y_subsamples.to(device)

        sigma_value = median_heuristic(x_subsamples, y_subsamples, adjustment_factor, device)
        mmd_value = mmd_square(x_subsamples, y_subsamples, sigma_value, gaussian_kernel_type)
        
        mmdlist.append(mmd_value.item())
    
    return mmdlist, resamplesize1, resamplesize2, int(np.median(genecount))


def get_result(h5ad_filename, yaml_filename, gaussian_kernel_type, adjustment_factor):
    adata = sc.read(h5ad_filename)
    # sc.pp.filter_genes(adata, min_cells=50)
    # adata.obs['mainCellType'] = adata.obs['cellType'].apply(lambda x: x.split('_')[0])
    meta = adata.obs

    with open(yaml_filename, 'r') as f:
        pairinfo = yaml.safe_load(f)

    """
    format for pairinfo
    XX:
    g1:
        cellType:
        - proNeu_Cd34
        stim:
        - BM1Y
    g2:
        cellType:
        - proNeu_Cd34
        stim:
        - BM1Y
    """

    result = {}
    index = 1

    for _, v in tqdm(pairinfo.items(), leave=True):
        g1, g2 = list(v.keys())
        g1 = v[g1]
        g2 = v[g2]

        meta['group1_cell2keep'] = True
        for n, m in g1.items():
            meta['group1_cell2keep'] = meta[n].isin(m) & meta['group1_cell2keep']

        meta['group2_cell2keep'] = True
        for n, m in g2.items():
            meta['group2_cell2keep'] = meta[n].isin(m) & meta['group2_cell2keep']

        adata1 = adata[meta['group1_cell2keep']]
        adata2 = adata[meta['group2_cell2keep']]

        output = one_pair(adata1, adata2, gaussian_kernel_type=gaussian_kernel_type, adjustment_factor=adjustment_factor)
        g1['resamplesize'] = output[1]
        g2['resamplesize'] = output[2]
        output = {'mmdlist': output[0], 'genecount': output[3], 'para1': g1, 'para2': g2}

        result[index] = output

        index += 1
    
    return result


def main():
    parser = argparse.ArgumentParser(description='pairwise MMD, take h5ad as input')
    parser.add_argument("--h5ad", type=str, required = True, \
                        help="h5ad file")
    parser.add_argument("--yaml", type=str, required = True, \
                        help="yaml file")
    parser.add_argument("--kernel", type=str, required = False, default='sum', choices=['sum','mean'], \
                        help="mean: filter genes & other adjustment_factor; sum: default process")
    parser.add_argument("--factor", type=float, required = False, default=2.0, \
                        help="adjustment_factor for mean kernel, ~sqrt(gene count)")
    parser.add_argument("--out", type=str, required = False, default='output', \
                        help="output prefix")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.out+'.logfile', 'w') as log:
        log.write('{}\n'.format(sys.argv[0]))
        log.write('{}--h5ad   {}\n'.format(' '*4, args.h5ad))
        log.write('{}--yaml   {}\n'.format(' '*4, args.yaml))
        log.write('{}--kernel {}\n'.format(' '*4, args.kernel))
        log.write('{}--factor {}\n'.format(' '*4, args.factor))
        log.write('{}--out    {}\n\n'.format(' '*4, args.out))
        
        log.write('Hostname: '+socket.gethostname()+'\n')
        log.write('Working directory: '+os.getcwd()+'\n')
        log.write('Device: '+str(device)+'\n')
        log.write('Start time: '+time.strftime("%Y-%m-%d %X",time.localtime())+'\n\n')

    result = get_result(args.h5ad, args.yaml, args.kernel, args.factor)

    with open(args.out + '.pickle', 'wb') as fout:
        pickle.dump(result, fout)

    # with open(output_prefix + '.pickle', 'rb') as fin:
    #     result = pickle.load(fin)

    with open(args.out+'.logfile','a') as log:
        log.write("Output "+args.out+'.pickle\n')
        log.write('End time: '+time.strftime("%Y-%m-%d %X",time.localtime())+'\n\n')


if __name__ == '__main__':
    main()

