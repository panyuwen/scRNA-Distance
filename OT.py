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
import geomloss


def cost_func(a, b, p=2, metric='euclidean_mean'):
    """ 
    a, b in shape: (B, N, D) or (N, D)
    """ 
    # assert type(a)==torch.Tensor and type(b)==torch.Tensor, 'inputs should be torch.Tensor'
    
    if metric=='euclidean' and p==1:
        return geomloss.utils.distances(a, b)
    elif metric=='euclidean_mean' and p==2:
        return geomloss.utils.squared_distances(a, b) / a.shape[1]
    else:
        return geomloss.utils.squared_distances(a, b)


def one_pair(adata1, adata2, OTLoss, cost, min_cells=1, resamplesize=15, n_iter=100):
    x = torch.tensor(adata1.X.toarray())  ## cell x gene
    y = torch.tensor(adata2.X.toarray())  ## cell x gene

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, y = x.to(device), y.to(device)

    if (((x.shape[0] < 1) or (y.shape[0] < 1)) or ((x.shape[0] <= 2) and (y.shape[0] <= 2))):
        return [], 0, 0, 0

    if x.shape[0] < resamplesize:
        resamplesize1 = int(x.shape[0] * 0.8) + 1
    else:
        resamplesize1 = resamplesize

    if y.shape[0] < resamplesize:
        resamplesize2 = int(y.shape[0] * 0.8) + 1
    else:
        resamplesize2 = resamplesize

    otdlist, genecount = [], []
    for _ in range(n_iter):
        indices_x = torch.randperm(len(x))[:resamplesize1].to(device)
        x_subsamples = x[indices_x]

        indices_y = torch.randperm(len(y))[:resamplesize2].to(device)
        y_subsamples = y[indices_y]

        if cost == 'mean':
            gene2remain = ((x_subsamples > 0).sum(axis=0) >= min_cells) | ((y_subsamples > 0).sum(axis=0) >= min_cells)
            x_subsamples = x_subsamples[:,gene2remain]
            y_subsamples = y_subsamples[:,gene2remain]
            genecount.append(gene2remain.sum().item())
        else:
            genecount.append(x_subsamples.shape[1])

        pW = OTLoss(x_subsamples, y_subsamples)
        otdlist.append(pW.item())

    return otdlist, resamplesize1, resamplesize2, int(np.median(genecount))


def get_result(h5ad_filename, yaml_filename, cost):
    adata = sc.read(h5ad_filename)
    # sc.pp.filter_genes(adata, min_cells=50)
    # adata.obs['mainCellType'] = adata.obs['cellType'].apply(lambda x: x.split('_')[0])
    meta = adata.obs

    with open(yaml_filename, 'r') as f:
        pairinfo = yaml.safe_load(f)

    if cost == 'mean':
        metric = 'euclidean_mean'
    else:
        metric = 'euclidean'

    OTLoss = geomloss.SamplesLoss(
            loss='sinkhorn', 
            p=2,
            cost=lambda a, b: cost_func(a, b, metric=metric),
            blur=0.05, 
            backend='tensorized')

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

        output = one_pair(adata1, adata2, OTLoss, cost)
        g1['resamplesize'] = output[1]
        g2['resamplesize'] = output[2]
        output = {'otdlist': output[0], 'genecount': output[3], 'para1': g1, 'para2': g2}

        result[index] = output

        index += 1
    
    return result


def main():
    parser = argparse.ArgumentParser(description='pairwise OTD, take h5ad as input')
    parser.add_argument("--h5ad", type=str, required = True, \
                        help="h5ad file")
    parser.add_argument("--yaml", type=str, required = True, \
                        help="yaml file")
    parser.add_argument("--cost", type=str, required = False, choices=['sum','mean'], default='mean', \
                        help="yaml file")
    parser.add_argument("--out", type=str, required = False, default='output', \
                        help="output prefix")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.out+'.logfile', 'w') as log:
        log.write('{}\n'.format(sys.argv[0]))
        log.write('{}--h5ad   {}\n'.format(' '*4, args.h5ad))
        log.write('{}--yaml   {}\n'.format(' '*4, args.yaml))
        log.write('{}--cost   {}\n'.format(' '*4, args.cost))
        log.write('{}--out    {}\n\n'.format(' '*4, args.out))
        
        log.write('Hostname: '+socket.gethostname()+'\n')
        log.write('Working directory: '+os.getcwd()+'\n')
        log.write('Device: '+str(device)+'\n')
        log.write('Start time: '+time.strftime("%Y-%m-%d %X",time.localtime())+'\n\n')

    result = get_result(args.h5ad, args.yaml, args.cost)

    with open(args.out + '.pickle', 'wb') as fout:
        pickle.dump(result, fout)

    # with open(output_prefix + '.pickle', 'rb') as fin:
    #     result = pickle.load(fin)

    with open(args.out+'.logfile','a') as log:
        log.write("Output "+args.out+'.pickle\n')
        log.write('End time: '+time.strftime("%Y-%m-%d %X",time.localtime())+'\n\n')


if __name__ == '__main__':
    main()

