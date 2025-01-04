import sys
import argparse
import json
import logging
import os
import random

from Models.DOGNMF import sDOGNMF, kDOGNMF
from Models.ERWNMF import ERWNMF
from Models.GNMF import GNMF, TriPNMF, PNMF, TriNMF, TriONMF
from Models.DGRSNMF import GRDeepSNMF
from Models.GRSNMF import GRSNMF
from Models.NMF import NMF
from Models.OGNMF import OGNMF
from Models.RSCNMF import RSCNMF
from Models.dnsNMF import dnsNMF

import scipy.io

from Models.dsNMF import dsnmf
from Models.nsNMF import nsNMF
from Utils.utils import print_static

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )
    # python main.py --model TriPNMF --dataset jaffe
    parser.add_argument('--cuda', default=True, action='store_true', help='use GPU')
    parser.add_argument('--dataset', type=str, default='orl', choices=["yale_a", "yale_b", "umist", "warpar10p", "orl", "jaffe"])
    parser.add_argument('--model', type=str, default="RSCNMF") #choices=["sDOGNMF", "kDOGNMF", "dnsNMF", "dsnmf", "ERWNMF", "RSCNMF", "OGNMF", "GRSNMF", "GNMF", "NMF", "nsNMF", "DGRSNMF"])
    parser.add_argument('--layer', type=int, default=2, help='The number of layers')
    ## Comment/Uncomment for UPC runs
    #parser.add_argument('--k1', type=list, default=[80], help='The size of the first layer')
    #parser.add_argument('--k2', type=list, default=[10, 20], help='The size of the second layer') #[10, 20, 30, 40, 50, 60, 70]
    #parser.add_argument('--al', type=list, default=[1e-03], help='Alpha range') #[1e-03, 1e-02, 1e-01, 1e01]
    #parser.add_argument('--be', type=list, default=[0.1], help='Beta range') #[1e-02, 1e-01, 1]
    #parser.add_argument('--pos_al', type=list, default=[1e-8], help='Pos alpha range') #[1e03, 1e04, 1e05, 1e06]
    #parser.add_argument('--pos_be', type=list, default=[1e04], help='Pos beta range') #[10, 100, 1000]
    #parser.add_argument('--lda', type=list, default=[1, 10] , help='Lambda range') #[1, 10, 100]
    #parser.add_argument('--knn', type=list, default=[2, 20], help='k_knn_list') #[3, 5, 6, 11, 21]
    ## ___________________________
    ## Leave uncommented
    parser.add_argument('--iter', type=int, default=100, help='Maximum iteration')
    parser.add_argument('--inner', type=int, default=100, help='Maximum inner iteration')
    parser.add_argument('--kmeans', type=int, default=20, help='Maximum kmeans iteration')
    parser.add_argument('--eps1', type=float, default=1e-12, help='Epsilon 1')
    parser.add_argument('--eps2', type=float, default=1e-10, help='Epsilon 2')
    parser.add_argument('--att', type=float, default=-1, help='Attention')
    #  Comment/Uncomment for HPC runs
    parser.add_argument('--k1', type=list, default=[80, 100, 120, 200], help='The size of the first layer')
    parser.add_argument('--k2', type=list, default=[10, 20, 30, 40, 50, 60, 70], help='The size of the second layer')
    parser.add_argument('--al', type=list, default=[1e-03, 1e-02, 1e-01, 1e01], help='Alpha range')
    parser.add_argument('--be', type=list, default=[1e-02, 1e-01, 1], help='Beta range') #
    parser.add_argument('--pos_al', type=list, default=[1e-8, 1e-6, 1e-04, 1e-02, 1e02, 1e04, 1e06, 1e08, 1e10], help='Pos alpha range') #[1e03, 1e04, 1e05, 1e06]
    parser.add_argument('--pos_be', type=list, default=[1e-8, 1e-6, 1e-04, 1e-02, 1e02, 1e04, 1e06, 1e08, 1e10], help='Pos beta range') #[10, 100, 1000]
    parser.add_argument('--lda', type=list, default=[1, 10, 100] , help='Lambda range') #[1, 10, 100]
    parser.add_argument('--knn', type=list, default=[2, 5, 10, 20], help='k_knn_list') #[3, 5, 6, 11, 21]
    ## ___________________________

    return parser.parse_args(args)
def load_dataset(dataset):

    if dataset in ["jaffe", "yale_a", "yale_b"]:
        imgData = scipy.io.loadmat('./Image_Data/' + dataset +'.mat')
        matImg = imgData['fea'].astype('float32')
        matGnd = imgData['gnd']
        y = matGnd.ravel()
    elif dataset in ["orl", "warpar10p", "umist"]:
        imgData = scipy.io.loadmat('./Image_Data/orl.mat')
        matImg = imgData['X'].astype('float32')
        matGnd = imgData['Y']
        y = matGnd.ravel()
    else:
        print(">>The dataset is not supported, please change the name of your dataset.")


    return imgData, matImg, matGnd, y


def run_model(model, dataset, alphas, betas, matImg, matGnd, k1_list, k2_list, maxiter_kmeans, l, maxiter, eps_1, eps_2, y,
              maxiter_inner, pos_alpha_range, pos_beta_range, lambda_range, k_knn_list, plot_graphs, att=1):
    if model == "sDOGNMF":
        sDOGNMF.run_model(model, dataset, alphas, betas, matImg, matGnd, k1_list, k2_list, maxiter_kmeans, l, maxiter, eps_1,
                            eps_2, y)
    elif model == "kDOGNMF":
        kDOGNMF.run_model(model, dataset, alphas, betas, matImg, matGnd, k1_list, k2_list, k_knn_list, maxiter_kmeans, l, maxiter, eps_1,
                            eps_2, y)

    elif model == "dnsNMF":
        dnsNMF.run_model(model, dataset, l, alphas, matImg, y, k1_list, k2_list, maxiter, maxiter_inner, maxiter_kmeans)

    elif model == "dsnmf":
        dsnmf.run_model(model, dataset, matImg, y, k1_list, k2_list, maxiter_kmeans)

    elif model == "ERWNMF":
        ERWNMF.run_model(model, dataset, matImg, y, k2_list, maxiter, maxiter_kmeans, plot_graphs)

    elif model == "RSCNMF":
        RSCNMF.run_model(model, dataset, matImg, y, pos_alpha_range, pos_beta_range, k_knn_list, k2_list, lambda_range,
                         maxiter_kmeans, maxiter)

    elif model == "OGNMF":
        OGNMF.run_model(model, dataset, matImg, y, alphas, betas, k_knn_list, k2_list, maxiter_kmeans, eps_1, eps_2, max_iter)

    elif model == "GRSNMF":
        GRSNMF.run_model(model, dataset, matImg, y, k_knn_list, k2_list, alpha_range, maxiter, maxiter_kmeans)

    elif model == "GNMF":
        GNMF.run_model(model, dataset, matImg, y, k_knn_list, k2_list, alpha_range, maxiter, maxiter_kmeans)

    elif model == "NMF":
        NMF.run_model(model, dataset, matImg, y, k2_list, maxiter_kmeans)

    elif model == "nsNMF":
        nsNMF.run_model(model, dataset, alphas, matImg, y, k2_list, maxiter, maxiter_inner, maxiter_kmeans)

    elif model == "DGRSNMF":
        GRDeepSNMF.run_model(model, dataset, matImg, y, k_knn_list, k1_list, k2_list, alphas, l, maxiter_kmeans, maxiter, maxiter_inner)

    elif model == "TriPNMF":
        TriPNMF.run_model(model, dataset, matImg, y, k_knn_list, k2_list, pos_alpha_range, pos_beta_range, maxiter, maxiter_kmeans, att)
        #(model, dataset, matImg, y, k_knn_list, k1_list, k2_list, alphas, l, maxiter_kmeans, maxiter, maxiter_inner)
    elif model == "PNMF":
        PNMF.run_model(model, dataset, matImg, y, k_knn_list, k2_list, pos_beta_range, maxiter, maxiter_kmeans)
    elif model == "TriNMF":
        TriNMF.run_model(model, dataset, matImg, y, k_knn_list, k2_list, maxiter, maxiter_kmeans)
    elif model == "TriONMF":
        TriONMF.run_model(model, dataset, matImg, y, k_knn_list, k2_list, pos_alpha_range, pos_beta_range, maxiter, maxiter_kmeans)
    else:
        print("The model is not supported")


if __name__ == '__main__':
    # python main.py --model TriPNMF --dataset jaffe
    args = parse_args()
    dataset = args.dataset #"orl"
    model = args.model #"RSCNMF"  # Options : sDOGNMF, kDOGNMF, dnsNMF, dsnmf, ERWNMF, RSCNMF, OGNMF, GRSNMF, GNMF, NMF, nsNMF, DGRSNMF
    # Setting parameters and hyper-parameters
    l = args.layer
    k1_list = args.k1
    k2_list = args.k2
    alpha_range = args.al
    beta_range = args.be
    pos_alpha_range = args.pos_al
    pos_beta_range = args.pos_be
    lambda_range = args.lda
    k_knn_list = args.knn
    max_iter = args.iter
    maxiter_inner = args.inner
    maxiter_kmeans = args.kmeans
    eps_1 = args.eps1
    eps_2 = args.eps2
    att = args.att

    print(f'The attention value to the reconstruction errors is {att}\n {"<<<>>>"*20 }')

    write_to_file = False
    plot_graphs = False
    if model.lower() == 'all':
        models = ["sDOGNMF", "kDOGNMF", "dnsNMF", "ERWNMF", "RSCNMF", "OGNMF", "GRSNMF", "GNMF", "NMF", "nsNMF", "DGRSNMF", "TriPNMF", "PNMF", "TriNMF", "TriOPNMF"]# "dsnmf",
    else:
        models = [model]
    for model in models:
        if write_to_file:
            path = f"Results/{dataset}/output_new_{model}_{dataset}.out"
            sys.stdout = open(path, 'w')

        # Load dataset
        imgData, matImg, matGnd, y = load_dataset(dataset)
        # Print static params
        print_static(model, dataset, max_iter, eps_1, eps_2)

        # Run model
        s=run_model(model, dataset, alpha_range, beta_range, matImg, matGnd, k1_list, k2_list, maxiter_kmeans, l, max_iter, eps_1,
                  eps_2, y, maxiter_inner, pos_alpha_range, pos_beta_range, lambda_range, k_knn_list, plot_graphs, att=att)
        print(f"{'>'*10, ' '*3} Evaluation of the model {model} is completed")
