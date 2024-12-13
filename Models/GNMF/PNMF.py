import numpy as np
import torch
from Utils.metrics_evaluation import evaluate_nmi, accuracy, calculate_silhouette_score, calculate_davies_bouldin_score, \
    calculate_dunn_index
# from Utils.utils import Mat_Multi0, mobius_add, project, logmap0, logmapX, expmap0, expmapX, hyp_distance
import scipy.io
from numpy.linalg import linalg
from sklearn.cluster import KMeans
import statistics
import warnings
import csv
import json

from Utils.utils import KNN, init_kmeans, store_kmeans, ClusteringMeasure

warnings.filterwarnings('ignore')



def store_resl(data, model, dataset, name=''):
    path = f"Results/{dataset}/kmeans_{model}_{dataset}_{name}"
    import json
    #np.savez(path, data=data)
    json1 = json.dumps(data)
    # open file for writing, "w"
    f = open(name+".json","w")

    # write json object to file
    f.write(json1)

    # close file
    f.close()

def PNMF(XX_T, beta,m, k,maxiter):
    U = np.random.rand(m,k);
    #V = np.random.rand(k,n);
    #D = np.random.rand(k,k);
    #print(XXX.shape, n, m, k)
    for i in range(1,maxiter+1):   ####for i=1:maxiter
        ## Update U
        # Compute un-normalized U
        XX_TU = np.dot(XX_T,U)
        UU =  np.dot(U, U.T)
        numU = 2*XX_TU + beta*U
        denU = np.dot(UU,XX_TU) + np.dot((np.dot(XX_TU,U.T)),U) + beta*(np.dot(UU,U))
        denU[denU < 1e-10] = 1e-10
        reU = U*np.divide(numU,denU)###np.multiply is for elemen-twise multiplication of two matrices
        # Renormalization
        UU = np.dot(reU, reU.T)
        y = np.dot(UU, XX_T)
        numU = np.trace(y)
        denU = np.trace(np.dot(y, UU))
        denU = denU if denU >= 1e-10 else 1e-10
        U = reU * np.power(numU/denU, 1/2)
    return U

def run_model(model, dataset, matImg, y, k_knn_list, k_list, beta_list, maxiter, maxiter_kmeans):

    # Normalise data
    norma = np.linalg.norm(matImg, 2, 1)[:, None]
    norma += 1e-10
    normal_img = matImg / norma
    # normal_img = matImg / 255.0

    # Definiton of X
    X = normal_img.T
    m, n = X.shape
    ## ADD
    #XX = np.dot(X.T, X)
    XX_T = np.dot(X, X.T)
    ##
    # Initialise Kmeans
    kmeans = init_kmeans(y) #KMeans(n_clusters=y, init='random', n_init=10, max_iter=300, random_state=0)#

    # Util for plotting best cluster produced
    best_cluster_acc = {'acc': 0}

    # For convergence comparison
    iterations = []
    iterations_k2 = {}
    parameters = []
    for b in beta_list:
        parameters.append(b)
    for knn in k_knn_list: # Number of nearest neighbors
        #S_, _, _ = KNN(normal_img, knn) #=W
        #diag = np.sum(S_, axis=1)
        #P_ = np.diag(diag) #=D
        #L = P_ - S_

        maxAcc = {}
        maxNmi = {}
        maxRecon_reeor = {}
        maxSilScore = {}
        maxDunnScore = {}
        maxPurity = {}
        minDavisScore = {}

        ##
        meanAcc = {}
        meanNmi = {}
        meanRecon_reeor = {}
        meanSilScore = {}
        meanDunnScore = {}
        meanDavisScore = {}
        meanPurity = {}

        for k in k_list: # low embedding dimensions

            maxlst_acc = []
            maxlst_nmi = []
            maxlst_recon_err = []
            maxlst_sil_score = []
            maxlst_dunn_score = []
            minlst_davis_score = []
            maxlst_purity = []

            ##
            meanlst_acc = []
            meanlst_nmi = []
            meanlst_recon_err = []
            meanlst_sil_score = []
            meanlst_dunn_score = []
            meanlst_davis_score = []
            meanlst_purity = []

            for p in parameters:
                #alpha = p[0]
                beta = p ##<< Open a for loup here
                U = PNMF(XX_T, beta,m, k,maxiter)
                n_iteration = maxiter
                iterations.append(n_iteration)

                ## Reconstructed matix
                X_reconstructed = (U @ U.T) @ X

                ## Reconstruction Error
                a = np.linalg.norm(X - X_reconstructed, 'fro') ## add
                recon_reeor = (a)/n

                ## Kmeans task
                lst_acc = []
                lst_nmi = []
                lst_recon_err = []
                lst_sil_score = []
                lst_dunn_score = []
                lst_davis_score = []
                lst_purity = []
                Clustermatrix = np.dot(U.T, X)
                Clustermatrix = Clustermatrix.T

                for i in range(1, maxiter_kmeans): # maxiter_kmeans is 20 in args
                    # Apparently kmeans results are not statics.
                    pred = kmeans.fit_predict(Clustermatrix)

                    ## NMI & ACC
                    #nmi, acc = 100 * evaluate_nmi(y, pred), 100 * accuracy(y, pred)
                    acc, nmi, Purity = ClusteringMeasure(y, pred)
                    if acc > best_cluster_acc['acc']:
                        best_cluster_acc['acc'] = acc
                        best_cluster_acc['data'] = Clustermatrix.T ## add
                        best_cluster_acc['pred'] = pred


                    # Silhoutte score
                    silhouette_score = calculate_silhouette_score(Clustermatrix, pred) ## add

                    # Davis-bouldin score
                    davis_score = calculate_davies_bouldin_score(Clustermatrix, pred) ## add

                    # dunn's index
                    dunn_score = calculate_dunn_index(Clustermatrix, y) ## add

                    lst_acc.append(round(acc, 4))
                    lst_nmi.append(round(nmi, 4))
                    lst_recon_err.append(round(recon_reeor, 4))
                    lst_sil_score.append(round(silhouette_score, 4))
                    lst_davis_score.append(round(davis_score, 4))
                    lst_dunn_score.append(round(dunn_score, 4))
                    lst_purity.append(round(Purity, 4))

                    ## End for

                ## The 20 results saved for each alpha, and beta fixed
                maxlst_acc.append(max(lst_acc))
                maxlst_nmi.append(max(lst_nmi))
                maxlst_recon_err.append(min(lst_recon_err))
                maxlst_sil_score.append(max(lst_sil_score))
                maxlst_dunn_score.append((max(lst_dunn_score)))
                minlst_davis_score.append(min(lst_davis_score))
                maxlst_purity.append((max(lst_purity)))

                ##
                meanlst_acc.append(statistics.mean(lst_acc))
                meanlst_nmi.append(statistics.mean(lst_nmi))
                meanlst_recon_err.append(statistics.mean(lst_recon_err))
                meanlst_sil_score.append(statistics.mean(lst_sil_score))
                meanlst_davis_score.append(statistics.mean(lst_davis_score))
                meanlst_dunn_score.append(statistics.mean(lst_dunn_score))
                meanlst_purity.append(statistics.mean(lst_purity))

                if k not in iterations_k2.keys(): #<<
                    iterations_k2[k] = [n_iteration]
                else:
                    iterations_k2[k].append(n_iteration)

                ## End for for Alpha
            # Best results for each k, reduced dimension
            # The following contains optimal results for all the alphas
            maxAcc[k] = maxlst_acc # dict (emb dim k, results/alpha) of list
            maxNmi[k] = maxlst_nmi
            maxRecon_reeor[k] = maxlst_recon_err
            maxSilScore[k] = maxlst_sil_score
            maxDunnScore[k] = maxlst_dunn_score
            minDavisScore[k] = minlst_davis_score
            maxPurity[k] = maxlst_purity

            ##
            meanAcc[k] = meanlst_acc
            meanNmi[k] = meanlst_nmi
            meanRecon_reeor[k] = meanlst_recon_err
            meanSilScore[k] = meanlst_sil_score
            meanDavisScore[k] = meanlst_davis_score
            meanDunnScore[k] = meanlst_dunn_score
            meanPurity[k] = meanlst_purity



            ## ENd for k2

        maxacc_final = {}
        maxnmi_final = {}
        maxrecon_final = {}
        maxSilScore_final = {}
        maxDunnScore_final = {}
        minDavisScore_final = {}
        maxpurity_final = {}

        store_resl(maxAcc, model, dataset, name='maxAcc')
        store_resl(maxNmi, model, dataset, name='maxNmi')
        store_resl(maxRecon_reeor, model, dataset, name='maxRecon_reeor')
        store_resl(maxSilScore, model, dataset, name='maxSilScore')
        store_resl(maxDunnScore, model, dataset, name='maxDunnScore')
        store_resl(minDavisScore, model, dataset, name='minDavisScore')

        print(f"The results of running the Kmeans method {maxiter_kmeans} times and the report of maximum the runs")
        for k in k_list: ## For each emb dim
            maxacc_final[k] = [max(maxAcc[k]), parameters[np.argmax(maxAcc[k])]]
            maxnmi_final[k] = [max(maxNmi[k]), parameters[np.argmax(maxNmi[k])]]
            maxrecon_final[k] = [min(maxRecon_reeor[k]), parameters[np.argmin(maxRecon_reeor[k])]]
            maxSilScore_final[k] = [max(maxSilScore[k]), parameters[np.argmax(maxSilScore[k])]]
            maxDunnScore_final[k] = [max(maxDunnScore[k]), parameters[np.argmax(maxDunnScore[k])]]
            minDavisScore_final[k] = [min(minDavisScore[k]), parameters[np.argmin(minDavisScore[k])]]
            maxpurity_final[k] = [max(maxPurity[k]), parameters[np.argmax(maxPurity[k])]]

            print(f"##################################################################################################")
            print(f" k_knn = {knn}  ")
            print(f" k = {k} :  Max ACC : {maxacc_final[k][0]}, with alpha, beta = {parameters[np.argmax(maxAcc[k])]}")
            print(f" k = {k} :  Max NMI : {maxnmi_final[k][0]}, with alpha, beta = {parameters[np.argmax(maxNmi[k])]}")
            print(f" k = {k} :  Reconstruction Error : {maxrecon_final[k][0]}, with alpha, beta = {parameters[np.argmin(maxRecon_reeor[k])]}")
            print(f" k = {k} :  Max Silhoutte score : {maxSilScore_final[k][0]}, with alpha, beta = "
                  f"{parameters[np.argmax(maxSilScore[k])]}")
            print(f" k = {k} :  Max Dunn's Index score : {maxDunnScore_final[k][0]}, with alpha, beta = "
                  f"{parameters[np.argmax(maxDunnScore[k])]}")
            print(f" k = {k} :  Min David Bouldin score : {minDavisScore[k][0]}, with alpha, beta = "
                  f"{parameters[np.argmin(minDavisScore[k])]}")
            print(f" k = {k} :  Max Purity : {maxpurity_final[k][0]}, with alpha, beta = {parameters[np.argmax(maxPurity[k])]}")

            print(f"##################################################################################################")
        ##

        meanacc_final = {}
        meannmi_final = {}
        meanrecon_final = {}
        meanSilScore_final = {}
        meanDunnScore_final = {}
        meanDavidScore_final = {}
        meanpurity_final = {}

        store_resl(meanAcc, model, dataset, name='meanAcc')
        store_resl(meanNmi, model, dataset, name='meanNmi')
        store_resl(meanRecon_reeor, model, dataset, name='meanRecon_reeor')
        store_resl(meanSilScore, model, dataset, name='meanSilScore')
        store_resl(maxDunnScore, model, dataset, name='maxDunnScore')
        store_resl(meanDavisScore, model, dataset, name='meanDavisScore')

        print(f"The results of running the Kmeans method  {maxiter_kmeans}  times and the average of the runs")
        for k in k_list: #<<
            meanacc_final[k] = [max(meanAcc[k]), parameters[np.argmax(meanAcc[k])]]
            meannmi_final[k] = [max(meanNmi[k]), parameters[np.argmax(meanNmi[k])]]
            meanrecon_final[k] = [min(meanRecon_reeor[k]), parameters[np.argmin(meanRecon_reeor[k])]]
            meanSilScore_final[k] = [max(meanSilScore[k]), parameters[np.argmax(meanSilScore[k])]]
            meanDunnScore_final[k] = [max(meanDunnScore[k]), parameters[np.argmax(meanDunnScore[k])]]
            meanDavidScore_final[k] = [max(meanDavisScore[k]), parameters[np.argmin(meanDavisScore[k])]]
            meanpurity_final[k] = [max(meanPurity[k]), parameters[np.argmax(meanPurity[k])]]

            print(f"##################################################################################################")
            print(f" k_knn = {knn} ")
            print(f" k = {k} :  Avg ACC : {meanacc_final[k][0]}, with alpha, beta = {parameters[np.argmax(meanAcc[k])]}")
            print(f" k = {k} :  Avg NMI : {meannmi_final[k][0]}, with alpha, beta = {parameters[np.argmax(meanNmi[k])]}")
            print(f" k = {k} :  Reconstruction Error : {meanrecon_final[k][0]}, Acc is {meanAcc[k][np.argmin(meanRecon_reeor[k])]}, mni is {meanNmi[k][np.argmin(meanRecon_reeor[k])]} with alpha, beta = {meanrecon_final[k][1]}")
            print(f" k = {k} :  Avg Silhoutte score : {meanSilScore_final[k][0]}, with alpha, beta = "
                  f"{parameters[np.argmax(meanSilScore[k])]}")
            print(f" k = {k} :  Avg Dunn's Index score : {meanDunnScore_final[k][0]}, with alpha, beta = "
                  f"{parameters[np.argmax(meanDunnScore[k])]}")
            print(f" k = {k} :  Avg David Bouldin score : {meanDavidScore_final[k][0]}, with alpha , beta= "
                  f"{parameters[np.argmin(meanDavisScore[k])]}")
            print(f" k = {k} :  Avg Purity : {meanpurity_final[k][0]}, with alpha, beta = {parameters[np.argmax(meanPurity[k])]}")

            print(f"##################################################################################################")
    ##**
    ## print for convergence comparison
    for k in k_list: #<<
        print(f"Average no. of iterations for k = {k} : {statistics.mean(iterations_k2[k])}")
    print(f"Overall average no. of iterations : {statistics.mean(iterations)}")

    # Storing details of best cluster
    data = best_cluster_acc['data']
    pred = best_cluster_acc['pred']
    store_kmeans(data, pred, model, dataset)

    print("done")
    return ' '
