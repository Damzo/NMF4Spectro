def run_model(model, dataset, matImg, y, k_knn_list, k_list, lambda_list, maxiter, maxiter_kmeans):
    data = loadmat('jaffe.mat') #matImg
    X = NormalizeFea(data['fea'], 1) #Normalized
    X = X.T
    c = len(np.unique(data['gnd']))
    Namedata = 'Jaffe'
    x=[]
    for i in data['gnd']:
        x.append(i[0])
    label = np.array(x)

    # Parameters
    m, n = X.shape
