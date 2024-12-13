from sklearn.cluster import KMeans
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import DistanceMetric
from Utils.metrics_evaluation import evaluate_nmi, accuracy

import torch

MIN_NORM = 1e-15
BALL_EPS = {torch.float32: 4e-3, torch.float64: 1e-5}

def ClusteringMeasure(Y, predY):
    # Evaluate clustering accuracy, NMI, and purity
    nmi = 100 * evaluate_nmi(Y, predY)
    acc = 100 * accuracy(Y, predY)
    # Ensure Y and predY are column vectors

    Y = Y.reshape(-1, 1) if Y.ndim != 2 else Y
    predY = predY.reshape(-1, 1) if predY.ndim != 2 else predY

    # Compute purity
    correnum = 0
    for ci in np.unique(predY):
        incluster = Y[predY.flatten() == ci]
        inclunub = np.histogram(incluster, bins=np.arange(1, np.max(incluster) + 2))[0]
        correnum += np.max(inclunub)
    Purity = (correnum / len(predY))*100
    return acc, nmi, Purity

def init_kmeans(y):
    kmeans_classes = np.unique(y).shape[0]
    kmeans = KMeans(kmeans_classes)

    return kmeans


def print_static(model, dataset, max_iter, eps_1, eps_2):
    print(f"Model : {model}")
    print(f"Dataset : {dataset}")
    print(f"max iterations : {max_iter}")
    print(f"myeps_1 : {eps_1}") if eps_1 else ''
    print(f"myeps_2 : {eps_2}") if eps_2 else ''


def construct_similarity_matrix(gnd):

    m = len(gnd)
    S = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            if gnd[i] == gnd[j]:
                S[i][j] = 1

    return S


def KNN(X, k=6):
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(X)
    distances, indices = neigh.kneighbors(X)
    dist = DistanceMetric.get_metric('euclidean')
    x = dist.pairwise(X)
    z = np.zeros(x.shape)
    for i in range(x.shape[0]):
        z[i, indices[i]] = 1
        z[indices[i], i] = 1
    n = z.shape[0]
    Q = np.identity(n)
    z -= Q
    return z, x, indices


# Store best predicted cluster in file for later plotting
def store_kmeans(data, pred, model, dataset):
    path = f"Results/{dataset}/kmeans_{model}_{dataset}"
  #  np.savez(path, data=data, kmneans_pred=pred)


# ################# Hyperbolic operations utils functions ########################

class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-5, 1 - 1e-5)
        ctx.save_for_backward(x)
        dtype = x.dtype
        x = x.double()
        return (torch.log_(1 + x).sub_(torch.log_(1 - x))).mul_(0.5).to(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 - input ** 2)


def artanh(x):
    return Artanh.apply(x)


def tanh(x):
    return x.clamp(-15, 15).tanh()


# ################# HYP OPS ########################

def expmap0(u, c):
    """Exponential map taken at the origin of the Poincare ball with curvature c.

    Args:
        u: torch.Tensor of size B x d with hyperbolic points
        c: torch.Tensor of size 1 or B x 1 with absolute hyperbolic curvatures

    Returns:
        torch.Tensor with tangent points.
    """
    return u
    sqrt_c = torch.abs(c) ** 0.5
    u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
    return project(gamma_1, c)


def logmap0(y, c):
    """Logarithmic map taken at the origin of the Poincare ball with curvature c.

    Args:
        y: torch.Tensor of size B x d with tangent points
        c: torch.Tensor of size 1 or B x 1 with absolute hyperbolic curvatures

    Returns:
        torch.Tensor with hyperbolic points.
    """
    sqrt_c = torch.abs(c) ** 0.5
    y_norm = y.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    return y / y_norm / sqrt_c * artanh(sqrt_c * y_norm)


def project(x, c):
    """Project points to Poincare ball with curvature c.

    Args:
        x: torch.Tensor of size B x d with hyperbolic points
        c: torch.Tensor of size 1 or B x 1 with absolute hyperbolic curvatures

    Returns:
        torch.Tensor with projected hyperbolic points.
    """
    norm = x.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    eps = BALL_EPS[x.dtype]
    maxnorm = (1 - eps) / (torch.abs(c) ** 0.5)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)

def proj_into_hyplane(x, norm_vec, number = 1):
    """
    :param x: Vector in ambient space
    :param norm_vec: Normal vector of the hyper plane
    :return: Projection of x into the hyperplane
    """
    return x - number * torch.sum(x * norm_vec, dim=-1, keepdim=True) * norm_vec

def mobius_add(x, y, c):
    """Mobius addition of points in the Poincare ball with curvature c.

    Args:
        x: torch.Tensor of size B x d with hyperbolic points
        y: torch.Tensor of size B x d with hyperbolic points
        c: torch.Tensor of size 1 or B x 1 with absolute hyperbolic curvatures

    Returns:
        Tensor of shape B x d representing the element-wise Mobius addition of x and y.
    """
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    y2 = torch.sum(y * y, dim=-1, keepdim=True)
    xy = torch.sum(x * y, dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
    return num / denom.clamp_min(MIN_NORM)

def Scalar_Multi0(r, y, c):
    a = expmap0(r * logmap0(y,c), c)
    return a

def Mat_Multi0(M, y, c):
    a = expmap0(M @ logmap0(y,c), c)
    return a

def Scalar_MultiX(x, r, y, c):
    a = expmapX(x, r * logmap0(y,c), c)
    return a

def Mat_MultiX(x, M, y, c):
    a = expmapX(x, M @ logmap0(y,c), c)
    return a
def expmapX(x, u, c):
    """Exponential map taken at the origin of the Poincare ball with curvature c.

    Args:
        u: torch.Tensor of size B x d with hyperbolic points
        c: torch.Tensor of size 1 or B x 1 with absolute hyperbolic curvatures

    Returns:
        torch.Tensor with tangent points.
    """
    sqrt_c = torch.abs(c) ** 0.5
    u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
    return project(mobius_add(x, gamma_1, c), c)

def logmapX(x, y, c):
    """Logarithmic map taken at the origin of the Poincare ball with curvature c.

    Args:
        y: torch.Tensor of size B x d with tangent points
        c: torch.Tensor of size 1 or B x 1 with absolute hyperbolic curvatures

    Returns:
        torch.Tensor with hyperbolic points.
    """
    sqrt_c = torch.abs(c) ** 0.5
    ned_xy = mobius_add(-x, y, c)
    ned_xy_norm = ned_xy.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)

    return ned_xy / ned_xy_norm / sqrt_c * artanh(sqrt_c * ned_xy_norm)

# ################# HYP DISTANCES ########################

def hyp_distance(x, y, c, eval_mode=False):
    """Hyperbolic distance on the Poincare ball with curvature c.

    Args:
        x: torch.Tensor of size B x d with hyperbolic queries
        y: torch.Tensor with hyperbolic queries, shape n_entities x d if eval_mode is true else (B x d)
        c: torch.Tensor of size 1 with absolute hyperbolic curvature

    Returns: torch,Tensor with hyperbolic distances, size B x 1 if eval_mode is False
            else B x n_entities matrix with all pairs distances
    """
    sqrt_c = torch.abs(c) ** 0.5
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    if eval_mode:
        y2 = torch.sum(y * y, dim=-1, keepdim=True).transpose(0, 1)
        xy = x @ y.transpose(0, 1)
    else:
        y2 = torch.sum(y * y, dim=-1, keepdim=True)
        xy = torch.sum(x * y, dim=-1, keepdim=True)
    c1 = 1 - 2 * c * xy + c * y2
    c2 = 1 - c * x2
    num = torch.sqrt((c1 ** 2) * x2 + (c2 ** 2) * y2 - (2 * c1 * c2) * xy)
    denom = 1 - 2 * c * xy + c ** 2 * x2 * y2
    pairwise_norm = num / denom.clamp_min(MIN_NORM)
    dist = artanh(sqrt_c * pairwise_norm)
    return 2 * dist / sqrt_c


def hyp_distance_multi_c(x, v, c, eval_mode=False):
    """Hyperbolic distance on Poincare balls with varying curvatures c.

    Args:
        x: torch.Tensor of size B x d with hyperbolic queries
        y: torch.Tensor with hyperbolic queries, shape n_entities x d if eval_mode is true else (B x d)
        c: torch.Tensor of size B x d with absolute hyperbolic curvatures

    Return: torch,Tensor with hyperbolic distances, size B x 1 if eval_mode is False
            else B x n_entities matrix with all pairs distances
    """
    sqrt_c = torch.abs(c) ** 0.5
    #######################  ME
    # v = expmap0(v1, c)
    if eval_mode:
        vnorm = torch.norm(v, p=2, dim=-1, keepdim=True).transpose(0, 1)
        xv = x @ v.transpose(0, 1) / vnorm
    else:
        vnorm = torch.norm(v, p=2, dim=-1, keepdim=True)
        xv = torch.sum(x * v / vnorm, dim=-1, keepdim=True)
    gamma = tanh(sqrt_c * vnorm) / sqrt_c
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    c1 = 1 - 2 * c * gamma * xv + c * gamma ** 2
    c2 = 1 - c * x2
    num = torch.sqrt((c1 ** 2) * x2 + (c2 ** 2) * (gamma ** 2) - (2 * c1 * c2) * gamma * xv)
    denom = 1 - 2 * c * gamma * xv + (c ** 2) * (gamma ** 2) * x2
    pairwise_norm = num / denom.clamp_min(MIN_NORM)
    dist = artanh(sqrt_c * pairwise_norm)
    return 2 * dist / sqrt_c
