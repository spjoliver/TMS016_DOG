from scipy.special import kv, betainc
from scipy.special import gammaln as lgamma
from scipy import sparse
from scipy.optimize import minimize
import numpy as np
from scipy.spatial.distance import squareform, pdist
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, Tuple, Optional, Union, Callable, List
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import pairwise_distances


def emp_variogram(D: np.ndarray, data: np.ndarray, N: int) -> Dict[str, np.ndarray]:
    """
    Calculate the empirical variogram from input data and distance matrix.
    
    Parameters
    ----------
    D : numpy.ndarray
        Input distance matrix or coordinates array.
    data : numpy.ndarray
        Input data values corresponding to each point in D.
    N : int
        Number of bins for calculating the variogram.
    
    Returns
    -------
    out : dict
        Dictionary containing the variogram information (bin centers, variogram values, and number of pairs per bin).
    """
    
    # Ensure that D is a square matrix. If not, convert it using pdist and squareform from scipy.
    if D.shape[0] != D.shape[1]:
        D = squareform(pdist(D))
        #print(D.shape)
        #D = pairwise_distances(D, n_jobs=-1)
        #print(D.shape)
        
    
    max_dist = D.max()
    
    # Create an array of linearly spaced distances from 0 to max_dist with N elements.
    d = np.linspace(0, max_dist, N)
    
    # Set the lower triangle of the distance matrix to -1 to avoid duplicate calculations.
    D[np.tril_indices(D.shape[0], k=-1)] = -1
    
    out = {}
    # Calculate bin centers.
    out['h'] = (d[1:] + d[:-1]) / 2
    out['variogram'] = np.zeros(N-1)
    out['N'] = np.zeros(N-1)
    
    # Calculate the empirical variogram.
    for i in range(N-1):
        I, J = np.where((d[i] < D) & (D <= d[i+1]))
        out['N'][i] = len(I)
        out['variogram'][i] = 0.5 * np.mean((data[I] - data[J])**2)
    return out


def stencil2prec(sz: Tuple[int, int], q: np.ndarray) -> sparse.csc_matrix:
    """
    Create a precision matrix from a stencil matrix.
    
    Parameters
    ----------
    sz : tuple of int
        Tuple containing the dimensions of the desired precision matrix.
    q : numpy.ndarray
        Stencil matrix used to create the precision matrix.
    
    Returns
    -------
    Q : scipy.sparse.csc_matrix
        The resulting precision matrix in Compressed Sparse Column (CSC) format.
    """

    II = []
    KK = []
    JJ_I = []
    JJ_J = []

    I, J = np.meshgrid(np.arange(1, 1 + sz[0]), np.arange(1, 1 + sz[1]), indexing='ij')
    I = I.flatten()
    J = J.flatten()

    ones_arr = np.ones((sz[0] * sz[1], 1))

    # Loop over the stencil matrix elements to compute II, JJ_I, JJ_J, and KK lists.
    for i in range(1, 1 + q.shape[0]):
        for j in range(1, 1 + q.shape[1]):
            if q[i - 1, j - 1] != 0:
                II.append(I + sz[0] * (J - 1))
                JJ_I.append(I + i - int((q.shape[0] + 1) / 2))
                JJ_J.append(J + j - int((q.shape[1] + 1) / 2))
                KK.append(q[i - 1, j - 1] * ones_arr)

    II = np.concatenate(II)
    JJ_I = np.concatenate(JJ_I)
    JJ_J = np.concatenate(JJ_J)
    KK = np.concatenate(KK)[:, 0]
    JJ = JJ_I + sz[0] * (JJ_J - 1)

    
    # Filter out indices that are out of bounds.
    ok = (JJ_I >= 1) & (JJ_I <= sz[0]) & (JJ_J >= 1) & (JJ_J <= sz[1])

    # Create a sparse matrix in COO format using the filtered data.
    Q = sparse.coo_matrix((KK[ok], (II[ok] - 1, JJ[ok] - 1)), shape=(np.prod(sz), np.prod(sz)))

    # Convert the sparse matrix to the CSC format.
    return Q.tocsc()

def matern_covariance(h, sigma, kappa, nu):
    if nu <= 0:
        raise ValueError('nu must be postive')
    if kappa <= 0:
        raise ValueError('kappa must be postive')
    if sigma <= 0:
        raise ValueError('sigma must be postive')
    
    # some error when h is zero, add a small number to avoid this
    h = h + 1e-10
    if not isinstance(h, np.ndarray):
        h = np.asarray(h)[np.newaxis]
    rho = np.sqrt(8 * nu) / kappa
    hpos = h > 0
    r = np.zeros_like(h)
    
    r[~hpos] = sigma**2
    hunique, J = np.unique(h[hpos], return_inverse=True)
    
    B = np.log(kv(nu, kappa * hunique))
    B = np.exp(2 * np.log(sigma) - lgamma(nu) - (nu - 1) * np.log(2) + nu * np.log(kappa * hunique) + B)
    
    if np.any(np.isinf(B)):
        B[np.isinf(B)] = (sigma**2) * np.exp(-2 * (hunique[np.isinf(B)] / rho)**2)
    r[hpos] = B[J]
    return r

def spherical_covariance(h, sigma, theta):
    Sigma = np.zeros_like(h)
    nz = h < theta
    hs = h[nz] / theta
    Sigma[nz] = sigma**2 * (1 - 1.5 * hs + 0.5 * hs**3)
    return Sigma

def gaussian_covariance(h, sigma, rho):
    if sigma <= 0:
        raise ValueError('sigma must be postive')
    if rho <= 0:
        raise ValueError('rho must be postive')
    hpos = h > 0
    Sigma = np.zeros_like(h)
    Sigma[~hpos] = sigma**2
    hunique, J = np.unique(h[hpos], return_inverse=True)
    B = sigma**2 * np.exp(-2 * (hunique / rho)**2)
    Sigma[hpos] = B[J]
    return Sigma

def exponential_covariance(h, sigma, kappa):
    if kappa <= 0:
        raise ValueError('kappa must be postive')
    if sigma <= 0:
        raise ValueError('sigma must be postive')
    hpos = h > 0
    Sigma = np.zeros_like(h)
    Sigma[~hpos] = sigma**2
    hunique, J = np.unique(h[hpos], return_inverse=True)
    B = sigma**2 * np.exp(-kappa * hunique)
    Sigma[hpos] = B[J]
    return Sigma

def euclidshat_covariance(h, sigma, theta, n):
    if sigma <= 0:
        raise ValueError('sigma should be positive')
    if theta <= 0:
        raise ValueError('theta should be positive')
    Sigma = np.zeros_like(h)
    nz = h < theta
    hs = h[nz] / theta
    Sigma[nz] = sigma**2 * betainc(1 - hs**2, (n+1) / 2, 1 / 2)
    return Sigma

def matern_variogram(h, sigma, kappa, nu, sigma_e=0):
    sv = sigma**2 + sigma_e**2 - matern_covariance(h, sigma, kappa, nu)
    return sv


def log_like(
    p: np.ndarray,
    covf: Callable,
    data: np.ndarray,
    X: np.ndarray,
    D: np.ndarray,
    I: np.ndarray,
    fixed: Dict[str, float],
    n_cov: int,
    names: List[str],
    reml: bool
) -> float:
    """
    Log likelihood function for covariance parameter estimation.
    
    Parameters
    ----------
    p : numpy.ndarray
        Parameters to estimate.
    covf : callable
        Covariance function.
    data : numpy.ndarray
        Input data values.
    X : numpy.ndarray
        Auxiliary data for regression.
    D : numpy.ndarray
        Distance matrix.
    I : numpy.ndarray
        Identity matrix.
    fixed : dict
        A dictionary of fixed parameters for the covariance model.
    n_cov : int
        Number of covariates.
    names : list
        Names of parameters to estimate.
    reml : bool
        Whether to use Restricted Maximum Likelihood (REML) estimation.
    
    Returns
    -------
    ll : float
        Log likelihood value.
    """

    # Compute covariance matrix
    if names[-1] == 'sigma_e':  # nugget not fixed
        Sigma = covf(D, np.exp(p[:-1])) + np.exp(2*p[-1]) * I
    else:
        Sigma = covf(D, np.exp(p)) + (fixed['sigma_e'])**2 * I

    # Compute Cholesky factor; if it fails, return a large value
    R, p = np.linalg.cholesky(Sigma)
    if p > 0:
        ll = np.inf
        return ll

    SiY = np.linalg.solve(R, np.linalg.solve(R.T, data))

    if n_cov > 0:
        SiX = np.linalg.solve(R, np.linalg.solve(R.T, X))
        v = SiY - SiX.dot(np.linalg.solve(X.T.dot(SiX), X.T.dot(SiY)))
    else:
        v = SiY

    ll = -np.sum(np.log(np.diag(R))) - 0.5 * data.T.dot(v)

    if reml and n_cov > 0:
        Rreml, p = np.linalg.cholesky(X.T.dot(SiX))
        if p > 0:
            ll = np.inf
            return ll
        ll = ll - np.sum(np.log(np.diag(Rreml)))

    ll = -ll
    return ll


def select_covariance(cov,fixed):
    sigma_fixed = 0
    kappa_fixed = 0
    rho_fixed = 0
    nu_fixed = 0
    theta_fixed = 0
    if fixed:
        names = list(fixed.keys())
        for i in range(len(names)):
            if names[i] == 'sigma':
                sigma_fixed = 1
            elif names[i] == 'kappa':
                kappa_fixed = 1
            elif names[i] == 'rho':
                rho_fixed = 1
            elif names[i] == 'nu':
                nu_fixed = 1
            elif names[i] == 'theta':
                theta_fixed = 1
    if cov == 'exponential':
        if sigma_fixed == 0 and kappa_fixed == 0:
            covf = lambda d,x: exponential_covariance(d,x[0],x[1])
        elif sigma_fixed == 1 and kappa_fixed == 0:
            covf = lambda d,x: exponential_covariance(d,fixed['sigma'],x[0])
        elif sigma_fixed == 0 and kappa_fixed == 1:
            covf = lambda d,x: exponential_covariance(d,x[0],fixed['kappa'])
        else:
            covf = lambda d,x: exponential_covariance(d,fixed['sigma'],fixed['kappa'])
    elif cov == 'gaussian':
        if sigma_fixed == 0 and rho_fixed == 0:
            covf = lambda d,x: gaussian_covariance(d,x[0],x[1])
        elif sigma_fixed == 1 and rho_fixed == 0:
            covf = lambda d,x: gaussian_covariance(d,fixed['sigma'],x[0])
        elif sigma_fixed == 0 and rho_fixed == 1:
            covf = lambda d,x: gaussian_covariance(d,x[0],fixed['rho'])
        else:
            covf = lambda d,x: gaussian_covariance(d,fixed['sigma'],fixed['rho'])
    elif cov == 'matern':
        if sigma_fixed == 0 and kappa_fixed == 0 and nu_fixed == 0:
            covf = lambda d,x: matern_covariance(d,x[0],x[1],x[2])
        elif sigma_fixed == 1 and kappa_fixed == 0 and nu_fixed == 0:
            covf = lambda d,x: matern_covariance(d,fixed['sigma'],x[1],x[2])
        elif sigma_fixed == 0 and kappa_fixed == 1 and nu_fixed == 0:
            covf = lambda d,x: matern_covariance(d,x[0],fixed['kappa'],x[2])
        elif sigma_fixed == 0 and kappa_fixed == 0 and nu_fixed == 1:
            covf = lambda d,x: matern_covariance(d,x[0],x[1],fixed['nu'])
        elif sigma_fixed == 1 and kappa_fixed == 1 and nu_fixed == 0:
            covf = lambda d,x: matern_covariance(d,fixed['sigma'],fixed['kappa'],x[0])
        elif sigma_fixed == 1 and kappa_fixed == 0 and nu_fixed == 1:
            covf = lambda d,x: matern_covariance(d,fixed['sigma'],x[0],fixed['nu'])
        elif sigma_fixed == 0 and kappa_fixed == 1 and nu_fixed == 1:
            covf = lambda d,x: matern_covariance(d,x[0],fixed['kappa'],fixed['nu'])
        elif sigma_fixed == 1 and kappa_fixed == 1 and nu_fixed == 1:
            covf = lambda d,x: matern_covariance(d,fixed['sigma'],fixed['kappa'],fixed['nu'])
    elif cov == 'spherical':
        if sigma_fixed == 0 and theta_fixed == 0:
            covf = lambda d,x: spherical_covariance(d,x[0],x[1])
        elif sigma_fixed == 1 and theta_fixed == 0:
            covf = lambda d,x: spherical_covariance(d,fixed['sigma'],x[0])
        elif sigma_fixed == 0 and theta_fixed == 1:
            covf = lambda d,x: spherical_covariance(d,x[0],fixed['theta'])
        else:
            covf = lambda d,x: spherical_covariance(d,fixed['sigma'],fixed['theta'])
    else:
        TypeError('Unknown covariance function.')
    return covf


def init_cov_est(cov, fixed, d_max, s2):
    if cov == 'exponential':
        n_pars = 3
        parameter_names = ['sigma','kappa','sigma_e']
    elif cov == 'gaussian':
        n_pars = 3
        parameter_names = ['sigma','rho','sigma_e']
    elif cov == 'matern':
        n_pars = 4
        parameter_names = ['sigma','kappa','nu','sigma_e']
    else:
        TypeError('Unknown covariance function.')
    p0 = np.zeros(n_pars)
    fixed_pars = np.zeros(n_pars)
    if fixed:
        names = list(fixed.keys())
        for i in range(len(names)):
            if names[i] == 'sigma':
                p0[0] = fixed[names[i]]
                fixed_pars[0] = 1
            elif names[i] == 'kappa':
                p0[1] = fixed[names[i]]
                fixed_pars[1] = 1
            elif names[i] == 'nu':
                if cov == 'matern':
                    p0[2] = fixed[names[i]]
                    fixed_pars[2] = 1
            elif names[i] == 'sigma_e':
                p0[-1] = fixed[names[i]]
                fixed_pars[-1] = 1
            else:
                TypeError('Unknown fixed parameter.')
    #Set start value for variance parameters
    if p0[0]== 0:
        if p0[-1] == 0: #no fixed parameters
            p0[0] = np.sqrt(s2*4/5)
            p0[-1] = np.sqrt(s2/5)
        else: #nugget fixed
            s2_start = s2 - p0[-1]**2
            if s2_start < 0:
                #warning('Nugget fixed to value larger than data variance.')
                s2_start = s2/5
            p0[0] = np.sqrt(s2_start)
    else: #variance of field fixed
        if p0[-1] != 0:
            s2_start = s2 - p0[0]**2
            if s2_start < 0:
                s2_start = s2/5
            p0[-1] = np.sqrt(s2_start)
    #set start value for smoothness
    if cov == 'matern':
        if p0[2] == 0:
            p0[2] = 1
    #Set start value for range parameter
    if p0[1] == 0:
        if cov == 'matern':
            p0[1] = np.sqrt(8*p0[2])/(d_max/2)
        elif cov == 'exponential':
            p0[1] = 2/(d_max/2)
        elif cov == 'gaussian':
            p0[1] = d_max/2
    p0 = p0[np.where(fixed_pars==0)[0]]
    names = [parameter_names[i] for i in np.where(fixed_pars==0)[0]]
    return p0, names

def cov_ml_est(
    data: np.ndarray,
    cov: str,
    loc: np.ndarray,
    X: Optional[np.ndarray] = None,
    fixed: Optional[Dict[str, float]] = None,
    reml: bool = False
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Estimate covariance parameters using maximum likelihood.
    
    Parameters
    ----------
    data : numpy.ndarray
        Input data values.
    cov : str
        Covariance model name.
    loc : numpy.ndarray
        Locations of the data points.
    X : numpy.ndarray, optional
        Auxiliary data for regression.
    fixed : dict, optional
        A dictionary of fixed parameters for the covariance model.
    reml : bool, optional
        Whether to use Restricted Maximum Likelihood (REML) estimation.
    
    Returns
    -------
    pars : dict
        Estimated covariance parameters.
    """

    if X is None:
        n_cov = 0
    else:
        n_cov = X.shape[1]

    if X is None and reml:
        raise ValueError('X must be supplied if REML is used.')

    # Select covariance function
    covf = select_covariance(cov, fixed)

    # Compute distance matrix
    D = squareform(pdist(loc))

    # Precompute identity matrix
    I = np.eye(loc.shape[0])

    # Initial regression estimate of beta
    if n_cov > 0:
        e = data - X.dot(np.linalg.solve(X.T.dot(X), X.T.dot(data)))
        s2 = np.var(e)
    else:
        s2 = np.var(data)

    # Set initial values for parameters
    p0, names = init_cov_est(cov, fixed, np.max(D), s2)

    # Define the objective function
    objective = lambda x: log_like(x, covf, data, X, D, I, fixed, n_cov, names, reml)

    # Use the 'minimize' function with the 'Nelder-Mead' and 'TNC' methods instead of 'fmin' and 'fmin_tnc'
    res_nm = minimize(objective, np.log(p0), method='Nelder-Mead')
    res_tnc = minimize(objective, res_nm.x, method='TNC')

    # Extract parameters
    pars = fixed
    for i in range(len(names)):
        pars[names[i]] = np.exp(res_tnc.x[i])

    # Compute regression coefficients
    if names[-1] == 'sigma_e':  # nugget not fixed
        Sigma = covf(D, np.exp(res_tnc.x[:-1])) + np.exp(2 * res_tnc.x[-1]) * I
    else:
        Sigma = covf(D, np.exp(res_tnc.x)) + (fixed['sigma_e'])**2 * I
    if n_cov > 0:
        pars['beta'] = np.linalg.solve(X.T.dot(Sigma).dot(X), X.T.dot(Sigma).dot(data))

    return pars

def WLS_loss(p, variogram, fixed, covf, names):
    #compute variogram
    if names[-1] == 'sigma_e': # nugget not fixed
        r = covf(variogram['h'], np.exp(p[:-1]))
        r0 = np.exp(2*p[-1]) + covf(0,np.exp(p[:-1]))
    else:
        r = covf(variogram['h'][variogram['N']>0],np.exp(p))
        r0 = (fixed['sigma_e'])**2 + covf(0,np.exp(p))
    v = r0 - r
    w = variogram['N']/v**2

    #compute loss function
    mask = variogram['N'] > 0
    S = np.sum(w[mask] * (variogram['variogram'][mask] - v[mask])**2)
    #print("Loss: ", S, "Parameters: ", np.exp(p), "r0: ", r0)
    # write the same print statement above but in matlab code
    
    return S


def cov_ls_est(
    e: np.ndarray,
    cov: str,
    variogram: Dict[str, np.ndarray],
    fixed: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Estimate covariance parameters using weighted least squares.
    
    Parameters
    ----------
    e : numpy.ndarray
        Input data values.
    cov : str
        Covariance model name.
    variogram : dict
        Dictionary containing the variogram information (bin centers, variogram values, and number of pairs per bin).
    fixed : dict, optional
        A dictionary of fixed parameters for the covariance model.
    
    Returns
    -------
    pars : dict
        Estimated covariance parameters.
    """

    if fixed is None:
        fixed = {}
    if not isinstance(fixed, dict):
        raise ValueError('fixed should be a dict.')

    # Select covariance function
    covf = select_covariance(cov, fixed)

    # Set initial values for parameters
    p0, names = init_cov_est(cov, fixed, np.max(variogram['h']), np.var(e))

    # Define the objective function
    objective = lambda x: WLS_loss(x, variogram, fixed, covf, names)

    # Use the 'minimize' function with the 'Nelder-Mead' method instead of 'fmin'
    res = minimize(objective, np.log(p0), method='Nelder-Mead', options={'maxiter': 10000})

    # Extract parameters
    pars = fixed
    for i in range(len(names)):
        pars[names[i]] = np.exp(res.x[i])

    return pars


def normmix_kmeans(x, K, maxiter=300, verbose=0):
    # Parse input parameters
    if maxiter is None:
        maxiter = 1

    if verbose > 0:
        kmeans = KMeans(n_clusters=K, max_iter=maxiter, verbose=1, init='k-means++')
    else:
        kmeans = KMeans(n_clusters=K, max_iter=maxiter, init='k-means++')

    idx = kmeans.fit_predict(x)
    pars = {'mu': [], 'Sigma': [], 'p': np.ones(K) / K}
    n, d = x.shape
    sigma2 = 0

    for k in range(K):
        pars['mu'].append(kmeans.cluster_centers_[k])
        # so y is a vector of the differences between the x values of the cluster and the clusters centers
        y = x[idx == k] - pars['mu'][k]
        sigma2 += np.sum(y ** 2)

    Sigma = sigma2 * np.eye(d) / (n * d)

    for k in range(K):
        pars['Sigma'].append(Sigma)

    return idx, pars

def normmix_posterior(x: np.ndarray, pars: Dict) -> np.ndarray:
    """
    Computes class probabilities for a Gaussian mixture model estimated using normmix_sgd.
    
    Parameters:
    x (np.ndarray): n-by-d matrix with values to be classified
    pars (Dict): result object obtained by running normmix_sgd
    
    Returns:
    np.ndarray: The posterior class probabilities, n-by-K matrix
    """

    n, d = x.shape
    K = len(pars["mu"])

    # Calculate log-probabilities for each class
    p = np.zeros((n, K))

    for k in range(K):
        y = (x.T - pars["mu"][k].reshape(-1, 1))
        v = np.linalg.solve(pars["Sigma"][k], y)
        v = np.sum(v * y, axis=0)
        p[:, k] = -0.5 * v - 0.5 * np.log(np.linalg.det(pars["Sigma"][k])) - (d / 2) * np.log(2 * np.pi)

    # Add prior information and transform to linear scale
    p = p + np.log(pars["p"])
    p = p - np.max(p, axis=1, keepdims=True)
    p = np.exp(p)
    p = p / np.sum(p, axis=1, keepdims=True)

    return p


def normmix_classify(x: np.ndarray, pars: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classifies an image based on a Gaussian mixture model estimated using normmix_sgd.
    
    Parameters:
    x (np.ndarray): n-by-d matrix with values to be classified
    pars (Dict): result object obtained by running normmix_sgd
    
    Returns:
    Tuple[np.ndarray, np.ndarray]: A tuple containing the classes for each pixel (n-by-1 vector) and the posterior class probabilities (n-by-K matrix)
    """

    p = normmix_posterior(x, pars)
    cl = np.argmax(p, axis=1)

    return cl, p



def prior_o_alpha(prior_alpha, in_alpha):
    """
    Internal function that changes between parametrization [p_1, p_2, ...]
    and [alpha_1, alpha_2, ...] where p_i = exp(alpha_i) / sum(exp(alpha_j))
    alpha_1 = 0
    """
    
    if in_alpha:
        alpha = prior_alpha
        prior = np.zeros(len(alpha))
        sum_exp = np.sum(np.exp(alpha))
        
        for i in range(len(prior)):
            prior[i] = np.exp(alpha[i]) / sum_exp
        
        prior_alpha = prior
    else:
        prior = prior_alpha
        alpha = np.zeros(len(prior))
        sum_exp = 1 / prior[0]
        
        for i in range(1, len(prior)):
            alpha[i] = np.log(prior[i] * sum_exp)
        
        prior_alpha = alpha

    return prior_alpha

def duplicatematrix(n):
    """
    Creates the matrix D such that for a symmetric matrix one has
    D * vech*(A) = vec(A)
    where vech*(A) = [a_11 a_22 a_nn a_21 a 31 ... a_n-1n]
    thus first the diagonal entries, then the lower triangular entries
    column stacked
    """
    I = np.where(np.eye(n))
    I2 = np.where(np.tril(np.ones((n, n)), -1))
    I3 = np.where(np.triu(np.ones((n, n)), 1))
    n_I = n
    n_I2 = n * (n - 1) // 2
    rows = np.hstack([I[0], I2[0], I3[1]])
    cols = np.hstack([np.arange(n_I), n_I + np.arange(n_I2), n_I + np.arange(n_I2)])
    data = np.ones(rows.size)
    D = coo_matrix((data, (rows, cols)), shape=(n * n, n_I + n_I2))
    return D

def duplicatematrix_old(n):
    """
    Creates the matrix D such that for a symmetric matrix one has
    D * vech*(A) = vec(A)
    where vech*(A) = [a_11 a_22 a_nn a_21 a 31 ... a_n-1n]
    thus first the diagonal entries, then the lower triangular entries
    column stacked
    """

    I = np.eye(n, dtype=bool)
    I2 = np.tril(np.ones((n, n), dtype=bool), -1)
    I3 = np.triu(np.ones((n, n), dtype=bool), 1)
    n_I = n
    n_I2 = n * (n - 1) // 2

    rows = np.hstack([I.ravel(), I2.ravel(), I3.ravel()])
    cols = np.hstack([np.arange(n_I), n_I + np.arange(n_I2), n_I + np.arange(n_I2)])
    data = np.ones(rows.size)

    D = coo_matrix((data, (rows, cols)), shape=(n * n, n_I + n_I2))

    return D

def gradient_alpha(alpha, p):
    """
    Computes the gradient and search direction of the probabilities using the
    transformation p_i = exp(\alpha_i) / sum (exp(\alpha_j))
    """
    
    g = np.zeros(alpha.shape)
    H = np.eye(len(alpha))
    sum_exp = np.sum(np.exp(alpha))
    p_sum = np.sum(p, axis=0)
    n = p.shape[0]
    g[1:] = p_sum[1:]
    g[1:] = g[1:] - n * np.exp(alpha[1:]) / sum_exp
    H[1:, 1:] = -n * np.diag(np.exp(alpha[1:]) / sum_exp)
    H[1:, 1:] = H[1:, 1:] + n * np.outer(np.exp(alpha[1:]), np.exp(alpha[1:])) / sum_exp ** 2
    p = -np.linalg.solve(H, g)
    g = g / n

    return g, p

def normmix_gradient(pars, y, P):
    """
    Calculates the gradients of the likelihood of a Gaussian mixture model.
    pars is a dictionary with the parameters of the model:
        pars['mu']: a list with the mean values for each class.
        pars['Sigma']: a list with the covariance matrices for each class.
        pars['p']: a list with the class probabilities.
    y is an n x d matrix with the data, and P are the posterior class probabilities.
    """
    
    K = len(pars['mu'])
    n, d = y.shape

    D = duplicatematrix(d)

    step = {'mu': [None] * K, 'Sigma': [None] * K, 'alpha': np.zeros(K)}
    grad = {'mu': [None] * K, 'Sigma': [None] * K, 'alpha': np.zeros(K)}

    for k in range(K):
        Qk = np.linalg.inv(pars['Sigma'][k])
        z_k = P[:, k]
        yc = y - pars['mu'][k]
        Syc = yc @ Qk
        dmu = np.sum(z_k[:, np.newaxis] * Syc, axis=0)
        sum_z_k = np.sum(z_k)
        ddmu = -sum_z_k * Qk

        Syc_z = z_k[:, np.newaxis] * Syc
        dSigma = (Syc_z.T @ Syc) / 2
        dSigma = D.T @ dSigma.ravel()
        dSigma = dSigma - sum_z_k * D.T @ Qk.ravel() / 2
        ddSigma = -(sum_z_k / 2) * D.T @ np.kron(Qk, Qk) @ D

        dmuSigma = Qk @ (np.kron(np.sum(Syc_z, axis=0), np.eye(d)) @ D)

        H = np.block([[ddmu, dmuSigma], [dmuSigma.T, ddSigma]])
        H = (H + H.T) / 2
        e = np.sort(np.linalg.eigvalsh(H))
        b = 1e-12
        if e[-1] / e[0] < b:
            H = H + np.eye(H.shape[0]) * (b * e[0] - e[-1]) / (1 - b)

        grad['mu'][k] = dmu / n
        grad['Sigma'][k] = (D @ dSigma).reshape(d, d) / n
        g_temp = np.concatenate([dmu, dSigma])
        p_temp = -np.linalg.solve(H, g_temp)
        step['mu'][k] = p_temp[:d]
        step['Sigma'][k] = (D @ p_temp[d:]).reshape(d, d)

    alpha = prior_o_alpha(pars['p'], 0)
    g_temp, p_temp = gradient_alpha(alpha, P)
    step['alpha'] = p_temp
    grad['alpha'] = g_temp

    return step, grad


def normmix_loglike(x, pars):
    """
    Compute class probabilities for a Gaussian mixture model.
    
    Input:
    x: n-by-d matrix
    pars: A dictionary with keys 'mu', 'Sigma', and 'p'.
        pars['mu'][k]: 1-by-d matrix, class expected value.
        pars['Sigma'][k]: d-by-d matrix, class covariance.
        pars['p']: 1-by-K matrix with the class probabilities.
        
    Output:
    p: The posterior class probabilities, n-by-K matrix.
    """
    n, d = x.shape
    K = len(pars['mu'])

    # calculate log-probabilities for each class
    p = np.zeros(n)

    for k in range(K):
        p += pars['p'][k] * multivariate_normal.pdf(x, mean=pars['mu'][k], cov=pars['Sigma'][k])
        
    ll = np.sum(np.log(p))
    
    return p, ll


def vech(m):
    """
    h = vech(m)
    h is the column vector of elements on or below the main diagonal of m.
    m must be square.
    """
    rows, cols = m.shape
    r = np.repeat(np.arange(1, rows + 1)[:, np.newaxis], cols, axis=1)
    c = np.repeat(np.arange(1, cols + 1)[np.newaxis, :], rows, axis=0)
    
    # c <= r is same as np.tril(np.ones(m.shape))
    h = m[c <= r]
    
    return h


def normmix_sgd(x: np.ndarray, K: int, Niter: Optional[int] = 100, step0: Optional[float] = 1, plotflag: Optional[int] = 0) -> Tuple[Dict, Dict]:
    """
    Uses gradient-descent optimization to estimate a Gaussian mixture model.
    
    Parameters:
    x (np.ndarray): n-by-d matrix (column-stacked image with n pixels)
    K (int): number of classes
    Niter (int): number of iterations to run the algorithm
    step0 (float): the initial step size
    plotflag (int): if 1, then parameter tracks are plotted
    
    Returns:
    Tuple[Dict, Dict]: A tuple containing the estimated parameters and the parameter trajectories
    """
    
    # Note: You will need to implement the following functions: normmix_kmeans, normmix_posterior, normmix_gradient, prior_o_alpha, and normmix_loglike
    # These functions have not been provided in this code snippet

    n, d = x.shape

    # Obtain some reasonable starting values using K-means
    idx, pars = normmix_kmeans(x, K, Niter)


    for i in range(Niter):
        # Compute posterior probabilities
        p_tmp = normmix_posterior(x, pars)

        # Compute gradients
        step, grad = normmix_gradient(pars, x, p_tmp)

        # Take step
        gamma = step0
        for k in range(K):
            pars['Sigma'][k] = pars['Sigma'][k] + gamma * step['Sigma'][k]
            pars['Sigma'][k] = (pars['Sigma'][k] + pars['Sigma'][k].T) / 2  # Make sure symmetric
            pars['mu'][k] = pars['mu'][k] + gamma * step['mu'][k]

        alpha = prior_o_alpha(pars['p'], 0)
        alpha = alpha + gamma * step['alpha']
        pars['p'] = prior_o_alpha(alpha, 1).T



    return pars













# matlab standard colormap
cm_data = [[0.2081, 0.1663, 0.5292], [0.2116238095, 0.1897809524, 0.5776761905], 
 [0.212252381, 0.2137714286, 0.6269714286], [0.2081, 0.2386, 0.6770857143], 
 [0.1959047619, 0.2644571429, 0.7279], [0.1707285714, 0.2919380952, 
  0.779247619], [0.1252714286, 0.3242428571, 0.8302714286], 
 [0.0591333333, 0.3598333333, 0.8683333333], [0.0116952381, 0.3875095238, 
  0.8819571429], [0.0059571429, 0.4086142857, 0.8828428571], 
 [0.0165142857, 0.4266, 0.8786333333], [0.032852381, 0.4430428571, 
  0.8719571429], [0.0498142857, 0.4585714286, 0.8640571429], 
 [0.0629333333, 0.4736904762, 0.8554380952], [0.0722666667, 0.4886666667, 
  0.8467], [0.0779428571, 0.5039857143, 0.8383714286], 
 [0.079347619, 0.5200238095, 0.8311809524], [0.0749428571, 0.5375428571, 
  0.8262714286], [0.0640571429, 0.5569857143, 0.8239571429], 
 [0.0487714286, 0.5772238095, 0.8228285714], [0.0343428571, 0.5965809524, 
  0.819852381], [0.0265, 0.6137, 0.8135], [0.0238904762, 0.6286619048, 
  0.8037619048], [0.0230904762, 0.6417857143, 0.7912666667], 
 [0.0227714286, 0.6534857143, 0.7767571429], [0.0266619048, 0.6641952381, 
  0.7607190476], [0.0383714286, 0.6742714286, 0.743552381], 
 [0.0589714286, 0.6837571429, 0.7253857143], 
 [0.0843, 0.6928333333, 0.7061666667], [0.1132952381, 0.7015, 0.6858571429], 
 [0.1452714286, 0.7097571429, 0.6646285714], [0.1801333333, 0.7176571429, 
  0.6424333333], [0.2178285714, 0.7250428571, 0.6192619048], 
 [0.2586428571, 0.7317142857, 0.5954285714], [0.3021714286, 0.7376047619, 
  0.5711857143], [0.3481666667, 0.7424333333, 0.5472666667], 
 [0.3952571429, 0.7459, 0.5244428571], [0.4420095238, 0.7480809524, 
  0.5033142857], [0.4871238095, 0.7490619048, 0.4839761905], 
 [0.5300285714, 0.7491142857, 0.4661142857], [0.5708571429, 0.7485190476, 
  0.4493904762], [0.609852381, 0.7473142857, 0.4336857143], 
 [0.6473, 0.7456, 0.4188], [0.6834190476, 0.7434761905, 0.4044333333], 
 [0.7184095238, 0.7411333333, 0.3904761905], 
 [0.7524857143, 0.7384, 0.3768142857], [0.7858428571, 0.7355666667, 
  0.3632714286], [0.8185047619, 0.7327333333, 0.3497904762], 
 [0.8506571429, 0.7299, 0.3360285714], [0.8824333333, 0.7274333333, 0.3217], 
 [0.9139333333, 0.7257857143, 0.3062761905], [0.9449571429, 0.7261142857, 
  0.2886428571], [0.9738952381, 0.7313952381, 0.266647619], 
 [0.9937714286, 0.7454571429, 0.240347619], [0.9990428571, 0.7653142857, 
  0.2164142857], [0.9955333333, 0.7860571429, 0.196652381], 
 [0.988, 0.8066, 0.1793666667], [0.9788571429, 0.8271428571, 0.1633142857], 
 [0.9697, 0.8481380952, 0.147452381], [0.9625857143, 0.8705142857, 0.1309], 
 [0.9588714286, 0.8949, 0.1132428571], [0.9598238095, 0.9218333333, 
  0.0948380952], [0.9661, 0.9514428571, 0.0755333333], 
 [0.9763, 0.9831, 0.0538]]

parula_map = LinearSegmentedColormap.from_list('parula', cm_data)





