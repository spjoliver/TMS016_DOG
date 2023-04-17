from scipy.special import kv, betainc
from scipy.special import gammaln as lgamma
from scipy import sparse
from scipy.optimize import fmin, fmin_tnc
import numpy as np
from scipy.spatial.distance import squareform, pdist
from matplotlib.colors import LinearSegmentedColormap
from scipy.sparse import linalg as splinalg
import sys

def emp_variogram(D, data, N):
    if D.shape[0] != D.shape[1]:
        D = squareform(pdist(D))
    max_dist = D.max()
    d = np.linspace(0, max_dist, N)
    D[np.tril_indices(D.shape[0], k=-1)] = -1
    
    out = {}
    out['h'] = (d[1:] + d[:-1]) / 2
    out['variogram'] = np.zeros(N-1)
    out['N'] = np.zeros(N-1)
    
    for i in range(N-1):
        I, J = np.where((d[i] < D) & (D <= d[i+1]))
        out['N'][i] = len(I)
        out['variogram'][i] = 0.5 * np.mean((data[I] - data[J])**2)
    return out

def stencil2prec(sz, q):
    II = []
    KK = []
    JJ_I = []
    JJ_J = []

    I, J = np.meshgrid(np.arange(1, 1 + sz[0]), np.arange(1, 1 + sz[1]), indexing='ij')
    I = I.flatten()
    J = J.flatten()

    ones_arr = np.ones((sz[0]*sz[1], 1))
    for i in range(1, 1 + q.shape[0]):
        for j in range(1, 1 + q.shape[1]):
            if q[i-1,j-1] != 0:
                II.append(I + sz[0]*(J-1))
                JJ_I.append(I + i - int((q.shape[0]+1)/2))
                JJ_J.append(J + j - int((q.shape[1]+1)/2))
                KK.append(q[i-1,j-1] * ones_arr)

    II = np.concatenate(II)[:, np.newaxis]
    JJ_I = np.concatenate(JJ_I)[:, np.newaxis]
    JJ_J = np.concatenate(JJ_J)[:, np.newaxis]
    KK = np.concatenate(KK)
    JJ = JJ_I + sz[0]*(JJ_J-1)
    ok = (JJ_I >= 1) & (JJ_I <= sz[0]) & (JJ_J >= 1) & (JJ_J <= sz[1])
    II = II[ok]
    JJ = JJ[ok]
    KK = KK[ok]
    Q = sparse.coo_matrix((KK, (II-1, JJ-1)), shape=(np.prod(sz), np.prod(sz)))
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


def log_like(p, covf, data, X, D, I, fixed, n_cov, names, reml):
    #compute covariance matrix 
    if names[-1] == 'sigma_e': # nugget not fixed
        Sigma = covf(D,np.exp(p[:-1])) + np.exp(2*p[-1])*I
    else:
        Sigma = covf(D,np.exp(p)) + (fixed['sigma_e'])**2*I
    
    #compute Cholesky factor, if it fails, return large value
    R,p = np.linalg.cholesky(Sigma)
    if p > 0:
        ll = np.inf
        return ll
    
    SiY = np.linalg.solve(R,np.linalg.solve(R.T,data))
    
    if n_cov > 0:
        SiX = np.linalg.solve(R,np.linalg.solve(R.T,X))
        v = SiY - SiX.dot(np.linalg.solve(X.T.dot(SiX),X.T.dot(SiY)))
    else:
        v = SiY
    ll = -np.sum(np.log(np.diag(R))) -0.5*data.T.dot(v)
    
    if reml == 1 and n_cov > 0:
        Rreml,p = np.linalg.cholesky(X.T.dot(SiX))
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
        names = fixed.keys()
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
        names = fixed.keys()
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


def cov_ml_est(data, cov, loc, X=None, fixed=None, reml=False):
        
    if X is None:
        n_cov = 0
    else:
        n_cov = X.shape[1]
        
    if X is None and reml:
        raise ValueError('X must be supplied if REML is used.')
        
    #select covariance function
    covf = select_covariance(cov, fixed)
    
    #compute distance matrix
    D = squareform(pdist(loc))
    
    #precompute identiy matrix
    I = np.eye(loc.shape[0])
    
    #initial regression estimate of beta
    if n_cov > 0:
        e = data - X.dot(np.linalg.solve(X.T.dot(X),X.T.dot(data)))
        s2 = np.var(e)
    else:
        s2 = np.var(data)
    
    #set initial values for parameters
    p0, names = init_cov_est(cov, fixed, np.max(D), s2)
    
    #minimize loss function 
    par, _, _, _, _, _ = fmin(lambda x: log_like(x, covf, data, X, D, I, fixed, n_cov, names, reml), np.log(p0))
    par, _, _ = fmin_tnc(lambda x: log_like(x, covf, data, X, D, I, fixed, n_cov, names, reml), par)
    
    #extract parameters
    pars = fixed
    for i in range(len(names)):
        pars[names[i]] = np.exp(par[i])
    
    #compute regression coefficients
    if names[-1] == 'sigma_e': # nugget not fixed
        Sigma = covf(D,np.exp(par[:-1])) + np.exp(2*par[-1])*I
    else:
        Sigma = covf(D,np.exp(par)) + (fixed['sigma_e'])**2*I
    if n_cov > 0:
        pars['beta'] = np.linalg.solve(X.T.dot(Sigma).dot(X),X.T.dot(Sigma).dot(data))
    
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
    S = np.sum(w[variogram['N']>0]*(variogram['variogram'][variogram['N']>0]-v[variogram['N']>0])**2)
    #print("Loss: ", S, "Parameters: ", np.exp(p), "r0: ", r0)
    # write the same print statement above but in matlab code
    
    return S


def cov_ls_est(e, cov, variogram, fixed=None):
        
    if fixed is None:
        fixed = {}
    if not isinstance(fixed, dict):
        raise ValueError('fixed should be a dict.')
        
    #select covariance function
    covf = select_covariance(cov, fixed)
    
    #set initial values for parameters
    p0, names = init_cov_est(cov, fixed, np.max(variogram['h']), np.var(e))

    par = fmin(lambda x: WLS_loss(x, variogram, fixed, covf, names), np.log(p0), maxiter=10000)

    #extract parameters
    pars = fixed
    for i in range(len(names)):
        pars[names[i]] = np.exp(par[i])
    
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