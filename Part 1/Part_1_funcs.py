import numpy as np
from matlab_functions import emp_variogram, cov_ls_est, matern_variogram, stencil2prec, stencil2prec_2, parula_map
from matplotlib import pyplot as plt
from sksparse.cholmod import cholesky as cholesky_sparse
from scipy.sparse.linalg import spsolve, cg
from typing import Optional
from scipy.optimize import minimize
from time import perf_counter


def data_setup(
        data: np.ndarray, 
        p: float, 
        uo: int
        ):
    """
    p = proportion of pixels to be regarded as missing
    uo = number of observed pixels to be used for estimation of parameters
    returns: 
        covariates_o_used, observed_values_used, loc_o_used, observed_values, covariates_o, covariates_m, index_o, index_m, index_o_used
    """
    # add X and Y coordinate values
    xmax = data.shape[0]
    ymax = data.shape[1]
    spread_x = np.arange(0, xmax)
    spread_y = np.arange(0, ymax)
    X, Y = np.meshgrid(spread_x, spread_y)
    X_cov = X.ravel()
    Y_cov = Y.ravel()

    covariates = np.vstack((np.ones((X_cov.shape[0])), X_cov, Y_cov)).T

    index = np.arange(0, data.shape[0]*data.shape[1])
    index_o = np.random.choice(index, size=round(index.shape[0]*p), replace=False)
    index_o_used = index_o[:uo]
    # Could do smarter sampling above, i.e. instead shuffle tindex as: np.random.shuffle(tindex)
    # and then take the first round(tindex.shape[0]*tp) elements
    # but setdiff1d seems like a good function to know about.
    index_m = np.setdiff1d(index, index_o)
    index = None
    # order="F" is extremely important here. OH MA FUKIN GAD.
    data_flat = data.ravel(order="F")
    observed_values = data_flat[index_o]
    observed_values_used = data_flat[index_o_used]
    #missing_values = data_flat[index_m]

    # dist matrix and stuff for variogram estimation
    loc = np.column_stack((X_cov, Y_cov))

    #loc_o = loc[index_o, :]
    loc_o_used = loc[index_o_used, :]
    #loc_m = loc[index_m, :]
    covariates_o = covariates[index_o, :]
    covariates_o_used = covariates[index_o_used, :]
    covariates_m = covariates[index_m, :]
    return covariates_o_used, observed_values_used, loc_o_used, observed_values, covariates_o, covariates_m, index_o, index_m


def lse_regparam_est(
        covariates_o_used: np.ndarray, 
        covariates_o: np.ndarray, 
        covariates_m: np.ndarray, 
        observed_values_used: np.ndarray, 
        print_est: bool = True
        ):
    """
    returns:
        lse, e, mu_o, mu_m
    """

    lse = np.linalg.solve(covariates_o_used.T @ covariates_o_used, covariates_o_used.T @ observed_values_used)
    mu_ou = covariates_o_used @ lse
    mu_o = covariates_o @ lse
    mu_m = covariates_m @ lse
    e = observed_values_used - mu_ou
    mu_ou = None
    if print_est:
        print("Estimated regression parameters: ", lse)
    return lse, e, mu_o, mu_m

def get_emp_var(
        loc_o_used: np.ndarray, 
        e: np.ndarray
        ):
    """
    returns:
        emp_v
    """
    return emp_variogram(loc_o_used, e, 100)

def estimate_var_params(
        emp_v: np.ndarray, 
        e: np.ndarray, 
        nu_fixed: bool = False, 
        print_est: bool = True
        ):
    """
    
    returns:
        lse_estimates, mat_v_new
    """
    # they say to fix nu to 1.0, but I don't know why, and it seems to make the fit worse
    #Least-squares estimation of a Matern variogram to the binned estimate

    if nu_fixed:
        lse_estimates = cov_ls_est(e, 'matern' , emp_v, {"nu": 1.0})
        mat_v_new = matern_variogram(emp_v["h"],lse_estimates["sigma"],lse_estimates["kappa"], lse_estimates["nu"], lse_estimates["sigma_e"])
    else:
        lse_estimates = cov_ls_est(e, 'matern' , emp_v)
        mat_v_new = matern_variogram(emp_v["h"],lse_estimates["sigma"],lse_estimates["kappa"], lse_estimates["nu"], lse_estimates["sigma_e"])
    
    if print_est:
        print(f"Estimated variogram parameters (nu_fixed={nu_fixed}): ", lse_estimates)
    
    plot_variogram(emp_v, mat_v_new)
    mat_v_new = None

    return lse_estimates

def plot_variogram(
        emp_v: np.ndarray, 
        mat_v_new: np.ndarray
        ):

    plt.rcParams['figure.figsize'] = [10, 7.5]
    plt.rcParams['figure.dpi'] = 100
    plt.plot(emp_v["h"], mat_v_new, color="orange")
    plt.plot(emp_v["h"], emp_v["variogram"],'o')
    plt.legend(["Estimated variogram", "Binned estimate"])
    plt.show()


def get_variogram_estimates_and_plot(
        data: np.ndarray, 
        p: float, 
        uo: int, 
        nu_fixed: bool = False
        ):
    covariates_o_used, observed_values_used, loc_o_used, missing_values, observed_values, loc_o, loc_m, covariates_o, covariates_m, index_o, index_m = data_setup(data, p, uo)
    lse, e = lse_regparam_est(covariates_o_used, covariates_o, covariates_m, observed_values_used)
    emp_v = get_emp_var(loc_o_used, e)
    lse_estimates, mat_v_new = estimate_var_params(emp_v, e, nu_fixed)
    plot_variogram(emp_v, mat_v_new)
    return lse_estimates, mat_v_new

def get_Q(
        lse_est: dict, 
        data_shape: tuple, 
        show_mid_cov: bool = True,
        kappa_setting: Optional[float] = None
        ):

    kappa = lse_est["kappa"] if kappa_setting is None else kappa_setting
    sigma = lse_est["sigma"]
    tau = (2 * np.pi) / sigma**2
    # hardcode 5x5 numpy array of zeros
    q1 = np.array(
        [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
        ]
    )

    q2 = np.array(
        [
        [0, 0, 0, 0, 0],
        [0, 0, -1, 0, 0],
        [0, -1, 4, -1, 0],
        [0, 0, -1, 0, 0],
        [0, 0, 0, 0, 0]
        ]
    )
    q3 = np.array(
        [
        [0, 0,   1,   0, 0],
        [0, 2,  -8,   2, 0],
        [1, -8, 20,  -8, 1],
        [0, 2,  -8,   2, 0],
        [0, 0,   1,   0, 0]
        ]
    )
    q = (kappa**4)*q1 + 2*(kappa**2)*q2 + q3
    
    print("Q is about to be calculated")
    tstart = perf_counter()
    Q = tau * stencil2prec([data_shape[0], data_shape[1]], q)
    print("Q was calculated in ", perf_counter() - tstart, " seconds")
    print("Q is calculated")
    if show_mid_cov:# and data_shape[0]*data_shape[1] < 90000:
        v = np.zeros(data_shape[0]*data_shape[1])
        v[(data_shape[0]*data_shape[1])//2 - 750] = 1

        #c = spsolve(Q, v)
        print("Covariance plot is about to be calculated")
        tstart = perf_counter()
        c, res = cg(Q, v)
        print("Covariance plot was calculated in ", perf_counter() - tstart, " seconds")
        
        print("Sparse cg res (covariance plot): ", res)
        plt.imshow(c.reshape([data_shape[0], data_shape[1]], order='F'))
        plt.title("Covariance between the middle pixel and all other pixels")
        plt.show()
    
    return Q, kappa



def reconstruct_data(
        ind_o: np.ndarray,
        ind_m: np.ndarray, 
        mu_m: np.ndarray, 
        mu_o: np.ndarray,
        Q: np.ndarray, 
        observed_values: np.ndarray,
        data: np.ndarray,
        kappa: float,
        plot_reconstruction: bool = True,
        print_mse: bool = True
        ):
    """
    Reconstructs the data using the given Q, mu_o, mu_m, and observed values
    """
    data_shape = data.shape

    #mu_m_o = mu_m - spsolve(Q_m, Q_om.T @ (x_obs - mu_o))
    print("Kriging is about to be calculated")
    tstart = perf_counter()
    #mu_m_o = mu_m - spsolve(Q[ind_m, :][:, ind_m], Q[ind_m, :][:, ind_o] @ (observed_values - mu_o))
    mu_m_o, res = mu_m - cg(Q[ind_m, :][:, ind_m], Q[ind_m, :][:, ind_o] @ (observed_values - mu_o))
    print("Kriging was calculated in ", perf_counter() - tstart, " seconds")
    print("Sparse cg res (covariance plot): ", res)
    print("Kriging is calculated")

    x_rec = np.zeros((data_shape[0]*data_shape[1]))
    x_rec[ind_o] = observed_values
    x_rec[ind_m] = mu_m_o
    x_seen = np.zeros((data_shape[0]*data_shape[1]))
    x_seen[ind_o] = observed_values

    MSE = np.mean((x_rec - data.ravel(order="F"))**2)
    if print_mse:
        print("-"*50)
        print(f"MSE for kappa={kappa} and proportion of seen observations={ind_o.shape[0]/(x_seen.shape[0])}: {MSE}")
        print("-"*50)
    if plot_reconstruction:

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

        ax1.imshow(x_seen.reshape([data_shape[0], data_shape[1]], order='F'), cmap="gray")
        ax1.set_title("Observed data")

        ax2.imshow(x_rec.reshape([data_shape[0], data_shape[1]], order='F'), cmap="gray")
        ax2.set_title("Reconstructed data")

        ax3.imshow(data, cmap="gray")
        ax3.set_title("Simulated/True data")
        plt.show()
    
    return MSE

def minimize_mse(kappa, index_o, index_m, mu_m, mu_o, observed_values, data, lse_est):

    Q = get_Q(lse_est, data.shape, show_mid_cov=False, kappa_setting=kappa)[0]

    return reconstruct_data(
        index_o, 
        index_m, 
        mu_m, 
        mu_o, 
        Q, 
        observed_values, 
        data, 
        kappa, 
        plot_reconstruction=False, 
        print_mse=False
        )

def run_reconstruction(
        data: np.ndarray, 
        p: float, 
        uo: int, 
        nu_fixed: bool = False, 
        show_mid_cov: bool = True, 
        plot_reconstruction: bool = True,
        kappa_setting: Optional[float] = None,
        find_min_kappa: Optional[bool] = False
        ):
    """
    Runs the reconstruction process
    """
    covariates_o_used, observed_values_used, loc_o_used, observed_values, covariates_o, covariates_m, index_o, index_m = data_setup(data, p, uo)
    lse, e, mu_o, mu_m = lse_regparam_est(covariates_o_used, covariates_o, covariates_m, observed_values_used)
    emp_v = get_emp_var(loc_o_used, e)
    lse_estimates= estimate_var_params(emp_v, e, nu_fixed)
    if find_min_kappa:
        objective = lambda kappa: minimize_mse(kappa, index_o, index_m, mu_m, mu_o, observed_values, data, lse_estimates)
        optimal_kappa = minimize(fun=objective, x0=np.array(0.5), method="Nelder-Mead") #, bounds=[(0, np.inf)]
        print(f"Optimal kappa: {optimal_kappa.x[0]}")
        Q, kappa = get_Q(lse_estimates, data.shape, show_mid_cov, optimal_kappa.x[0])
        reconstruct_data(index_o, index_m, mu_m, mu_o, Q, observed_values, data, optimal_kappa.x[0], plot_reconstruction)
    else:
        Q, kappa = get_Q(lse_estimates, data.shape, show_mid_cov, kappa_setting)
        reconstruct_data(index_o, index_m, mu_m, mu_o, Q, observed_values, data, kappa, plot_reconstruction)