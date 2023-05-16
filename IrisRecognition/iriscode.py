import numpy as np
import matplotlib.pyplot as plt


"""
This is the definition of a 2D Gabor wavelet in polar coordinates.
theta0, r0 are anchor points,
omega, alpha, beta are parameters,
rho and phi are the polar coordinates in wavelet space.
"""
def gabor_wavelet_2D(rho, phi, theta0, r0, omega, alpha, beta):
    return np.exp( -1j*omega * (theta0 - phi) ) * np.exp( -(rho - r0)**2 / alpha**2 ) * np.exp(-(phi - theta0)**2 / beta**2)


"""
Calculates the double integral from Daugman for a given normalized 
image and some fixed value (theta0, r0). By taking the sign of the 
real and imaginary part of the returned number one can form the bits.
"""
def project_gabor(tranf_img, theta0, r0, omega, alpha, beta):
    n_rho = tranf_img.shape[0]
    n_phi = tranf_img.shape[1]
    ph = np.linspace(0., 2*np.pi, n_phi)
    rh = np.linspace(0., 1., n_rho)
    Phi, Rho = np.meshgrid(ph, rh)
    return np.mean(Rho * tranf_img * gabor_wavelet_2D(Rho, Phi, omega, theta0, r0, alpha, beta))


"""
Produces the iriscode from greyscale image img.
theta_psize=15, r_psize=15 are odd number so that the center point is 
included in the window.
alpha ranges from 0.15 to 1.2 mm of the iris, which is about 12 mm, and 
should therefore be between 0.1 and 0.01. 
omega is supposed to span three octaves and is inversely proportional
to beta.
"""
def calculate_iris_code(transf_img, theta_psize=15, r_psize=15, 
                        alpha=0.4, beta=2.5, omega=4):
    # Impose the demand that the dimensions of the transfromed iris is 
    # divisible by the patch sizes.
    assert(transf_img.shape[0]%r_psize == 0) 
    assert(transf_img.shape[1]%theta_psize == 0)
    
    # Create the code matrix. Each row is a different w.
    iriscode = np.zeros([transf_img.shape[0]//r_psize, 
                         transf_img.shape[1]//theta_psize * 2])
    
    norm_img = (transf_img - np.mean(transf_img))/np.std(transf_img)
    # The iriscode is calculated from each patch
    for i in range(transf_img.shape[0]//r_psize):
        for j in range(transf_img.shape[1]//theta_psize):
            pi = i*r_psize
            pj = j*theta_psize
            patch = norm_img[np.ix_(np.arange(pi, pi+r_psize), np.arange(pj, pj+theta_psize))]
            h = project_gabor(patch, np.pi, 0.5, omega, alpha, beta)
            iriscode[i, j*2] = np.real(h)
            iriscode[i, j*2+1] = np.imag(h)
    iriscode[iriscode >= 0] = 1
    iriscode[iriscode < 0] = 0
    return iriscode


"""
Cycles the iriscode theta wise. Allows for stacked thetas on the column direction.
Takes the iriscode and n_theta_patches used when calculating the code as input.
"""
def cycle_iriscode(iriscode, n_theta_patches):
    n = iriscode.shape[1]
    new_code = np.zeros(iriscode.shape)
    assert(n%n_theta_patches == 0)
    repeats = n//(n_theta_patches*2)
    print(repeats)
    for i in range(repeats):
        start = i*n_theta_patches*2
        print(start)
        new_code[:, start:start+2*1] = iriscode[:, start + 2*7:start + 2*8]
        new_code[:, start + 2*1: start + 2*8] = iriscode[:, start:start + 2*7]
    return new_code


"""
This function will create iriscodes using different filters. The codes are concatenated column-wise
meaning that each filter is separated over (transf_img.shape[1]/theta_psize)*2 columns.
Pairwise determines if the filters are chosen for all pairwise combinations (equal to len(alpha) codes) 
or over all len(alphas)*len(betas)*len(omegas) combinations if False.
"""
def calculate_iriscode_different_filters(transf_img, 
                                         alphas=[0.4, 0.8], betas=[2.5, 1.25], omegas=[4, 8],
                                         theta_psize = 15, rho_psize = 15,
                                         pairwise=True):
    iriscodes = []
    if pairwise:
        assert(len(alphas) == len(betas) and len(betas) == len(omegas))
        for alpha, beta, omega in zip(alphas, betas, omegas):
            code = calculate_iris_code(transf_img, theta_psize=theta_psize, r_psize=rho_psize, 
                                alpha=alpha, beta=beta, omega=omega)
            iriscodes.append(code)
        return np.concatenate(iriscodes, axis = 1)
    else:
        for alpha in alphas:
            for beta in betas:
                for omega in omegas:
                    code = calculate_iris_code(transf_img, theta_psize=theta_psize, r_psize=rho_psize, 
                                alpha=alpha, beta=beta, omega=omega)
                    iriscodes.append(code)
        return np.concatenate(iriscodes, axis = 1)


