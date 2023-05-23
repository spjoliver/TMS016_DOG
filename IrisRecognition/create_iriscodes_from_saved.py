"""
File that creates the iriscodes from a folder of iris images.
Simply run with 
$ python3 create_iriscodes_from_saved.py
Change resolution of the iris codes by altering the input parameters 
to the create_iriscodes function.
"""

import os
import pickle
import numpy as np

from util import transform_cropped_iris
from IrisSegmentationFinal import FastIrisPupilScanner2
from iriscode import calculate_iriscode_different_filters


ROOTDIR = "UTIRIS_infrared_segmented/"
SAVEDIR = "iriscodes/"


def main():
    print("Creating codes.")
    create_iriscodes(ROOTDIR, SAVEDIR,
                     alphas=[0.4, 0.4], betas=[2.5, 2.5], omegas=[4, 2],
                     code_shape = [8, 32], patch_shape = [30, 30])


"""
Creates iris codes from images in rootdir and saves them into savedir.
alphas, betas, omegas are parameters to the Gabor wavelet used to
convolve the cropped iris. They are required to have the same length
and multiple filters are concatenated as
   code = [code(alphas[0], betas[0], omegas[0]), code(alphas[1], betas[1], omegas[1]), ...] 
code_shape is the [rho, phi] dimension of the iriscode.
patch_shape is the [px_height, px_width] of each patch of the iriscode. 
"""
def create_iriscodes(rootdir, savedir,
                     alphas=[10, 0.4, 0.4], betas=[0.1, 2.5, 2.5], omegas=[4, 4, 2],
                     code_shape = [8, 16], patch_shape = [30, 30]):
    n_rho_patches = code_shape[0]
    n_theta_patches = code_shape[1]
    rho_psize = patch_shape[0]
    theta_psize = patch_shape[1]
    codes = []
    names = []
    i = 1
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if file.endswith(".pkl"):
                print(file)
                print(f"iter {i}...")
                i+=1
                image_name = os.path.join(subdir, file)
                try:                    
                    with open(image_name, "rb") as f:
                        out = pickle.load(f)
                        if out is None:
                            raise TypeError
                except TypeError:
                    print(f"Warning {image_name} read as None")
                    continue
                try:
                    transf_image = transform_cropped_iris(out['iris']/255, 
                                                     out['pupil_xy'], out['pupil_r'],
                                                     out['iris_xy'], out['iris_r'], 
                                                     theta_res = theta_psize*n_theta_patches,
                                                     rho_res = rho_psize*n_rho_patches)
                except TypeError:
                    print(f"Warning, TypeError in transform_cropped_iris for {image_name}, pupil: c{out['pupil_xy']} r{out['pupil_r']}, iris: c{out['iris_xy']} r{out['iris_r']}")
                    continue
                except IndexError:
                    print(f"Warning, IndexError in transform_copped_iris for {image_name}")
                    continue
                except Exception:
                    print("Warning, max iter reached. The image segmentation or normalization probably failed.")
                    continue
                
                code = calculate_iriscode_different_filters(transf_image, 
                                         alphas=alphas, betas=betas, omegas=omegas,
                                         theta_psize = theta_psize, rho_psize = rho_psize,
                                         pairwise=True)
                codes.append(code)
                names.append(f"{savedir}{file[:-4]}.npy")
                if i%100 == 0:
                    assert(len(codes) == len(names))
                    for name, code in zip(names, codes):
                        np.save(name, code)
                    codes = []
                    names = []
                    
    assert(len(codes) == len(names))
    for name, code in zip(names, codes):
        np.save(name, code)

if __name__ == "__main__":
    main()
