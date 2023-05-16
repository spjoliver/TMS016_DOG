"""
File that creates the iriscodes from a folder of iris images.
Simply run with 
$ python3 create_iriscodes.py

"""

import os
from util import transform_cropped_iris
from IrisSegmentation import FastIrisPupilScanner



ROOTDIR = "UTIRIS_infrared/"
SAVEDIR = "iriscodes/"


def main():
    print("Creating codes.")
    create_iriscodes(ROOTDIR, SAVEDIR)
    


def create_iriscodes(rootdir, savedir, alphas=[0.4], betas=[2.5], omegas=[4],
                     code_shape = [8, 8], patch_shape = [15, 15]):
    n_rho_patches = code_shape[0]
    n_theta_patches = code_shape[1]
    rho_psize = patch_shape[0]
    theta_psize = patch_shape[1]
    codes = []
    names = []
    i = 1
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if file.endswith(".bmp"):
                #print(file)
                print(f"iter {i}...")
                i+=1
                image_name = os.path.join(subdir, file)
                try:
                    out = FastIrisPupilScanner(image_name)
                except IndexError:
                    print(f"Warning, IndexError in FastIrisPupilScanner for {image_name}")
                    continue
                try:
                    transf_image = transform_copped_iris(out['iris']/255, 
                                                     out['pupil_xy'], out['pupil_r'],
                                                     out['iris_xy'], out['iris_r'], 
                                                     theta_res = theta_psize*n_theta_patches,
                                                     rho_res = rho_psize*n_rho_patches)
                except TypeError:
                    print(f"Warning, TypeError in transform_copped_iris for {image_name}, pupil: c{out['pupil_xy']} r{out['pupil_r']}, iris: c{out['iris_xy']} r{out['iris_r']}")
                    continue
                code = calculate_iriscode_different_filters(transf_image, 
                                         alphas=alphas, betas=betas, omegas=omegas,
                                         theta_psize = theta_psize, rho_psize = rho_psize,
                                         pairwise=True)
                codes.append(code)
                names.append(f"{savedir}{file[:-4]}.npy")
                if i%100 == 0:
                    assert(len(codes) == len(names))
                    for code, name in zip(codes, names):
                        np.save(code, name)
                    codes = []
                    names = []


if __name__ == "__main__":
    main()
