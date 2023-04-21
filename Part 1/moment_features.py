import numpy as np
from typing import Union


def generate_moments(img: np.ndarray, moment_choice: Union[list, None], p_to: int=5, q_to: int=5) -> np.ndarray:

    row_index_matrix = np.repeat(np.arange(img.shape[0])[:, np.newaxis], axis=1, repeats=img.shape[1])
    col_index_matrix = np.repeat(np.arange(img.shape[1])[np.newaxis, :].T, axis=1, repeats=img.shape[0]).T

    moments = []
    if moment_choice is not None:
        for p, q in moments:
            moments.append(((row_index_matrix**p)*(col_index_matrix**q)*img).sum())
    else:
        for p in range(p_to):
            for q in range(q_to):
                moments.append(((row_index_matrix**p)*(col_index_matrix**q)*img).sum())
    
    return np.array(moments)

def generate_central_moments(img: np.ndarray, moment_choice: Union[list, None], p_to: int=5, q_to: int=5) -> np.ndarray:

    centroid_moments = generate_moments(
        img, 
        moment_choice=[(0, 0), (1, 0), (0, 1)]
        )
    x_bar = centroid_moments[1] / centroid_moments[0]
    y_bar = centroid_moments[2] / centroid_moments[0]

    row_index_matrix = np.repeat(np.arange(img.shape[0])[:, np.newaxis], axis=1, repeats=img.shape[1]) - x_bar
    col_index_matrix = np.repeat(np.arange(img.shape[1])[np.newaxis, :].T, axis=1, repeats=img.shape[0]).T - y_bar

    moments = []
    if moment_choice is not None:
        for p, q in moments:
            moments.append(((row_index_matrix**p)*(col_index_matrix**q)*img).sum())
    else:
        for p in range(p_to):
            for q in range(q_to):
                moments.append(((row_index_matrix**p)*(col_index_matrix**q)*img).sum())
    
    return np.array(moments)


def generate_hu_moments(img: np.ndarray) -> np.ndarray:

    central_moments = generate_central_moments(
        img, 
        p_to=3,
        q_to=3
        ).reshape(3, 3)
    scm = np.zeros((3, 3))
    for p in range(3):
        for q in range(3):
            scm[p, q] = central_moments[p, q] / (central_moments[0, 0]**(1 + ((p + q) / 2)))
    
    hu_moments = []
    hu_moments.append(scm[2, 0] + scm[0, 2])
    hu_moments.append((scm[2, 0] - scm[0, 2])**2 + 4*(scm[1, 1]**2))
    hu_moments.append((scm[3, 0] - 3*scm[1, 2])**2 + (3*scm[2, 1] - scm[0, 3])**2)
    hu_moments.append((scm[3, 0] + scm[1, 2])**2 + (scm[2, 1] + scm[0, 3])**2)
    hu_moments.append((scm[3, 0] - 3*scm[1, 2])*(scm[3, 0] + scm[1, 2])*((scm[3, 0] + scm[1, 2])**2 - 3*((scm[2, 1] + scm[0, 3])**2)) + (3*scm[2, 1] - scm[0, 3])*(scm[2, 1] + scm[0, 3])*(3*(scm[3, 0] + scm[1, 2])**2 - (scm[2, 1] + scm[0, 3])**2))
    hu_moments.append((scm[2, 0] - scm[0, 2])*((scm[3, 0] + scm[1, 2])**2 - (scm[2, 1] + scm[0, 3])**2) + 4*scm[1, 1]*(scm[3, 0] + scm[1, 2])*(scm[2, 1] + scm[0, 3]))
    hu_moments.append((3*scm[2, 1] - scm[0, 3])*(scm[3, 0] + scm[1, 2])*((scm[3, 0] + scm[1, 2])**2 - 3*((scm[2, 1] + scm[0, 3])**2)) - (scm[3, 0] - 3*scm[1, 2])*(scm[2, 1] + scm[0, 3])*(3*(scm[3, 0] + scm[1, 2])**2 - (scm[2, 1] + scm[0, 3])**2))
    hu_moments.append(scm[1, 1]*((scm[3, 0] + scm[1, 2])**2 - (scm[0, 3] + scm[2, 1])**2) - (scm[2, 0] - scm[0, 2])*(scm[3, 0] + scm[1, 2])*(scm[0, 3] + scm[2, 1])) 

    
    return np.array(hu_moments)