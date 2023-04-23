import numpy as np
from typing import Union


def generate_moments(img: np.ndarray, moment_choice: Union[list, None]=None, p_to: int=5, q_to: int=5) -> np.ndarray:

    row_index_matrix = np.repeat(np.arange(img.shape[0])[:, np.newaxis], axis=1, repeats=img.shape[1])
    col_index_matrix = np.repeat(np.arange(img.shape[1])[np.newaxis, :].T, axis=1, repeats=img.shape[0]).T

    #col_index_matrix, row_index_matrix = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]))

    moments = []
    if moment_choice is not None:
        for p, q in moment_choice:
            moments.append(((row_index_matrix**p)*(col_index_matrix**q)*img).sum())
    else:
        for p in range(p_to):
            for q in range(q_to):
                moments.append(((row_index_matrix**p)*(col_index_matrix**q)*img).sum())
    
    return np.array(moments)

def generate_central_moments(img: np.ndarray, moment_choice: Union[list, None]=None, p_to: int=5, q_to: int=5) -> np.ndarray:
    """
    Translation invariant moments
    """

    centroid_moments = generate_moments(
        img, 
        moment_choice=[(0, 0), (1, 0), (0, 1)]
        )
    x_bar = centroid_moments[1] / centroid_moments[0]
    y_bar = centroid_moments[2] / centroid_moments[0]

    row_index_matrix = np.repeat(np.arange(img.shape[0])[:, np.newaxis], axis=1, repeats=img.shape[1]) - x_bar
    col_index_matrix = np.repeat(np.arange(img.shape[1])[np.newaxis, :].T, axis=1, repeats=img.shape[0]).T - y_bar

    #col_index_matrix, row_index_matrix = np.meshgrid(np.arange(img.shape[0]) - x_bar, np.arange(img.shape[1]) - y_bar)

    moments = []
    if moment_choice is not None:
        for p, q in moment_choice:
            moments.append(((row_index_matrix**p)*(col_index_matrix**q)*img).sum())
    else:
        for p in range(p_to):
            for q in range(q_to):
                moments.append(((row_index_matrix**p)*(col_index_matrix**q)*img).sum())
    
    return np.array(moments)


def generate_scaled_central_moments(img: np.ndarray, moment_choice: Union[list, None]=None, p_to: int=5, q_to: int=5) -> np.ndarray:
    """
    Translation and scale invariant moments
    """

    central_moments = generate_central_moments(
        img, 
        moment_choice=moment_choice,
        p_to=p_to,
        q_to=q_to
        ).reshape(p_to, q_to)
    scm = np.zeros((p_to, q_to))
    for p in range(p_to):
        q_start = 0 if p >= 2 else 2 - p
        for q in range(q_start, q_to):
            scm[p, q] = central_moments[p, q] / (central_moments[0, 0]**(1 + ((p + q) / 2)))
    
    return scm.reshape(-1)


def generate_hu_moments(img: np.ndarray) -> np.ndarray:

    """
    Hu moments are translation, scale and rotation invariant
    """

    central_moments = generate_central_moments(
        img, 
        p_to=4,
        q_to=4
        ).reshape(4, 4)
    scm = np.zeros((4, 4))
    for p in range(4):
        for q in range(4):
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