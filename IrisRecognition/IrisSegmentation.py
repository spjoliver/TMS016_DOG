from scipy.signal import convolve2d, convolve
from matplotlib.patches import Circle
from typing import Optional
import numpy as np
import cv2
import matplotlib.pyplot as plt


def circle_mask(r, xmax: int, ymax: int, x0r: int, y0r: int, lateral: bool=False) -> np.ndarray:
    circle = np.zeros((r + 1, r + 1)).astype(bool)
    circle_x, circle_y = np.meshgrid(np.arange(0, r + 1), np.arange(0, r + 1))
    circle_y = np.flip(circle_y, axis=0)
    dist = np.hstack([np.abs(circle_x ** 2 + circle_y ** 2 - r ** 2), np.ones(r + 1)[:, np.newaxis]*1e10])
    x0 = 0
    y0 = 0
    circle[y0, x0] = True
    incomplete_circle = True
    while incomplete_circle:

        if dist[y0, x0 + 1] < dist[y0 + 1, x0 + 1]:
            if dist[y0, x0 + 1] < dist[y0 + 1, x0]:
                x0 += 1
            else:
                y0 += 1
        elif dist[y0 + 1, x0 + 1] < dist[y0 + 1, x0]:
            y0 += 1
            x0 += 1
        else:
            y0 += 1
        circle[y0, x0] = True
        if y0 == r:
            incomplete_circle = False

    circle = np.hstack([np.flip(circle[:, 1:], axis=1), circle])
    circle = np.vstack([circle, np.flip(circle[:-1, :], axis=0)])
    if lateral:
        circle[:, r//2 + 1:-r//2] = False
    rxup = r
    ryup = r
    rxdown = r
    rydown = r
    if y0r < r:
        ryup = y0r
        circle = circle[(r - y0r):, :]
    if x0r < r:
        rxup = x0r
        circle = circle[:, (r - x0r):]
    if y0r + r + 1 > ymax:
        diff = y0r + r + 1 - ymax
        rydown = ymax - y0r - 1
        circle = circle[:-diff, :]
    if x0r + r + 1 > xmax:
        diff = x0r + r + 1 - xmax
        rxdown = xmax - x0r - 1
        circle = circle[:, :-diff]
    
    return circle, ryup, rydown, rxup, rxdown

def G(x: float, sigma: float=1.0) -> float:
    return np.exp(-x**2 / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)

def LineIntegral(img: np.ndarray, r: int, x0: int, y0: int, lateral: bool=False) -> float:
    circle, ryup, rydown, rxup, rxdown = circle_mask(r, lateral=lateral, xmax=img.shape[1], ymax=img.shape[0], x0r=x0, y0r=y0)
    a = img[y0 - ryup:y0 + rydown + 1, x0 - rxup:x0 + rxdown + 1][circle]
    return (img[y0 - ryup:y0 + rydown + 1, x0 - rxup:x0 + rxdown + 1][circle]).sum() / circle.sum()

def drLineIntegral(img: np.ndarray, r: int, x0: int, y0: int, lateral: bool=False) -> float:
    return LineIntegral(img, r + 1, x0, y0, lateral=lateral) - LineIntegral(img, r, x0, y0, lateral=lateral)

def drLineIntegralMulti(img: np.ndarray, rmin: int, rmax: int, x0: int, y0: int, lateral: bool=False) -> np.ndarray:
    lint = np.zeros(rmax - rmin + 2)
    for r in range(rmin, rmax + 2):
        lint[r - rmin] = LineIntegral(img, r, x0, y0, lateral=lateral)
    return np.diff(lint)

def ConvolveGaussiandrLI(drLIM: np.ndarray, filter_size: int=3, sigma: float=1.0) -> np.ndarray:

    gf = np.exp(-(np.arange(filter_size) - filter_size // 2)**2/(2*sigma**2)) / (np.sqrt(2 * np.pi) * sigma)
    return np.convolve(drLIM, gf, mode="same")

def FindPupileCenter(img: np.ndarray, top_index: Optional[tuple]=None) -> tuple:
    ## First locate pupile
    ymax, xmax = img.shape
    ymax20 = int(ymax*0.05)
    xmax20 = int(xmax*0.05)
    pupile_frame = img[ymax20:-ymax20, xmax20:-xmax20]
    pupile_frame[pupile_frame > 0.7] = 0.2
    if top_index is None:
        struct_elem100 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (100, 100))
        struct_elem50 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
        pupil_open = cv2.morphologyEx(pupile_frame, cv2.MORPH_OPEN, struct_elem50, iterations=1)
        poc = convolve2d(pupil_open, struct_elem100, mode="same", fillvalue=1)
        top_index2 = np.unravel_index(np.argmin(poc), poc.shape)
    else:
        top_index2 = top_index
    ## probably not a super good guess, use gaussian filters and detect edges now
    filter_size = 8
    data_length = 275
    sigma = 0.5
    width = 50
    gf = np.exp(-(np.arange(filter_size) - filter_size // 2)**2/(2*sigma**2)) / (np.sqrt(2 * np.pi) * sigma)
    gf = np.tile(gf, (5, 1))
    gf2 = gf.T
    img = convolve2d(pupile_frame, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20)), mode="same", fillvalue=0)
    for i in range(2):
        right_datah = img[max(top_index2[0] - width//2, 0): top_index2[0] + width//2, top_index2[1]: top_index2[1] + data_length]
        left_datah = img[max(top_index2[0] - width//2, 0): top_index2[0] + width//2, max(top_index2[1] - data_length, 0): top_index2[1]]
        right_datav = img[top_index2[0]: top_index2[0] + data_length, max(top_index2[1] - width//2, 0): top_index2[1] + width//2]
        left_datav = img[max(top_index2[0] - data_length, 0): top_index2[0], max(top_index2[1] - width//2, 0): top_index2[1] + width//2]

        right_conv = np.abs(np.diff(convolve2d(right_datah, gf, mode="valid"), axis=1))
        plt.imshow(right_conv, cmap="gray")
        plt.show()
        dist_right = np.median(np.argmax(right_conv, axis=1)) + filter_size
        if top_index2[1] + data_length > img.shape[1] - 20:
            dist_right = np.median(np.argmax(right_conv[:, :-30], axis=1)) + filter_size
            print("dist right: ", dist_right)
        print("dist right: ", dist_right)
        left_conv = np.abs(np.diff(convolve2d(left_datah, gf, mode="valid"), axis=1))
        plt.imshow(left_conv, cmap="gray")
        plt.show()
        dist_left = left_datah.shape[1] - (np.median(np.argmax(left_conv, axis=1)) + filter_size)
        if max(top_index2[1] - data_length, 0) < 20:
            dist_left = left_datah.shape[1] - (np.median(np.argmax(left_conv[:, 30:], axis=1)) + filter_size + 30)
            print("dist left: ", dist_left)
        print("dist left: ", dist_left)
        top_conv = np.abs(np.diff(convolve2d(right_datav, gf2, mode="valid"), axis=0))
        plt.imshow(top_conv, cmap="gray")
        plt.show()
        dist_top = np.median(np.argmax(top_conv, axis=0)) + filter_size
        if top_index2[0] + data_length > img.shape[0] - 20:
            dist_top = np.median(np.argmax(top_conv[:-30, :], axis=0)) + filter_size
        print("dist top: ", dist_top)
        bottom_conv = np.abs(np.diff(convolve2d(left_datav, gf2, mode="valid"), axis=0))
        plt.imshow(bottom_conv, cmap="gray")
        plt.show()
        dist_bottom = left_datav.shape[0] - (np.median(np.argmax(bottom_conv, axis=0)) + filter_size)
        if max(top_index2[0] - data_length, 0) < 20:
            dist_bottom = left_datav.shape[0] - (np.median(np.argmax(bottom_conv[30:, :], axis=0)) + filter_size + 30)
        
        print("dist bottom: ", dist_bottom)

        if dist_right > dist_left:
            x0 = top_index2[1] + round((dist_right - dist_left) / 2)
        else:
            x0 = top_index2[1] - round((dist_left - dist_right) / 2)
        if dist_top > dist_bottom:
            y0 = top_index2[0] + round((dist_top - dist_bottom) / 2)
        else:
            y0 = top_index2[0] - round((dist_bottom - dist_top) / 2)
        top_index2 = [int(y0), int(x0)]
        
    return [int(y0) + ymax20, int(x0) + xmax20]

def EstimateRadius(top_index: list, img: np.ndarray, pupil: bool=True, pupil_radius: int=100):

    if pupil:
        filter_size = 6
        data_length = 200
        sigma = 0.5
        width = 50
        img[img > 0.7] = 0.2
        img = convolve2d(img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20)), mode="same", fillvalue=0)
        img = convolve2d(img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)), mode="same", fillvalue=0)
        gf = np.exp(-(np.arange(filter_size) - filter_size // 2)**2/(2*sigma**2)) / (np.sqrt(2 * np.pi) * sigma)
        gf = np.tile(gf, (5, 1))
        right_datah = img[max(top_index[0] - width//2, 0): top_index[0] + width//2, top_index[1]: top_index[1] + data_length]
        left_datah = img[max(top_index[0] - width//2, 0): top_index[0] + width//2, max(top_index[1] - data_length, 0): top_index[1]]

        right_conv = np.abs(np.diff(convolve2d(right_datah, gf, mode="valid"), axis=1))
        plt.imshow(right_conv,  cmap='gray')
        plt.show()
        right_conv = right_conv
        dist_right = np.median(np.argmax(right_conv, axis=1)) + filter_size
        print(dist_right, np.median(np.argmax(right_conv, axis=1)) + filter_size)
        left_conv = np.abs(np.diff(convolve2d(left_datah, gf, mode="valid"), axis=1))
        plt.imshow(left_conv,  cmap='gray')
        plt.show()
        dist_left = data_length - np.median(np.argmax(left_conv, axis=1)) + filter_size
        print(dist_left, np.median(np.argmax(left_conv, axis=1)) + filter_size)
        radius_estimate = (dist_right + dist_left) / 2
    else:
        filter_size = 10
        data_length = 270
        pupile_adder = int(pupil_radius*1.4)
        sigma = 0.5
        width = 60
        gf = np.exp(-(np.arange(filter_size) - filter_size // 2)**2/(2*sigma**2)) / (np.sqrt(2 * np.pi) * sigma)
        gf = np.tile(gf, (7, 1))
        right_datah = img[max(top_index[0] - width//2, 0): top_index[0] + width//2, top_index[1] + pupile_adder: top_index[1] + data_length + pupile_adder]
        left_datah = img[max(top_index[0] - width//2, 0): top_index[0] + width//2, max(top_index[1] - data_length - pupile_adder, 0): top_index[1] - pupile_adder]

        right_conv = np.abs(np.diff(convolve2d(right_datah, gf, mode="valid"), axis=1))
        plt.imshow(right_conv,  cmap='gray')
        plt.show()
        dist_right = np.median(np.argmax(right_conv, axis=1)) + filter_size + pupile_adder

        left_conv = np.abs(np.diff(convolve2d(left_datah, gf, mode="valid"), axis=1))
        dist_left = np.median(np.argmax(left_conv, axis=1)) + filter_size + pupile_adder
        plt.imshow(left_conv,  cmap='gray')
        plt.show()
        plt.imshow(left_datah,  cmap='gray')
        plt.show()
        left_conv = left_conv.sum(axis=0)
        
        print(dist_right, dist_left)
        radius_estimate = (dist_right + dist_left) / 2
    return radius_estimate

def FindEdgeLoss(
        img: np.ndarray, 
        rmin: int, 
        rmax: int, 
        filter_size: int=3, 
        sigma: float=1.0, 
        lateral: bool=False, 
        x0: Optional[int]=None,
        y0: Optional[int]=None,
        return_radius: bool=False,
        ):
    
    r_vec = np.arange(rmin, rmax + 1)

    r_vec_current = r_vec.copy()

    rmin = r_vec_current[0]
    rmax = r_vec_current[-1]
    drLIM = drLineIntegralMulti(img, rmin, rmax, x0, y0, lateral=lateral)
    cgdrLIM = ConvolveGaussiandrLI(drLIM, filter_size=filter_size, sigma=sigma)
    arg_max_blur = np.argmax(cgdrLIM)
    if return_radius:
        return r_vec_current[arg_max_blur]
    return cgdrLIM[arg_max_blur]
    

def FindEdge(
        img: np.ndarray, 
        rmin: int, 
        rmax: int, 
        search_radius: int=5, 
        filter_size: int=3, 
        sigma: float=1.0, 
        lateral: bool=False, 
        plot_img: Optional[np.ndarray]=None, 
        plot_: bool=True,
        x0: Optional[int]=None,
        y0: Optional[int]=None,
        ):
    
    
    top_index = (y0, x0)
    img_shape = img.shape
    n = search_radius*2 + 1
    max_blur = np.zeros((n, n))
    opt_r = np.zeros((n, n))
    index_ = np.zeros((n, n, 2))
    r_vec = np.arange(rmin, rmax + 1)

    for i, y0 in enumerate(range(top_index[0] - search_radius, top_index[0] + search_radius + 1)):
        for j, x0 in enumerate(range(top_index[1] - search_radius, top_index[1] + search_radius + 1)):
            index_[i, j] = (y0, x0)
            r_vec_current = r_vec.copy()

            rmin = r_vec_current[0]
            rmax = r_vec_current[-1]
            

            drLIM = drLineIntegralMulti(img, rmin, rmax, x0, y0, lateral=lateral)
            cgdrLIM = ConvolveGaussiandrLI(drLIM, filter_size=filter_size, sigma=sigma)
            arg_max_blur = np.argmax(cgdrLIM)
            max_blur[i, j] = cgdrLIM[arg_max_blur]
            opt_r[i, j] = r_vec_current[arg_max_blur]
    
    max_blur_idx = np.unravel_index(np.argmax(max_blur), max_blur.shape)
    opt_xy = index_[max_blur_idx]
    opt_r = opt_r[max_blur_idx]
    if plot_:
        fig, ax = plt.subplots(1)
        if plot_img is not None:
            img = plot_img
        ax.imshow(img, cmap="gray")
        ax.plot(opt_xy[1], opt_xy[0], "r+")
        ax.plot(top_index[1], top_index[0], "g+")
        circle = Circle(opt_xy[::-1], opt_r, color="r", fill=False)
        ax.add_patch(circle)
        plt.show()
    print("Optimal location and radius: ", f"\nx={opt_xy[1]}\ny={opt_xy[0]}\nr={opt_r}")
    opt_xy = (int(opt_xy[0]), int(opt_xy[1]))
    return opt_xy, int(opt_r)

def FindPupilIris(img: np.ndarray, filter_size: int=3, sigma: float=1.0, lateral: bool=False, plot_img: Optional[np.ndarray]=None) -> int:

    #struct_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    #img_erode = cv2.morphologyEx(img_use, cv2.MORPH_OPEN, struct_elem, iterations=1)


    top_index = FindPupileCenter(img)
    r_estimate_pupil = int(EstimateRadius(top_index, img, pupil=True))
    print("Estimated pupil radius: ", r_estimate_pupil)
    print("Estimated pupil center: ", top_index)

    pup_xy, pup_r = FindEdge(img, rmin=r_estimate_pupil - 10, rmax=r_estimate_pupil + 15, search_radius=20, filter_size=filter_size, sigma=sigma, lateral=True, plot_img=plot_img, plot_=False, x0=int(top_index[1]), y0=int(top_index[0]))
    #r_estimate_iris = int(EstimateRadius(pup_xy, img, pupil=False, pupil_radius=pup_r))
    r_estimate_iris = int(FindEdgeLoss(img, rmin=round(pup_r*1.5), rmax=round(pup_r*2.5), sigma=sigma, lateral=True, x0=int(pup_xy[1]), y0=int(pup_xy[0]), return_radius=True))
    print("Estimated iris radius: ", r_estimate_iris)
    iris_xy, iris_r = FindEdge(img, rmin=r_estimate_iris - 30, rmax=r_estimate_iris + 25, search_radius=10, filter_size=filter_size, sigma=sigma, lateral=True, plot_img=plot_img, plot_=False, x0=int(pup_xy[1]), y0=int(pup_xy[0]))

    fig, ax = plt.subplots(1)
    if plot_img is not None:
        img = plot_img
    ax.imshow(img, cmap="gray", interpolation="none")
    ax.plot(pup_xy[1], pup_xy[0], "r+")
    ax.plot(iris_xy[1], iris_xy[0], "g+")
    ax.plot(top_index[1], top_index[0], "b+")
    circle_pup = Circle(pup_xy[::-1], pup_r, color="r", fill=False)
    circle_iris = Circle(iris_xy[::-1], iris_r, color="g", fill=False)
    ax.add_patch(circle_pup)
    ax.add_patch(circle_iris)
    ax.legend(["Pupil", "Iris", "First pupil center estimate"])
    plt.show()

    LocateEyelids(iris_r, pup_r, iris_xy[1], iris_xy[0], img)


def LocateEyelids(
        r: int,
        pupil_radius: int,
        x0: int,
        y0: int,
        img: np.ndarray
        ):
    eye = img[y0 - r:y0 + r + 1, x0 - r:x0 + r + 1]
    min_r = 4
    circle, ryup, rydown, rxup, rxdown = circle_mask(r - min_r, lateral=False, xmax=img.shape[1], ymax=img.shape[0], x0r=x0, y0r=y0)

    circ_res = np.zeros((circle.shape[0], circle.shape[1]))

    filter_size = 15
    sigma = 0.5
    gf = np.exp(-(np.arange(filter_size) - filter_size // 2)**2/(2*sigma**2)) / (np.sqrt(2 * np.pi) * sigma)

        
    circle_ru = circle[:r, r:]
    circle_ru_idx = np.where(circle_ru)
    circle_ru_idx = np.vstack(circle_ru_idx).T
    circle_ru_idx[:, 1] += r
    circle_ru_data = eye[circle_ru_idx[:, 0] + min_r, circle_ru_idx[:, 1] - min_r]

    circle_ru_conv = np.abs(np.diff(convolve(circle_ru_data, gf, mode="valid")))
    circle_ru_conv_idx = np.argmax(circle_ru_conv) + filter_size + 1
    #circ_res[:r, :r][circle_ru][filter_size + 1:-filter_size] = circle_ru_conv
    circle_ru_conv_idx_yx = circle_ru_idx[circle_ru_conv_idx]
    plt.plot(np.arange(circle_ru_conv.shape[0]), circle_ru_conv)
    plt.show()
    print(circle_ru_conv_idx)

    
    
    circle_rl = circle[r:, r:]
    circle_rl_idx = np.where(circle_rl)
    circle_rl_idx = np.vstack(circle_rl_idx).T
    circle_rl_idx[:, 0] += r
    circle_rl_idx[:, 1] += r

    circle_rl_data = eye[circle_rl_idx[:, 0] - min_r, circle_rl_idx[:, 1] - min_r]

    circle_rl_conv = np.abs(np.diff(convolve(circle_rl_data, gf, mode="valid")))
    circle_rl_conv_idx = np.argmax(circle_rl_conv) + filter_size + 1
    circle_rl_conv_idx_yx = circle_rl_idx[circle_rl_conv_idx]

    circle_ll = circle[r:, :r + 1]
    circle_ll_idx = np.where(circle_ll)
    circle_ll_idx = np.vstack(circle_ll_idx).T
    circle_ll_idx[:, 0] += r

    circle_ll_data = eye[circle_ll_idx[:, 0] - min_r, circle_ll_idx[:, 1] + min_r]

    circle_ll_conv = np.abs(np.diff(convolve(circle_ll_data, gf, mode="valid")))
    circle_ll_conv_idx = np.argmax(circle_ll_conv) + filter_size + 1
    circle_ll_conv_idx_yx = circle_ll_idx[circle_ll_conv_idx] 

    circle_lu = circle[:r, :r + 1]
    circle_lu_idx = np.where(circle_lu)
    circle_lu_idx = (np.flip(circle_lu_idx[0], axis=0), np.flip(circle_lu_idx[1], axis=0))
    circle_lu_idx = np.vstack(circle_lu_idx).T

    circle_lu_data = eye[circle_lu_idx[:, 0] + min_r, circle_lu_idx[:, 1] + min_r]

    circle_lu_conv = np.abs(np.diff(convolve(circle_lu_data, gf, mode="valid")))
    circle_lu_conv_idx = np.argmax(circle_lu_conv) + filter_size + 1
    circle_lu_conv_idx_yx = circle_lu_idx[circle_lu_conv_idx]

    right_side_idx = np.vstack([circle_ru_idx, circle_rl_idx ])
    left_side_idx = np.vstack([circle_lu_idx, circle_ll_idx])
    full_circle_idx = np.vstack([right_side_idx, left_side_idx])
    print(full_circle_idx[:, 0].max(), full_circle_idx[:, 0].min(), full_circle_idx[:, 1].max(), full_circle_idx[:, 1].min())

    plt.imshow(eye, cmap="gray", interpolation="none")
    plt.plot(circle_ru_conv_idx_yx[0], circle_ru_conv_idx_yx[1], "r+")
    plt.plot(circle_rl_conv_idx_yx[0], circle_rl_conv_idx_yx[1], "r+")
    plt.plot(circle_ll_conv_idx_yx[0], circle_ll_conv_idx_yx[1], "r+")
    plt.plot(circle_lu_conv_idx_yx[0], circle_lu_conv_idx_yx[1], "r+")
    plt.show()
    
    print(eye.shape)


    rs_c = right_side_idx.shape[0]
    lr_el = right_side_idx[rs_c//2 + 85:, :]
    ur_el = right_side_idx[:rs_c//2 + 85, :]

    ls_c = left_side_idx.shape[0]
    ll_el = left_side_idx[:ls_c//2 - 85, :]
    ul_el = left_side_idx[ls_c//2 - 85:, :]

    filter_size = 20
    sigma = 1
    gf = np.exp(-(np.arange(filter_size) - filter_size // 2)**2/(2*sigma**2)) / (np.sqrt(2 * np.pi) * sigma)

    # right side
    lr_el_conv = convolve(eye[lr_el[:, 0], lr_el[:, 1]], gf, mode="valid")
    lr_el_conv_max = np.argmax(lr_el_conv) + filter_size
    lr_el_conv_max_idx = lr_el[lr_el_conv_max, :]
    ur_el_conv = convolve(eye[lr_el[:, 0], lr_el[:, 1]], gf, mode="valid")
    ur_el_conv_max = np.argmax(ur_el_conv) + filter_size
    ur_el_conv_max_idx = ur_el[ur_el_conv_max, :]
    # left side
    ll_el_conv = convolve(eye[ll_el[:, 0], ll_el[:, 1]], gf, mode="valid")
    ll_el_conv_max = np.argmax(ll_el_conv) + filter_size
    ll_el_conv_max_idx = ll_el[ll_el_conv_max, :]
    ul_el_conv = convolve(eye[ul_el[:, 0], ul_el[:, 1]], gf, mode="valid")
    ul_el_conv_max = np.argmax(ul_el_conv) + filter_size
    ul_el_conv_max_idx = ul_el[ul_el_conv_max, :]

    # try something else
    filter_size = 8
    sigma = 1
    gf = np.exp(-(np.arange(filter_size) - filter_size // 2)**2/(2*sigma**2)) / (np.sqrt(2 * np.pi) * sigma)

    eye_mid = eye[:r - pupil_radius, :]
    eye_mid_down = eye[r + pupil_radius:, :]
    eye_mid_true = eye_mid.copy()
    filter_size3 = 20
    gf2 = np.tile(gf, (filter_size3, 1)).T
    gf3 = np.tile(gf, (6, 1)).T
    gf3[3:, :] = -gf3[3:, :]
    eye_mid[eye_mid > 0.7] = 0.2
    eye_mid_down[eye_mid_down > 0.7] = 0.2

    plt.imshow(eye_mid_down, cmap="gray")
    plt.show()

    eye_mid = cv2.erode(eye_mid, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)), iterations=2)
    eye_mid_down = cv2.erode(eye_mid_down, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6)), iterations=1)
    eye_mid_conv = np.abs(np.diff(convolve2d(eye_mid, gf2, mode="valid"), axis=0))
    eye_mid_down_conv = np.abs(np.diff(convolve2d(eye_mid_down, gf2, mode="valid"), axis=0))


    print(eye_mid_conv.shape)
    filter_size2 = 5
    eye_mid_conv = convolve2d(cv2.dilate(eye_mid_conv, np.ones((filter_size2, filter_size2)), iterations=2), np.ones((filter_size2,filter_size2)), mode="same", fillvalue=0)
    eye_mid_down_conv = convolve2d(cv2.dilate(eye_mid_down_conv, np.ones((filter_size2, filter_size2)), iterations=1), np.ones((filter_size2,filter_size2)), mode="same", fillvalue=0)
    plt.imshow(eye_mid_down_conv, cmap="gray")
    plt.show()
    if True:
        plt.imshow(eye, cmap="gray")
        for i in range(eye_mid_conv.shape[1]):
            y = np.argmax(eye_mid_conv[:, i]) + filter_size + 5
            y2 = np.argmax(eye_mid_down_conv[:, i]) + filter_size + 5 + r + 90
            x = i + filter_size3
            plt.plot(x, y, "ro")
            plt.plot(x, y2, "ro")
        plt.show()

    eye_lid_top_mid = np.median(np.argmax(eye_mid_conv[:, max(r - filter_size3 - 150, 0):r + 150], axis=0) + filter_size + filter_size2)
    eye_lid_top_right = np.median(np.argmax(eye_mid_conv[:, max(3*r//2 - filter_size3 - 80, 0):3*r//2 + 80], axis=0) + filter_size + filter_size2)
    eye_lid_top_left = np.median(np.argmax(eye_mid_conv[:, max(r//2 - filter_size3 - 80, 0):r//2 + 80], axis=0) + filter_size + filter_size2)

    eye_lid_bottom_mid = np.median(np.argmax(eye_mid_down_conv[:, max(r - filter_size3 - 80, 0):r + 80], axis=0) + filter_size + filter_size2 + r + 90)
    eye_lid_bottom_right = np.median(np.argmax(eye_mid_down_conv[:, max(3*r//2 - filter_size3 - 80, 0):3*r//2 + 80], axis=0) + filter_size + filter_size2 + r + 90)
    eye_lid_bottom_left = np.median(np.argmax(eye_mid_down_conv[:, max(r//2 - filter_size3 - 80, 0):r//2 + 80], axis=0) + filter_size + filter_size2 + r + 90)

    #eye_mid_conv_max = np.argmax(eye_mid_conv.sum(axis=1)) + filter_size
    eye_mid_top_mid = np.array([eye_lid_top_mid, r])
    eye_mid_top_right = np.array([eye_lid_top_right, 3*r//2])
    eye_mid_top_left = np.array([eye_lid_top_left, r//2])
    top_points = np.vstack([eye_mid_top_mid, eye_mid_top_right, eye_mid_top_left])

    eye_mid_bottom_mid = np.array([eye_lid_bottom_mid, r])
    eye_mid_bottom_right = np.array([eye_lid_bottom_right, 3*r//2])
    eye_mid_bottom_left = np.array([eye_lid_bottom_left, r//2])
    bottom_points = np.vstack([eye_mid_bottom_mid, eye_mid_bottom_right, eye_mid_bottom_left])

    if True:#2*r - eye_mid_bottom_mid[0] > -1:
        # fit a second order polynomial
        top_poly = np.polyfit(bottom_points[:, 1], bottom_points[:, 0], 2)
        points_i = np.arange(r, 2*r)
        y_lidb1 = np.polyval(top_poly, points_i)
        y_iris_b = np.argmax(circle[r:, r+1:].astype(int), axis=0) + r
        #plt.imshow(y, cmap="gray")
        diff = np.abs(y_lidb1 - y_iris_b)
        intersection_br = np.argmin(diff)
        y_br = round(y_iris_b[intersection_br])
        y_lidb2 = np.polyval(top_poly, np.arange(0, r))
        diff = np.abs(y_lidb2 - np.flip(y_iris_b, axis=0))
        intersection_bl = np.argmin(diff)
        y_bl = round(np.flip(y_iris_b, axis=0)[intersection_bl])
    
    if True:#eye_mid_top_mid[0] > 18:
        # fit a second order polynomial
        top_poly = np.polyfit(top_points[:, 1], top_points[:, 0], 2)
        points_i = np.arange(r, 2*r)
        y_lid1 = np.polyval(top_poly, points_i)
        y_iris = np.argmax(circle[:r+1, r+1:].astype(int), axis=0)   
        #plt.imshow(y, cmap="gray")
        diff = np.abs(y_lid1 - y_iris)
        intersection_topr = np.argmin(diff)
        y_topr = round(y_iris[intersection_topr])
        y_lid2 = np.polyval(top_poly, np.arange(0, r))
        diff = np.abs(y_lid2 - np.flip(y_iris, axis=0))
        intersection_topl = np.argmin(diff)
        y_topl = round(np.flip(y_iris, axis=0)[intersection_topl])



    
    if True:
        plt.imshow(eye, cmap="gray")
        plt.plot(eye_mid_top_mid[1], eye_mid_top_mid[0], "ro")
        plt.plot(eye_mid_top_right[1], eye_mid_top_right[0], "ro")
        plt.plot(eye_mid_top_left[1], eye_mid_top_left[0], "ro")
        plt.plot(eye_mid_bottom_mid[1], eye_mid_bottom_mid[0], "ro")
        plt.plot(eye_mid_bottom_right[1], eye_mid_bottom_right[0], "ro")
        plt.plot(eye_mid_bottom_left[1], eye_mid_bottom_left[0], "ro")
        plt.plot(intersection_topr + r, y_topr, "go")
        plt.plot(intersection_topl, y_topl, "go")
        plt.plot(intersection_br + r, y_br, "go")
        plt.plot(intersection_bl, y_bl, "go")
        plt.plot(points_i, y_iris, "g")
        plt.plot(np.arange(0, r), np.flip(y_iris, axis=0), "g")
        plt.plot(np.arange(r, 2*r), y_lid1, "r")
        plt.plot(np.arange(0, r), y_lid2, "r")
        plt.plot(np.arange(r, 2*r), y_lidb1, "r")
        plt.plot(np.arange(0, r), y_lidb2, "r")
        plt.plot(points_i, y_iris_b, "g")
        plt.plot(np.arange(0, r), np.flip(y_iris_b, axis=0), "g")
        #plt.legend(["Right lower eyelid", "Right upper eyelid", "Left lower eyelid", "Left upper eyelid"])
        plt.show()


if False:
    img_use = cv2.imread("IrisRecognition/UTIRIS_infrared/074/074_L/Img_074_L_2_1.bmp").astype(np.double)[: , :, 0] / 255

    FindPupilIris(img_use, filter_size=3, sigma=1.0, lateral=False, plot_img=None)
    #Optimal location and radius:  
    #x=592.0
    #y=333.0
    #r=267.0
    #LocateEyelids(
    #    r=267,
    #    pupil_radius=30,
    #    x0=592,
    #    y0=333,
    #    img=img_use
    #)




    


if False:
    import os
    print(os.getcwd())
    img_use = cv2.imread("IrisRecognition/UTIRIS_infrared/021/021_R/Img_021_R_5.bmp").astype(np.double)[: , :, 0] / 255
    plt.imshow(img_use, cmap="gray")
    plt.show()
    FindPupilIris(img_use, filter_size=3, sigma=1.0, lateral=False, plot_img=img_use)
