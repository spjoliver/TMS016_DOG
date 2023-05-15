from scipy.signal import convolve2d, convolve
from matplotlib.patches import Circle
from typing import Optional
import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage as skim


def FastIrisPupilScanner(
        img: str,
    ):
    
    rvec = np.arange(46, 160, 1)
    sigma = 0.5
    filter_size = 5

    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)[75:-75, 75:-75]
    img_shape = img.shape
    img_use = cv2.GaussianBlur(img, (filter_size, filter_size), 0)
    
    #edges = cv2.Canny(img_use, 20, 40), standard
    edges = cv2.Canny(img_use, 45, 55)
    plt.imshow(edges, cmap="gray")
    plt.show()
    
    hough_results = skim.transform.hough_circle(edges, rvec)

    ridx, r, c = np.unravel_index(np.argmax(hough_results), hough_results.shape)
    prmax = rvec[ridx]
    if edges.shape[1] - (c + prmax) < 45 and edges.shape[0] - (r + prmax) < 45:
        hough_results = skim.transform.hough_circle(edges[:-45, :-45], rvec)

        ridx, r, c = np.unravel_index(np.argmax(hough_results), hough_results.shape)
        prmax = rvec[ridx]
    elif edges.shape[1] - (c + prmax) < 45:
        hough_results = skim.transform.hough_circle(edges[:, :-45], rvec)

        ridx, r, c = np.unravel_index(np.argmax(hough_results), hough_results.shape)
        prmax = rvec[ridx]
    elif edges.shape[0] - (r + prmax) < 45:
        hough_results = skim.transform.hough_circle(edges[:-45, :], rvec)

        ridx, r, c = np.unravel_index(np.argmax(hough_results), hough_results.shape)
        prmax = rvec[ridx]
    elif (c - prmax) < 45:
        hough_results = skim.transform.hough_circle(edges[:, 45:], rvec)

        ridx, r, c = np.unravel_index(np.argmax(hough_results), hough_results.shape)
        prmax = rvec[ridx]
        c += 35
    elif (r - prmax) < 45:
        hough_results = skim.transform.hough_circle(edges[45:, :], rvec)

        ridx, r, c = np.unravel_index(np.argmax(hough_results), hough_results.shape)
        prmax = rvec[ridx]
        r += 45


    #r_estimate_iris = EstimateRadius([c, r], img=img_use, pupil=False, pupil_radius=prmax)
    #print("Estimated iris radius: ", r_estimate_iris)
    r_estimate_iris = int(FindEdgeLoss(img, rmin=round(prmax*1.5), rmax=max(round(prmax*3), 290), sigma=sigma, lateral=True, x0=int(c), y0=int(r), return_radius=True))
    iris_xy, iris_r = FindEdge(img, rmin=int(r_estimate_iris) - 25, rmax=int(r_estimate_iris), search_radius=5, filter_size=filter_size, sigma=sigma, lateral=True, plot_=False, x0=int(c), y0=int(r))

    fig, ax = plt.subplots()
    ax.imshow(img, cmap="gray")
    print("Estimated pupil radius: ", prmax)
    ax.add_patch(plt.Circle((c, r), prmax, color="r", fill=False))
    ax.add_patch(plt.Circle((iris_xy[1], iris_xy[0]), iris_r, color="g", fill=False))
    plt.show()

    if False:
        rho = 1
        theta = np.pi / 180
        threshold = 200
        min_line_length = edges.shape[0] - edges.shape[0] // 2
        max_line_gap = 20
        #lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
        lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
        for line in lines:
            for x1, y1, x2, y2 in line:
                plt.plot((y1, x1), (y2, x2), color="r")
        

    if False:
        result = skim.transform.hough_ellipse(edges, accuracy=20, threshold=250,
                        min_size=100, max_size=120)
        result.sort(order='accumulator')
        best = list(result[-1])
        yc, xc, a, b = (int(round(x)) for x in best[1:5])
        orientation = best[5]

        # Draw the ellipse on the original image
        cy, cx = skim.draw.ellipse_perimeter(yc, xc, a, b, orientation)
        # Draw the edge (white) and the resulting ellipse (red)
        edges = skim.color.gray2rgb(skim.img_as_ubyte(edges))
        edges[cy, cx] = 1
        plt.imshow(edges)
        plt.show()

    # Search for lines, something similar can be done
    edges = cv2.Canny(img_use, 15, 20)
    edges[max(r-prmax-60, 0):r+prmax+60, max(c-prmax-60, 0):c+prmax+60] = 0
    edges[max(r-prmax-70, 0):r+prmax+70, max(c-prmax-70, 0):c+prmax+70] = cv2.GaussianBlur(edges[max(r-prmax-70, 0):r+prmax+70, max(c-prmax-70, 0):c+prmax+70], (5, 5), 0)
    edges = convolve2d(edges, np.ones((5, 5)), mode="same", fillvalue=0)
    print(edges.shape)
    print(edges.max())
    print(edges.min())
    plt.imshow(edges, cmap="gray")
    plt.show()
    

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
    #a = img[y0 - ryup:y0 + rydown + 1, x0 - rxup:x0 + rxdown + 1][circle]
    return (img[y0 - ryup:y0 + rydown + 1, x0 - rxup:x0 + rxdown + 1][circle]).sum() / circle.sum()

def drLineIntegral(img: np.ndarray, r: int, x0: int, y0: int, lateral: bool=False) -> float:
    return LineIntegral(img, r + 1, x0, y0, lateral=lateral) - LineIntegral(img, r, x0, y0, lateral=lateral)

def drLineIntegralMulti(img: np.ndarray, rmin: int, rmax: int, x0: int, y0: int, lateral: bool=False, jump: int=1) -> np.ndarray:
    #lint = np.zeros(rmax - rmin + 2)
    r_range = np.arange(rmin, rmax + 2, jump)
    lint = np.zeros(r_range.shape)
    for i, r in enumerate(r_range):#:range(rmin, rmax + 2):
        lint[i] = LineIntegral(img, int(r), x0, y0, lateral=lateral)
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
    width = 80
    gf = np.exp(-(np.arange(filter_size) - filter_size // 2)**2/(2*sigma**2)) / (np.sqrt(2 * np.pi) * sigma)
    gf = np.tile(gf, (5, 1))
    gf2 = gf.T
    img = convolve2d(pupile_frame, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20)), mode="same", fillvalue=0)
    img = convolve2d(pupile_frame, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)), mode="same", fillvalue=0)
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
        ## This takes into account that the the vertical and horizontal radius should be the same
        # thus one can catch potential problems with the pupil detection and correct them, hopefully.
        if i == 1:
            if dist_top + dist_bottom < dist_right + dist_left + 50:
                
                if dist_top < dist_bottom:
                    dist_top = dist_top + round((dist_right + dist_left) / 2 - dist_top)
                else:
                    dist_bottom = dist_bottom + round((dist_right + dist_left) / 2 - dist_bottom)
            elif dist_right + dist_left < dist_top + dist_bottom + 50:

                if dist_right < dist_left:
                    dist_right = dist_right + round((dist_top + dist_bottom) / 2 - dist_right)
                else:
                    dist_left = dist_left + round((dist_top + dist_bottom) / 2 - dist_left)
        
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
        data_length = 250
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
        filter_size = 3
        data_length = 270
        pupile_adder = int(pupil_radius*1.4)
        sigma = 0.5
        width = 40
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
    jump = 4
    r_vec = np.arange(rmin, rmax + 1, jump)

    r_vec_current = r_vec.copy()

    rmin = r_vec_current[0]
    rmax = r_vec_current[-1]
    drLIM = drLineIntegralMulti(img, rmin, rmax, x0, y0, lateral=lateral, jump=jump)
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
    r_estimate_iris = int(FindEdgeLoss(img, rmin=round(pup_r*1.5), rmax=round(pup_r*3), sigma=sigma, lateral=True, x0=int(pup_xy[1]), y0=int(pup_xy[0]), return_radius=True))
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
    #IsolateIris(iris_r, pup_r, iris_xy[1], iris_xy[0], img)
    #LocateEyelids2(iris_r, pup_r, iris_xy[1], iris_xy[0], img)


def IsolateIris(
        r: int,
        pupil_radius: int,
        x0: int,
        y0: int,
        img: np.ndarray,
    ):
    eye = img[max(y0 - r, 0):y0 + r + 1, x0 - r:x0 + r + 1]
    eye_mask = np.zeros_like(eye)

    for i in range(pupil_radius, r):
        circle, ryup, rydown, rxup, rxdown = circle_mask(i, lateral=False, xmax=img.shape[1], ymax=img.shape[0], x0r=x0, y0r=y0)
        circle = np.pad(circle, r-i, mode="constant", constant_values=0)
        eye_mask[circle] = 1

    eye[eye_mask == 0] = 1
    plt.imshow(eye, cmap="gray")
    plt.show()


def EyelidFitter(
    r: int,
    pupil_radius: int,
    x0: int,
    y0: int,
    img: np.ndarray
    ):
    eye = img[max(y0 - r, 0):y0 + r + 1, x0 - r:x0 + r + 1]
    rmin = r

    circle, ryup, rydown, rxup, rxdown = circle_mask(r, lateral=False, xmax=img.shape[1], ymax=img.shape[0], x0r=x0, y0r=y0)
    print(circle.shape)
    print(eye.shape)
    circle_ru = circle[:rmin, rmin:]
    circle_ru_idx = np.where(circle_ru)
    circle_ru_idx = np.vstack(circle_ru_idx).T
    circle_ru_idx[:, 1] += r
    plt.imshow(eye, cmap="gray")
    for i in range(circle_ru_idx.shape[0]):
        plt.plot(circle_ru_idx[i, 1], circle_ru_idx[i, 0], "r+")
    
    circle_rl = circle[rmin:, rmin:]
    circle_rl_idx = np.where(circle_rl)
    circle_rl_idx = np.vstack(circle_rl_idx).T
    circle_rl_idx[:, 0] += r
    circle_rl_idx[:, 1] += r

    circle_ll = circle[rmin:, :rmin + 1]
    circle_ll_idx = np.where(circle_ll)
    circle_ll_idx = np.vstack(circle_ll_idx).T
    circle_ll_idx[:, 0] += r

    circle_lu = circle[:rmin, :rmin + 1]
    circle_lu_idx = np.where(circle_lu)
    circle_lu_idx = (np.flip(circle_lu_idx[0], axis=0), np.flip(circle_lu_idx[1], axis=0))
    circle_lu_idx = np.vstack(circle_lu_idx).T

    circle_lu_idx_use = circle_lu_idx.copy()[30:, :30]
    print(circle_lu_idx_use.shape)
    circle_ru_idx_use = circle_ru_idx.copy()[30:, :30]
    print(circle_ru_idx_use.shape)
    com_idx = np.arange(25, circle_lu_idx_use.shape[0], 3)

    max_blur = np.zeros((com_idx.shape[0], com_idx.shape[0]))
    leftidx = np.zeros((com_idx.shape[0], com_idx.shape[0]))
    rightidx = np.zeros((com_idx.shape[0], com_idx.shape[0]))
    optymid = np.zeros((com_idx.shape[0], com_idx.shape[0]))

    ymid = np.arange(5, r - pupil_radius - max(r - y0, 0), 3)
    xmid = r
    filter_size = 5
    for ii, i in enumerate(com_idx):
        for jj, j in enumerate(com_idx):
            leftidx[ii, jj] = i
            rightidx[ii, jj] = j
            x0 = circle_lu_idx_use[i, 1]
            y0 = circle_lu_idx_use[i, 0]
            x1 = circle_ru_idx_use[j, 1]
            y1 = circle_ru_idx_use[j, 0]
            xeval = np.arange(x0, x1, 1)
            res1 = drLineIntegralMulti2(eye, x0, y0, x1, y1, xmid, ymid, xeval, maxy=eye.shape[0] - 1, miny=0)
            res2 = ConvolveGaussiandrLI2(res1, filter_size=filter_size, sigma=0.5)
            arg_max_blur = np.argmax(res2) + filter_size // 2
            max_blur[ii, jj] = res2[arg_max_blur]
            optymid[ii, jj] = ymid[arg_max_blur]
            max_blur[ii, jj] = np.max(res2)
    

    max_blur_idx = np.unravel_index(np.argmax(max_blur), max_blur.shape)
    opt_i = leftidx[max_blur_idx]
    opt_j = rightidx[max_blur_idx]
    optymid = optymid[max_blur_idx]
    left_point = circle_lu_idx_use[int(opt_i), :]
    right_point = circle_ru_idx_use[int(opt_j), :]

    x = np.array([left_point[1], xmid, right_point[1]])
    y = np.array([left_point[0], optymid, right_point[0]])

    p = np.polyfit(x, y, 2)
    x_eval = np.arange(x[0], x[-1], 1)
    y_eval = np.polyval(p, x_eval)

    plt.imshow(eye, cmap="gray")
    #for i in range(circle_ru_idx.shape[0]-40):
    #    plt.plot(circle_ru_idx[i, 1], circle_ru_idx[i, 0], "r+")
    #    plt.plot(circle_lu_idx[i, 1], circle_lu_idx[i, 0], "g+")
    plt.plot(x_eval, y_eval, "r-")
    plt.show()

def EyeLidMask(x: np.ndarray, y: np.ndarray, x_eval: np.ndarray, maxy: int, miny: int):
    
    p = np.polyfit(x, y, 2)
    y = np.polyval(p, x_eval)
    return np.round(y[(y <= maxy) & (y >= miny)]), x_eval[(y <= maxy) & (y >= miny)]


def LineIntegral2(img: np.ndarray, x: np.ndarray, y: np.ndarray, x_eval: np.ndarray, maxy: int, miny: int) -> float:
    y, x_eval = EyeLidMask(x, y, x_eval, maxy, miny)
    return (img[y.astype(int), x_eval.astype(int)]).sum() / y.shape[0]


def drLineIntegralMulti2(img: np.ndarray, x_left: int, y_left: int, x_right: int, y_right: int, xmid: int, ymid: np.ndarray, x_eval: np.ndarray, maxy: int, miny: int) -> np.ndarray:
    lint = np.zeros_like(ymid)
    x = np.array([x_left, xmid, x_right])
    y = np.array([y_left, 0, y_right])
    for i, ymidi in enumerate(ymid):
        y[1] = ymidi
        lint[i] = LineIntegral2(img, x, y, x_eval, maxy, miny)
    return np.diff(lint)

def ConvolveGaussiandrLI2(drLIM: np.ndarray, filter_size: int=3, sigma: float=1.0) -> np.ndarray:

    gf = np.exp(-(np.arange(filter_size) - filter_size // 2)**2/(2*sigma**2)) / (np.sqrt(2 * np.pi) * sigma)
    return np.convolve(drLIM, gf, mode="valid")



def LocateEyelids2(
        r: int,
        pupil_radius: int,
        x0: int,
        y0: int,
        img: np.ndarray
        ):
    print(img.shape)
    print(y0 + r)
    plt.imshow(img, cmap="gray")
    plt.plot(x0, y0, "r+")
    plt.show()
    if  r > y0:
        adder = r - y0
    else:
        adder = 0
    eye = img[y0 - r + adder:y0 + r + 1, max(x0 - r, 0):x0 + r + 1]
    plt.imshow(eye, cmap="gray")
    plt.plot(2*x0 - r, 2*y0 - r + adder, "r+")
    plt.show()
    filter_size = 40
    width = 5
    sigma = 0.5
    gf = np.exp(-(np.arange(filter_size) - filter_size // 2)**2/(2*sigma**2)) / (np.sqrt(2 * np.pi) * sigma)
    gf2 = np.tile(gf, (width, 1)).T

    eye_above_pupil = eye[:r - pupil_radius - 10 - adder, :]
    eyeabp = eye_above_pupil.copy()
    conv_diff = np.abs(np.diff(convolve2d(eyeabp, gf2, mode="valid"), axis=0))
    conv_diff_argmax = np.argmax(conv_diff, axis=0) + filter_size

    x = np.arange(conv_diff_argmax.shape[0]) + width

    p = np.polyfit(x, conv_diff_argmax, 2)
    y = np.polyval(p, x)

    plt.imshow(eye_above_pupil, cmap="gray")
    plt.plot(x, conv_diff_argmax, "r+")
    plt.plot(x, y, "g+")
    plt.show()
    plt.imshow(eyeabp, cmap="gray")
    plt.show()
    plt.imshow(conv_diff, cmap="gray")
    plt.show()

    eye_below_pupil = eye[r + pupil_radius + 15 + adder:, :]
    eyeabp = eye_below_pupil.copy()
    conv_diff = np.abs(np.diff(convolve2d(eyeabp, gf2, mode="valid"), axis=0))
    plt.imshow(conv_diff, cmap="gray")
    plt.show()
    conv_diff_argmax = np.argmax(conv_diff, axis=0) + filter_size

    x = np.arange(conv_diff_argmax.shape[0]) + width

    p = np.polyfit(x, conv_diff_argmax, 2)
    y = np.polyval(p, x)

    plt.imshow(eye_below_pupil, cmap="gray")
    plt.plot(x, conv_diff_argmax, "r+")
    plt.plot(x, y, "g+")
    plt.show()

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
    rmin = r - min_r
    circle_f, ryup, rydown, rxup, rxdown = circle_mask(r, lateral=False, xmax=img.shape[1], ymax=img.shape[0], x0r=x0, y0r=y0)

    circ_res = np.zeros((circle.shape[0], circle.shape[1]))

    filter_size = 15
    sigma = 0.5
    gf = np.exp(-(np.arange(filter_size) - filter_size // 2)**2/(2*sigma**2)) / (np.sqrt(2 * np.pi) * sigma)

        
    circle_ru = circle[:rmin, rmin:]
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

    
    
    circle_rl = circle[rmin:, rmin:]
    circle_rl_idx = np.where(circle_rl)
    circle_rl_idx = np.vstack(circle_rl_idx).T
    circle_rl_idx[:, 0] += r
    circle_rl_idx[:, 1] += r

    circle_rl_data = eye[circle_rl_idx[:, 0] - min_r, circle_rl_idx[:, 1] - min_r]

    circle_rl_conv = np.abs(np.diff(convolve(circle_rl_data, gf, mode="valid")))
    circle_rl_conv_idx = np.argmax(circle_rl_conv) + filter_size + 1
    circle_rl_conv_idx_yx = circle_rl_idx[circle_rl_conv_idx]

    circle_ll = circle[rmin:, :rmin + 1]
    circle_ll_idx = np.where(circle_ll)
    circle_ll_idx = np.vstack(circle_ll_idx).T
    circle_ll_idx[:, 0] += r

    circle_ll_data = eye[circle_ll_idx[:, 0] - min_r, circle_ll_idx[:, 1] + min_r]

    circle_ll_conv = np.abs(np.diff(convolve(circle_ll_data, gf, mode="valid")))
    circle_ll_conv_idx = np.argmax(circle_ll_conv) + filter_size + 1
    circle_ll_conv_idx_yx = circle_ll_idx[circle_ll_conv_idx] 

    circle_lu = circle[:rmin, :rmin + 1]
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


if False:
    import glob
    import skimage as skim
    infr_img = glob.glob("IrisRecognition/UTIRIS_infrared/*/*/*.bmp")

    from IrisSegmentation import FindEdge, FindEdgeLoss, EstimateRadius
    rvec = np.arange(42, 160, 1)
    sigma = 0.5
    filter_size = 4
    for img in infr_img[:1]:
        img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)[90:-90, 90:-90]
        img_shape = img.shape
        img_use = cv2.medianBlur(img, 3)
        edges = cv2.Canny(img_use, 30, 40)
        plt.imshow(edges, cmap="gray")
        plt.show()
        
        hough_results = skim.transform.hough_circle(edges, rvec)

        ridx, r, c = np.unravel_index(np.argmax(hough_results), hough_results.shape)
        prmax = rvec[ridx]

        r_estimate_iris = EstimateRadius([r, c], img=img_use, pupil=False, pupil_radius=prmax)
        print("Estimated iris radius: ", r_estimate_iris)
        #r_estimate_iris = int(FindEdgeLoss(img, rmin=round(prmax*1.5), rmax=max(round(prmax*3), 290), sigma=sigma, lateral=True, x0=int(c), y0=int(r), return_radius=True))
        iris_xy, iris_r = FindEdge(img, rmin=int(r_estimate_iris) - 25, rmax=int(r_estimate_iris), search_radius=5, filter_size=filter_size, sigma=sigma, lateral=True, plot_=False, x0=int(r), y0=int(c))

        fig, ax = plt.subplots()
        ax.imshow(img, cmap="gray")
        print("Estimated pupil radius: ", prmax)
        ax.add_patch(plt.Circle((c, r), rvec[ridx], color="r", fill=False))
        ax.add_patch(plt.Circle((iris_xy[1], iris_xy[0]), iris_r, color="g", fill=False))
        plt.show()