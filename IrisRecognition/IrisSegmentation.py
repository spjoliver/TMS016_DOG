from scipy.signal import convolve2d, convolve
from matplotlib.patches import Circle
from typing import Optional
import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage as skim

def FastIrisPupilScanner2(
        filename: str,
        plot_print: bool = False,
    ) -> dict:
    """

    Returns:
        dict: containing the following key-value pairs:
            "iris_xy": tuple of x and y coordinates of the iris center
            "iris_r": radius of the iris
            "pupil_xy": tuple of x and y coordinates of the pupil center
            "pupil_r": radius of the pupil
            "iris": iris image, -1 values indicate pixels not identified as iris
            "full_iris": boolean, True if the full iris could be extracted from the original image

    """



    rvec = np.arange(80, 160, 1)
    sigma = 0.5
    filter_size = 5

    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)#[75:-75, 75:-75]
    lid_imgx = img.copy()
    lid_imgx[lid_imgx > 130] = 255
    lid_imgx[lid_imgx < 60] = 0
    lid_imgx = lid_imgx.astype(float)/255
    img_use = cv2.GaussianBlur(img, (filter_size, filter_size), 0)
    
    #edges = cv2.Canny(img_use, 20, 40)#, standard
    edges = cv2.Canny(img_use, 40, 50)
    if plot_print:
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
    r_estimate_iris = int(FindEdgeLoss(img, rmin=round(prmax*1.5), rmax=min(max(round(prmax*3), 290), 335), sigma=sigma, lateral=True, x0=int(c), y0=int(r), return_radius=True))
    iris_xy, iris_r = FindEdge(img, rmin=min(int(r_estimate_iris), 335) - 25, rmax=min(int(r_estimate_iris), 335) + 15, search_radius=5, filter_size=filter_size, sigma=sigma, lateral=True, plot_=False, x0=int(c), y0=int(r))

    if iris_r < 175:
        accums, cx, cy, radii = skim.transform.hough_circle_peaks(hough_results, rvec, total_num_peaks=20)
        for center_y, center_x, radius in zip(cy[1:], cx[1:], radii[1:]):
            r = center_y
            c = center_x
            prmax = radius
            r_estimate_iris = int(FindEdgeLoss(img, rmin=round(prmax*1.5), rmax=min(max(round(prmax*3), 290), 335), sigma=sigma, lateral=True, x0=int(c), y0=int(r), return_radius=True))
            iris_xy, iris_r = FindEdge(img, rmin=min(int(r_estimate_iris), 335) - 25, rmax=min(int(r_estimate_iris), 335) + 15, search_radius=5, filter_size=filter_size, sigma=sigma, lateral=True, plot_=False, x0=int(c), y0=int(r))
            if iris_r > 175:
                break
        
    lid_imgx = img.copy()
    lid_imgx[iris_xy[0]:, :][lid_imgx[iris_xy[0]:, :] > 133] = 255
    lid_imgx[lid_imgx < 60] = 0
    lid_imgx = lid_imgx.astype(float)/255

    iris_xy_out = (iris_xy[0], iris_xy[1])
    pupil_xy_out = (r, c)

    if plot_print:
        fig, ax = plt.subplots()
        ax.imshow(img, cmap="gray")
        print("Estimated pupil radius: ", prmax)
        ax.add_patch(plt.Circle((c, r), prmax, color="r", fill=False))
        ax.add_patch(plt.Circle((iris_xy[1], iris_xy[0]), iris_r, color="g", fill=False))
        plt.plot(iris_xy[1], iris_xy[0], "r+")
        plt.plot(c, r, "g+")
        plt.show()

    isolated_iris = img.copy()

    iris_mask = skim.morphology.disk(iris_r)
    pupile_mask = skim.morphology.disk(prmax + 2)
    
    if r-prmax < 0:
        pupile_mask = pupile_mask[-(r-(prmax+2) ):, :]
    if c-prmax < 0:
        pupile_mask = pupile_mask[:, -(c-(prmax+2) ):]
    if r+prmax  + 1 > edges.shape[0]:
        pupile_mask = pupile_mask[:-(r+(prmax+2)  + 1 - edges.shape[0]), :]
    if c+prmax  + 1 > edges.shape[1]:
        pupile_mask = pupile_mask[:, :-(c+(prmax+2)  + 1 - edges.shape[1])]
    
    # pupil set to -1
    isolated_iris[max(r-(prmax+2), 0):r+(prmax+2) + 1, max(c-(prmax+2), 0):c+(prmax+2) + 1][pupile_mask.astype(bool)] = -1

    full_iris = True
    r_top = 0
    if iris_xy[0]-iris_r < 0:
        iris_mask = iris_mask[-(iris_xy[0]-iris_r):, :]
        r_top = -(iris_xy[0]-iris_r)
        full_iris = False
    if iris_xy[1]-iris_r < 0:
        iris_mask = iris_mask[:, -(iris_xy[1]-iris_r):]
        full_iris = False

    if iris_xy[0]+iris_r + 1 > edges.shape[0]:
        iris_mask = iris_mask[:-(iris_xy[0]+iris_r + 1 - edges.shape[0]), :]
        full_iris = False

    if iris_xy[1]+iris_r + 1 > edges.shape[1]:
        iris_mask = iris_mask[:, :-(iris_xy[1]+iris_r + 1 - edges.shape[1])]
        full_iris = False
    
    usefull_eye = img_use[max(iris_xy[0]-iris_r, 0):iris_xy[0]+iris_r + 1, max(iris_xy[1]-iris_r, 0):iris_xy[1]+iris_r + 1]#[skim.morphology.disk(iris_r)]
    isolated_iris = isolated_iris[max(iris_xy[0]-iris_r, 0):iris_xy[0]+iris_r + 1, max(iris_xy[1]-iris_r, 0):iris_xy[1]+iris_r + 1]#[skim.morphology.disk(iris_r)]
    
    iris_xy_out = (iris_xy_out[0] - max(iris_xy[0]-iris_r, 0), iris_xy_out[1] - max(iris_xy[1]-iris_r, 0))
    pupil_xy_out = (pupil_xy_out[0] - max(iris_xy[0]-iris_r, 0), pupil_xy_out[1] - max(iris_xy[1]-iris_r, 0))


    ## Find eyelid contours

    lid_img = img.astype(float)/255

    fimg = img[max(iris_xy[0]-iris_r, 0):iris_xy[0]+iris_r + 1, max(iris_xy[1]-iris_r, 0):iris_xy[1]+iris_r + 1]
    lid_img = lid_img[max(iris_xy[0]-iris_r, 0):iris_xy[0]+iris_r + 1, max(iris_xy[1]-iris_r, 0):iris_xy[1]+iris_r + 1]
    lid_imgx = lid_imgx[max(iris_xy[0]-iris_r, 0):iris_xy[0]+iris_r + 1, max(iris_xy[1]-iris_r, 0):iris_xy[1]+iris_r + 1]
    lid_imgx = cv2.GaussianBlur(lid_imgx, (5, 5), 0)
    lid_imgx[:pupil_xy_out[0], :] = cv2.morphologyEx(lid_imgx[:pupil_xy_out[0], :], cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    #lid_imgx[:pupil_xy_out[0] - prmax, :] = cv2.morphologyEx(lid_imgx[:pupil_xy_out[0] -prmax, :], cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    lid_img[lid_img > 0.85] = lid_img.mean()
    pup_mask = skim.morphology.disk(prmax + 2).astype(bool)
    rr, cc = skim.draw.circle_perimeter(pupil_xy_out[0], pupil_xy_out[1], prmax + 5)
    rr_new = rr[rr > pupil_xy_out[0]]
    cc_new = cc[rr > pupil_xy_out[0]]
    lid_img[pupil_xy_out[0] - pup_mask.shape[0]//2:pupil_xy_out[0] + pup_mask.shape[0]//2 + 1, pupil_xy_out[1] - pup_mask.shape[0]//2:pupil_xy_out[1] + pup_mask.shape[0]//2 + 1][pup_mask] = lid_img[rr_new, cc_new].mean()

    lid_imgx[pupil_xy_out[0] - pup_mask.shape[0]//2:pupil_xy_out[0] + pup_mask.shape[0]//2 + 1, pupil_xy_out[1] - pup_mask.shape[0]//2:pupil_xy_out[1] + pup_mask.shape[0]//2 + 1][pup_mask] = lid_imgx[rr_new, cc_new].mean()
    if plot_print:
        plt.imshow(lid_imgx, cmap="gray")
        plt.show()
    #lid_img = cv2.morphologyEx(lid_img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)), iterations=1)
    #lid_img = cv2.morphologyEx(lid_img, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    amx, rmax, rposmax = FindEyeLidEdge(
                                        img=lid_imgx, 
                                        pupil_rpos=pupil_xy_out[0],
                                        c=pupil_xy_out[1],
                                        rposjump=7,
                                        iris_radius=iris_r,
                                        pupil_radius=prmax,
                                        n_radiusjumps=30,
                                        plot=plot_print,
                                        upper=True,
                                        anglemin=-np.pi/6,
                                        anglemax=np.pi/6,
                                        n_angles=15,
                                    )
    
    rr, cc = skim.draw.ellipse_perimeter(rposmax, pupil_xy_out[1], rmax, int(iris_r*1.5), orientation=amx, shape=lid_img.shape)
    rr_upper = rr[rr < rposmax]
    cc_upper = cc[rr < rposmax]
    if np.min(rr_upper) > 15:
        rr, cc = skim.draw.ellipse_perimeter(rposmax + 35, pupil_xy_out[1], rmax, int(iris_r*1.5), orientation=amx, shape=lid_img.shape)
        rr_upper = rr[rr < rposmax]
        cc_upper = cc[rr < rposmax]


    for r, c in zip(rr_upper, cc_upper):
        isolated_iris[:r, c] = -1


    amx, rmax, rposmax= FindEyeLidEdge(
                                        img=lid_imgx, 
                                        pupil_rpos=pupil_xy_out[0],
                                        c=pupil_xy_out[1],
                                        rposjump=7,
                                        n_radiusjumps=30,
                                        iris_radius=iris_r,
                                        pupil_radius=prmax,
                                        plot=plot_print,
                                        upper=False,
                                        anglemin=-np.pi/6,
                                        anglemax=np.pi/6,
                                        n_angles=15,
                                    )
    
    rr, cc = skim.draw.ellipse_perimeter(rposmax, pupil_xy_out[1], rmax, int(iris_r*1.5), orientation=amx, shape=lid_img.shape)
    rr_lower = rr[rr > rposmax]
    cc_lower = cc[rr > rposmax]
    if np.max(rr_lower) < lid_img.shape[0] - 30:
        rr, cc = skim.draw.ellipse_perimeter(rposmax - 5, pupil_xy_out[1], rmax, int(iris_r*1.5), orientation=amx, shape=lid_img.shape)
        rr_lower = rr[rr > rposmax]
        cc_lower = cc[rr > rposmax]

    for r, c in zip(rr_lower, cc_lower):
        isolated_iris[r:, c] = -1
    

    isolated_iris[iris_mask == 0] = -1

    
    if plot_print:
        fimg = np.dstack([fimg, fimg, fimg])
        fimg[rr_upper, cc_upper] = [0, 0, 255]
        fimg[rr_lower, cc_lower] = [0, 0, 255]
        fig, ax = plt.subplots()
        ax.imshow(fimg)
        ax.add_patch(plt.Circle((pupil_xy_out[1], pupil_xy_out[0]), prmax, color="r", fill=False))
        ax.add_patch(plt.Circle((iris_xy_out[1], iris_xy_out[0]), iris_r, color="g", fill=False))
        ax.plot(pupil_xy_out[1], pupil_xy_out[0], "r+")
        ax.plot(iris_xy_out[1], iris_xy_out[0], "g+")
        plt.show()
    
    return {
        "iris": isolated_iris,
        "iris_xy": iris_xy_out,
        "pupil_xy": pupil_xy_out,
        "iris_r": iris_r,
        "pupil_r": prmax,
        "full_iris": full_iris,
    }

def FastIrisPupilScanner(
        filename: str,
        plot_print: bool = False,
    ) -> dict:
    """

    Returns:
        dict: containing the following key-value pairs:
            "iris_xy": tuple of x and y coordinates of the iris center
            "iris_r": radius of the iris
            "pupil_xy": tuple of x and y coordinates of the pupil center
            "pupil_r": radius of the pupil
            "iris": iris image, -1 values indicate pixels not identified as iris
            "full_iris": boolean, True if the full iris could be extracted from the original image

    """
    
    rvec = np.arange(50, 160, 1)
    sigma = 0.5
    filter_size = 5

    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)#[75:-75, 75:-75]
    img_shape = img.shape
    img_use = cv2.GaussianBlur(img, (filter_size, filter_size), 0)
    
    edges = cv2.Canny(img_use, 20, 40)#, standard
    #edges = cv2.Canny(img_use, 42, 53)
    if plot_print:
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

    if plot_print:
        fig, ax = plt.subplots()
        ax.imshow(img, cmap="gray")
        print("Estimated pupil radius: ", prmax)
        ax.add_patch(plt.Circle((c, r), prmax, color="r", fill=False))
        ax.add_patch(plt.Circle((iris_xy[1], iris_xy[0]), iris_r, color="g", fill=False))
        plt.plot(iris_xy[1], iris_xy[0], "r+")
        plt.plot(c, r, "g+")
        plt.show()

    iris_xy_out = (iris_xy[0], iris_xy[1])
    pupil_xy_out = (r, c)
  
    # Search for lines, something similar can be done
    edges = cv2.Canny(img_use, 15, 20)
    edges[max(r-prmax-15, 0):r+prmax+15, max(c-prmax-15, 0):c+prmax+60] = 0
    edges[max(r-prmax-70, 0):r+prmax+70, max(c-prmax-70, 0):c+prmax+70] = cv2.GaussianBlur(edges[max(r-prmax-70, 0):r+prmax+70, max(c-prmax-70, 0):c+prmax+70], (5, 5), 0)
    edges = cv2.GaussianBlur(edges, (5, 5), 0)
    edges = convolve2d(edges, np.ones((5, 5)), mode="same", fillvalue=0)
    edges[edges < edges.max()//2] = 0
    edges[edges > 0] = 1
    edges[max(r-prmax-100, 0):r+prmax+100, max(c-prmax-100, 0):c+prmax+100] = cv2.GaussianBlur(edges[max(r-prmax-100, 0):r+prmax+100, max(c-prmax-100, 0):c+prmax+100], (5, 5), 0)
    struct_elem1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, struct_elem1, iterations=1)
    edges[edges < 1] = 0
    struct_elem3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, struct_elem1, iterations=1)
    edges[edges < 1] = 0
    struct_elem2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    edges = cv2.dilate(edges, struct_elem2, iterations=8)
    edges = cv2.erode(edges, struct_elem2, iterations=5)
    edges[edges < 1] = 0
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, struct_elem1, iterations=1)
    edges[edges < 1] = 0
    edges = cv2.morphologyEx(edges, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20)), iterations=1)
    edges[edges < 1] = 0
    edges = cv2.morphologyEx(edges, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=1)
    #edges = convolve2d(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20)), mode="same", fillvalue=0.5)
    edges[edges < 1] = 0
    edges = cv2.morphologyEx(edges, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)), iterations=1)
    edges[edges < 1] = 0
    edges = cv2.morphologyEx(edges, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=3)
    edges[edges < 1] = 0
    edges = cv2.dilate(edges, struct_elem2, iterations=3)
    edges[edges < 1] = 0
    edges = cv2.erode(edges, struct_elem2, iterations=2)
    edges[edges < 1] = 0
    edges = cv2.dilate(edges, struct_elem2, iterations=3)
    edges[edges < 1] = 0
    edges = cv2.erode(edges, struct_elem2, iterations=2)
    edges[edges < 1] = 0
    if plot_print:
        plt.imshow(edges, cmap="gray")
        plt.show()
    iris_mask = skim.morphology.disk(iris_r)
    pupile_mask = skim.morphology.disk(prmax)
    
    if r-prmax < 0:
        pupile_mask = pupile_mask[-(r-prmax ):, :]
    if c-prmax < 0:
        pupile_mask = pupile_mask[:, -(c-prmax ):]
    if r+prmax  + 1 > edges.shape[0]:
        pupile_mask = pupile_mask[:-(r+prmax  + 1 - edges.shape[0]), :]
    if c+prmax  + 1 > edges.shape[1]:
        pupile_mask = pupile_mask[:, :-(c+prmax  + 1 - edges.shape[1])]
    
    # pupil set to -1
    img[max(r-prmax, 0):r+prmax + 1, max(c-prmax, 0):c+prmax + 1][pupile_mask.astype(bool)] = -1

    full_iris = True
    r_top = 0
    if iris_xy[0]-iris_r < 0:
        iris_mask = iris_mask[-(iris_xy[0]-iris_r):, :]
        #iris_xy_out = (iris_xy_out[0] + (iris_xy[0]-iris_r), iris_xy_out[1])
        #pupil_xy_out = (pupil_xy_out[0] + (iris_xy[0]-iris_r), pupil_xy_out[1])
        r_top = -(iris_xy[0]-iris_r)
        full_iris = False
    if iris_xy[1]-iris_r < 0:
        iris_mask = iris_mask[:, -(iris_xy[1]-iris_r):]
        #iris_xy_out = (iris_xy_out[0], iris_xy_out[1] + (iris_xy[1]-iris_r))
        #pupil_xy_out = (pupil_xy_out[0], pupil_xy_out[1] + (iris_xy[1]-iris_r))
        full_iris = False

    if iris_xy[0]+iris_r + 1 > edges.shape[0]:
        iris_mask = iris_mask[:-(iris_xy[0]+iris_r + 1 - edges.shape[0]), :]
        full_iris = False

    if iris_xy[1]+iris_r + 1 > edges.shape[1]:
        iris_mask = iris_mask[:, :-(iris_xy[1]+iris_r + 1 - edges.shape[1])]
        full_iris = False
    
    edge_mask = edges[max(iris_xy[0]-iris_r, 0):iris_xy[0]+iris_r + 1, max(iris_xy[1]-iris_r, 0):iris_xy[1]+iris_r + 1].astype(bool)#[skim.morphology.disk(iris_r)]
    usefull_eye = img[max(iris_xy[0]-iris_r, 0):iris_xy[0]+iris_r + 1, max(iris_xy[1]-iris_r, 0):iris_xy[1]+iris_r + 1]#[skim.morphology.disk(iris_r)]
    iris_img = img_use[max(iris_xy[0]-iris_r, 0):iris_xy[0]+iris_r + 1, max(iris_xy[1]-iris_r, 0):iris_xy[1]+iris_r + 1]

    

    iris_xy_out = (iris_xy_out[0] - max(iris_xy[0]-iris_r, 0), iris_xy_out[1] - max(iris_xy[1]-iris_r, 0))
    pupil_xy_out = (pupil_xy_out[0] - max(iris_xy[0]-iris_r, 0), pupil_xy_out[1] - max(iris_xy[1]-iris_r, 0))

    #LocateEyelids(iris_r, prmax, iris_xy[1], iris_xy[0], img_use)
    test = cv2.Canny(img[max(iris_xy[0]-iris_r, 0):iris_xy[0]+iris_r + 1, max(iris_xy[1]-iris_r, 0):iris_xy[1]+iris_r + 1][(iris_r - r_top+ +15):, :], 18, 25)
    if test.shape[0] > 100 and False:
        test[test > 0.1] = 255
        plt.imshow(test, cmap="gray")
        plt.show()
        plt.imshow(img[max(iris_xy[0]-iris_r, 0):iris_xy[0]+iris_r + 1, max(iris_xy[1]-iris_r, 0):iris_xy[1]+iris_r + 1][(iris_r- r_top + +15):, :], cmap="gray")
        plt.show()
        
        test = cv2.GaussianBlur(test, (5, 5), 0)
        #test = cv2.erode(test, struct_elem2, iterations=1)
        test = cv2.dilate(test, struct_elem2, iterations=1)
        test[:int(test.shape[0]/2), test.shape[1]//2 - test.shape[1]//5:test.shape[1]//2 + test.shape[1]//5] = 0
        
        xtop = np.arange(test.shape[1])
        topy = np.argmax(test, axis=0)
        maskx = (xtop > test.shape[1]//2 - test.shape[1]//5) & (xtop < test.shape[1]//2 + test.shape[1]//5) & (topy < 50)
        masky1 = (xtop > test.shape[1]//2 + test.shape[1]//5) & (topy > test.shape[0] - 15)
        masky2 = (xtop < test.shape[1]//2 - test.shape[1]//5) & (topy > test.shape[0] - 15)
    
        topyn = topy[(~maskx) & (~masky1) & (~masky2)]
       
        xtopn = xtop[(~maskx) & (~masky1) & (~masky2)]
        p = np.polyfit(xtopn, topyn, 2)
        yvals = np.polyval(p, xtop)
        plt.imshow(img[max(iris_xy[0]-iris_r, 0):iris_xy[0]+iris_r + 1, max(iris_xy[1]-iris_r, 0):iris_xy[1]+iris_r + 1][(iris_r - r_top+ +15):, :], cmap="gray")
        plt.plot(xtopn, topyn, "r+")
        plt.plot(xtop, yvals, "g-")
        plt.show()

    # outside of iris set to -1
    usefull_eye[iris_mask == 0] = -1

    # edge_mask might have detected pixels not belonging to the iris, set them to -1
    #usefull_eye[edge_mask == True] = -1
    if plot_print:
        #plt.imshow(usefull_eye, cmap="gray")
        #plt.plot(iris_xy_out[1], iris_xy_out[0], "r+")
        #plt.plot(pupil_xy_out[1], pupil_xy_out[0], "b+")
        #plt.show()
        plt.imshow(usefull_eye, cmap="gray")
        ax.add_patch(plt.Circle((c, r), prmax, color="r", fill=False))
        ax.add_patch(plt.Circle((iris_xy[1], iris_xy[0]), iris_r, color="g", fill=False))
        plt.plot(iris_xy[1], iris_xy[0], "r+")
        plt.plot(c, r, "g+")
    
    return {
        "iris": usefull_eye,
        "iris_xy": iris_xy_out,
        "pupil_xy": pupil_xy_out,
        "iris_r": iris_r,
        "pupil_r": prmax,
        "full_iris": full_iris,
        "usefull_eye": iris_img
    }


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

def EyeLidMask(c: int, r: int, radius: int, iris_radius: int, angle: float, img_shape: tuple, upper: bool) -> np.ndarray:

    rr, cc = skim.draw.ellipse_perimeter(r, c, radius, int(iris_radius*1.5), orientation=angle, shape=(img_shape[0], img_shape[1]))
    if upper:
        rr_inbound = rr[rr < r]
        cc_inbound = cc[rr < r]
    else:
        rr_inbound = rr[rr > r]
        cc_inbound = cc[rr > r]

    
    return rr_inbound, cc_inbound

def LineIntegralEyeLid(img: np.ndarray, c: int, r: int, radius: int, iris_radius: int, angle: float, upper: bool) -> float:
    rr, cc = EyeLidMask(c, r, radius, iris_radius, angle, img.shape, upper)
    #a = img[y0 - ryup:y0 + rydown + 1, x0 - rxup:x0 + rxdown + 1][circle]
    nan = False
    if rr.shape[0] < (img.shape[0] + img.shape[1]) / 10:
        nan = True
        return np.nan, nan
    return (img[rr, cc]).sum() / rr.shape[0], nan

def drLineIntegralMultiEyeLid(img: np.ndarray, c: int, rvec: np.ndarray, radius: int, iris_radius: int, angle: float, upper: bool) -> np.ndarray:
    #lint = np.zeros(rmax - rmin + 2)
    lint = np.zeros(rvec.shape[0])
    nan_in = False
    for i, r in enumerate(rvec):#:range(rmin, rmax + 2):
        lint[i], nan = LineIntegralEyeLid(img, c, r, radius, iris_radius, angle, upper)
        nan_in = nan_in or nan
    
    if nan_in:
        # ugly but got some weird error one time so now this is it.
        try:
            mask = np.isnan(lint)
            lint[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), lint[~mask])
        except:
            pass
    
    return np.diff(lint)

def FindEyeLidEdge(
        img: np.ndarray, 
        pupil_rpos: int,
        pupil_radius: int,
        c: int, 
        rposjump: int, 
        iris_radius: int,
        radiusmin: Optional[int]=None,
        radiusmax: Optional[int]=None,
        n_radiusjumps: Optional[int]=150,
        anglemin: float=-np.pi/18, 
        anglemax: float=np.pi/18, 
        n_angles: int=10, 
        upper: bool=True,
        filter_size: int=3,
        sigma: float=1.0,
        plot: bool=False
        ):
    """
    Returns:
        angleval, radiusval, rposmaxval
    """
    if radiusmin is None:
        radiusmin = int(pupil_rpos * 0.3)
    if radiusmax is None:
        if upper:
            radiusmax = int(pupil_rpos * 3)
        else:
            radiusmax = int(pupil_rpos * 3.2)
            radiusmin = int(pupil_rpos * 1)

    radiusvec = np.round(np.linspace(radiusmin, radiusmax , n_radiusjumps)).astype(int)
    
    anglevec = np.linspace(anglemin, anglemax, n_angles)

    angleval = np.zeros((anglevec.shape[0], radiusvec.shape[0]))
    radiusval = np.zeros((anglevec.shape[0], radiusvec.shape[0]))
    rposmaxval = np.zeros((anglevec.shape[0], radiusvec.shape[0]))
    maxblurval = np.zeros((anglevec.shape[0], radiusvec.shape[0]))

    # this is stupid to have, since it is linear, but I started with the
    # assumption of a quadratic relationship, so I'll keep it for now.
    if upper:
        rmin = (np.array(
        [1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
        )*pupil_rpos).astype(int).astype(float)

        posmin = (np.array(
            [2.3, 2.2, 2.1, 2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3]
        )*pupil_rpos).astype(int).astype(float)

        radii_to_pos_min = np.polyfit(rmin, posmin, 2)

        rmax = (np.array(
            [1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
        )*pupil_rpos).astype(int).astype(float)

        posmax = (np.array(
            [1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
        )*pupil_rpos).astype(int).astype(float)

        radii_to_pos_max = np.polyfit(rmax, posmax, 2)
    else:
        iris_rmin = (np.array(
        [1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
        )*pupil_rpos).astype(int).astype(float)

        posmin = ((np.array(
            [-0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75]
        ) + 0.3)*pupil_rpos).astype(int).astype(float)

        radii_to_pos_min = np.polyfit(iris_rmin, posmin, 2)

        iris_rmax = (np.array(
            [1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
        )*pupil_rpos).astype(int).astype(float)

        posmax = ((np.array(
            [0.74, 0.84, 0.94, 1.04, 1.14, 1.24, 1.34, 1.44, 1.54, 1.64, 1.74]
        ) -0.2)*pupil_rpos).astype(int).astype(float)

        radii_to_pos_max = np.polyfit(iris_rmax, posmax, 2)

    if upper:
        rposjump = -rposjump

    for i, angle in enumerate(anglevec):
        
        for j, radius in enumerate(radiusvec):

            rposmin = int(np.polyval(radii_to_pos_min, radius))
            rposmax = int(np.polyval(radii_to_pos_max, radius))

            rposvec = np.arange(rposmin, rposmax + 1, rposjump)
            angleval[i, j] = angle
            radiusval[i, j] = radius
            drLIM = drLineIntegralMultiEyeLid(img, c, rposvec, radius, iris_radius, angle, upper)
            cgdrLIM = ConvolveGaussiandrLI(drLIM, filter_size=filter_size, sigma=sigma)
            arg_max_blur = np.argmax(cgdrLIM)
            rposmaxval[i, j] = rposvec[arg_max_blur]
            maxblurval[i, j] = cgdrLIM[arg_max_blur]
    
    maxblurval = maxblurval.flatten()
    rposmaxval = rposmaxval.flatten()
    angleval = angleval.flatten()
    radiusval = radiusval.flatten()
    maxbluridx = np.unravel_index(np.argmax(maxblurval), maxblurval.shape)
    amx = int(angleval[maxbluridx])
    rmax = int(radiusval[maxbluridx])
    rposmax = int(rposmaxval[maxbluridx])

    if not upper:
        top_indices = np.argsort(maxblurval)
        rr, cc = skim.draw.ellipse_perimeter(rposmax, c, rmax, int(iris_radius*1.5), orientation=amx, shape=img.shape)
        max_rr = rr[rr > rposmax].max()
        if max_rr - 50 < pupil_rpos + pupil_radius + 5:
            print("Under eyelid too closse to pupil, search for estimation further away:")
            for i in range(1, top_indices.shape[0]):
                top_blur = top_indices[-i]
                rposmax = int(rposmaxval[top_blur])
                amx = int(angleval[top_blur])
                rmax = int(radiusval[top_blur])
                rr, cc = skim.draw.ellipse_perimeter(rposmax, c, rmax, int(iris_radius*1.5), orientation=amx, shape=img.shape)
                max_rr = rr[rr > rposmax].max()
                if max_rr - 50 < pupil_rpos + pupil_radius + 5:
                    continue
                else:
                    break
        else:
            maxbluridx = np.unravel_index(np.argmax(maxblurval), maxblurval.shape)
            amx = int(angleval[maxbluridx])
            rmax = int(radiusval[maxbluridx])
            rposmax = int(rposmaxval[maxbluridx])
    
    if plot:
        img_plt = img.copy()
        rr, cc = skim.draw.ellipse_perimeter(rposmax, c, rmax, int(iris_radius*1.5), orientation=amx, shape=img.shape)
        if upper:
            img_plt[rr[rr < rposmax], cc[rr < rposmax]] = 0
        else:
            img_plt[rr[rr > rposmax], cc[rr > rposmax]] = 0
        plt.imshow(img_plt, cmap='gray')
        plt.show()

    rposmin_opt = int(np.polyval(radii_to_pos_min, rmax))
    rposmax_opt = int(np.polyval(radii_to_pos_max, rmax))


    #amx, rposmax = OptimizeEyeLidEdge(img, upper, rposmin_opt, rposmax_opt, c, rmax, iris_radius, filter_size=3)

    """
    if plot:
        img_plt = img.copy()
        rr, cc = skim.draw.ellipse_perimeter(rposmax, c, rmax, int(iris_radius*1.5), orientation=amx, shape=img.shape)
        if upper:
            img_plt[rr[rr < rposmax], cc[rr < rposmax]] = 0
        else:
            img_plt[rr[rr > rposmax], cc[rr > rposmax]] = 0
        plt.imshow(img_plt, cmap='gray')
        plt.show()
    """

    return amx, rmax, rposmax


def OptimizeEyeLidEdge(img, upper, rposmin, rposmax, c, rmax, iris_radius, filter_size=3, sigma=1.5):

    amin = -np.pi/18
    amax = -amin
    n_angles = 50
    if upper:
        arange = np.linspace(amax, amin, n_angles)
    else:
        arange = np.linspace(amin, amax, n_angles)
    amaxval = np.zeros(n_angles)
    maxblurval = np.zeros(n_angles)
    rposmaxval = np.zeros(n_angles)

    if upper:
        rposvec = np.arange(rposmin, rposmax + 1, -1)
    else:
        rposvec = np.arange(rposmin, rposmax + 1, 1)


    for i, angle in enumerate(arange):
        drLIM = drLineIntegralMultiEyeLid(img, c, rposvec, rmax, iris_radius, angle, upper)
        cgdrLIM = ConvolveGaussiandrLI(drLIM, filter_size=filter_size, sigma=sigma)

        arg_max_blur = np.argmax(cgdrLIM)
        rposmaxval[i] = rposvec[arg_max_blur]
        maxblurval[i] = cgdrLIM[arg_max_blur]
        amaxval[i] = angle
    
    maxbluridx = np.argmax(maxblurval)
    amx = amaxval[maxbluridx]
    rposmax = rposmaxval[maxbluridx]
    return int(amx), int(rposmax)

        

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
        
        left_conv = np.abs(np.diff(convolve2d(left_datah, gf, mode="valid"), axis=1))
        if False:
            print("dist right: ", dist_right)
            plt.imshow(left_conv, cmap="gray")
            plt.show()
        
       
        dist_left = left_datah.shape[1] - (np.median(np.argmax(left_conv, axis=1)) + filter_size)
        if max(top_index2[1] - data_length, 0) < 20:
            dist_left = left_datah.shape[1] - (np.median(np.argmax(left_conv[:, 30:], axis=1)) + filter_size + 30)

        
        top_conv = np.abs(np.diff(convolve2d(right_datav, gf2, mode="valid"), axis=0))
        if False:
            print("dist left: ", dist_left)
            plt.imshow(top_conv, cmap="gray")
            plt.show()
        dist_top = np.median(np.argmax(top_conv, axis=0)) + filter_size
        if top_index2[0] + data_length > img.shape[0] - 20:
            dist_top = np.median(np.argmax(top_conv[:-30, :], axis=0)) + filter_size
        
        bottom_conv = np.abs(np.diff(convolve2d(left_datav, gf2, mode="valid"), axis=0))
        if False:
            print("dist top: ", dist_top)
            plt.imshow(bottom_conv, cmap="gray")
            plt.show()
        dist_bottom = left_datav.shape[0] - (np.median(np.argmax(bottom_conv, axis=0)) + filter_size)
        if max(top_index2[0] - data_length, 0) < 20:
            dist_bottom = left_datav.shape[0] - (np.median(np.argmax(bottom_conv[30:, :], axis=0)) + filter_size + 30)
        if False:
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

        right_conv = right_conv
        dist_right = np.median(np.argmax(right_conv, axis=1)) + filter_size
        left_conv = np.abs(np.diff(convolve2d(left_datah, gf, mode="valid"), axis=1))
        plt.imshow(left_conv,  cmap='gray')
        plt.show()
        dist_left = data_length - np.median(np.argmax(left_conv, axis=1)) + filter_size
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
        dist_right = np.median(np.argmax(right_conv, axis=1)) + filter_size + pupile_adder

        left_conv = np.abs(np.diff(convolve2d(left_datah, gf, mode="valid"), axis=1))
        dist_left = np.median(np.argmax(left_conv, axis=1)) + filter_size + pupile_adder

        left_conv = left_conv.sum(axis=0)
        
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
        plot_: bool=False,
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

def FindPupilIris(img: np.ndarray, filter_size: int=3, sigma: float=1.0, lateral: bool=False, plot_img: Optional[np.ndarray]=None, print_plot: bool=False) -> int:

    #struct_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    #img_erode = cv2.morphologyEx(img_use, cv2.MORPH_OPEN, struct_elem, iterations=1)


    top_index = FindPupileCenter(img)
    r_estimate_pupil = int(EstimateRadius(top_index, img, pupil=True))
    if print_plot:
        print("Estimated pupil radius: ", r_estimate_pupil)
        print("Estimated pupil center: ", top_index)

    pup_xy, pup_r = FindEdge(img, rmin=r_estimate_pupil - 10, rmax=r_estimate_pupil + 15, search_radius=20, filter_size=filter_size, sigma=sigma, lateral=True, plot_img=plot_img, plot_=False, x0=int(top_index[1]), y0=int(top_index[0]))
    #r_estimate_iris = int(EstimateRadius(pup_xy, img, pupil=False, pupil_radius=pup_r))
    r_estimate_iris = int(FindEdgeLoss(img, rmin=round(pup_r*1.5), rmax=round(pup_r*3), sigma=sigma, lateral=True, x0=int(pup_xy[1]), y0=int(pup_xy[0]), return_radius=True))
    if print_plot:
        print("Estimated iris radius: ", r_estimate_iris)
    iris_xy, iris_r = FindEdge(img, rmin=r_estimate_iris - 30, rmax=r_estimate_iris + 25, search_radius=10, filter_size=filter_size, sigma=sigma, lateral=True, plot_img=plot_img, plot_=False, x0=int(pup_xy[1]), y0=int(pup_xy[0]))

    if print_plot:
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
    
    return pup_xy, pup_r, iris_xy, iris_r

