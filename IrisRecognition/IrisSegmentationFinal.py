from scipy.signal import convolve2d, convolve
from matplotlib.patches import Circle
from typing import Optional
import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage as skim
from scipy.interpolate import interp1d
from typing import Union, Tuple

def FastIrisPupilScanner2(
        filename: str,
        plot_print: bool = False,
        dilate_eyelid_threshold: bool = False,
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


    

    # parameters for blur and convolution
    sigma = 0.5
    filter_size = 3

    # Load image
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)#[75:-75, 75:-75]

    # rvec defines the radius of the circle to be searched for, pupil's seem to be in this range most of the time in the dataset UTIRIS
    rvec = np.arange(80, 160, 1)
    # Find pupil
    prmax, r, c, hres_left = FindPupil(
        img=img,
        filter_size=filter_size,
        rvec=rvec,
        plot_print=plot_print,
    )
    

    #r_estimate_iris = EstimateRadius([c, r], img=img_use, pupil=False, pupil_radius=prmax)
    #print("Estimated iris radius: ", r_estimate_iris)
    r_estimate_iris = int(FindEdgeLoss(img, rmin=round(prmax*1.5), rmax=min(max(round(prmax*3), 290), 335), sigma=sigma, lateral=True, x0=int(c), y0=int(r), return_radius=True))
    iris_xy, iris_r = FindEdge(img, rmin=min(int(r_estimate_iris), 335) - 25, rmax=min(int(r_estimate_iris), 335) + 15, search_radius=5, filter_size=filter_size, sigma=sigma, lateral=True, plot_=False, x0=int(c), y0=int(r))

    if iris_r < 175:
        for center_y, center_x, radius in hres_left:
            r = center_y
            c = center_x
            prmax = radius
            r_estimate_iris = int(FindEdgeLoss(img, rmin=round(prmax*1.5), rmax=min(max(round(prmax*3), 290), 335), sigma=sigma, lateral=True, x0=int(c), y0=int(r), return_radius=True))
            iris_xy, iris_r = FindEdge(img, rmin=min(int(r_estimate_iris), 335) - 25, rmax=min(int(r_estimate_iris), 335) + 15, search_radius=5, filter_size=filter_size, sigma=sigma, lateral=True, plot_=False, x0=int(c), y0=int(r))
            if iris_r > 175:
                break
    
    # save estimations for ease of use
    iris_xy_out = (iris_xy[0], iris_xy[1])
    pupil_xy_out = (r, c)
    isolated_iris = img.copy()

    if plot_print:
        fig, ax = plt.subplots()
        ax.imshow(img, cmap="gray")
        print("Estimated pupil radius: ", prmax)
        ax.add_patch(plt.Circle((c, r), prmax, color="r", fill=False))
        ax.add_patch(plt.Circle((iris_xy[1], iris_xy[0]), iris_r, color="g", fill=False))
        plt.plot(iris_xy[1], iris_xy[0], "r+")
        plt.plot(c, r, "g+")
        plt.show()

    
    ############################# PUPIL MASK ##################################
    # sort of bias adjustment
    prmax += 2
    pupile_mask = skim.morphology.disk(prmax)
    # Make sure pupil is not out of bounds
    if r-prmax < 0:
        pupile_mask = pupile_mask[-(r-(prmax) ):, :]
    if c-prmax < 0:
        pupile_mask = pupile_mask[:, -(c-(prmax) ):]
    if r+prmax  + 1 > img.shape[0]:
        pupile_mask = pupile_mask[:-(r+(prmax)  + 1 - img.shape[0]), :]
    if c+prmax  + 1 > img.shape[1]:
        pupile_mask = pupile_mask[:, :-(c+(prmax)  + 1 - img.shape[1])]
    
    ############################# IRIS MASK ##################################
    iris_mask = skim.morphology.disk(iris_r)
    full_iris = True
    # Make sure iris is not out of bounds
    if iris_xy[0]-iris_r < 0:
        iris_mask = iris_mask[-(iris_xy[0]-iris_r):, :]
        full_iris = False
    if iris_xy[1]-iris_r < 0:
        iris_mask = iris_mask[:, -(iris_xy[1]-iris_r):]
        full_iris = False
    if iris_xy[0]+iris_r + 1 > img.shape[0]:
        iris_mask = iris_mask[:-(iris_xy[0]+iris_r + 1 - img.shape[0]), :]
        full_iris = False
    if iris_xy[1]+iris_r + 1 > img.shape[1]:
        iris_mask = iris_mask[:, :-(iris_xy[1]+iris_r + 1 - img.shape[1])]
        full_iris = False
    
    ############################# PUPIL SET TO -1 IN ISOLATED IRIS AND IRIS INDEXED, CENTER ESTIMATIONS ADJUSTED FOR THIS ##################################
    # pupil set to -1
    isolated_iris[max(r-(prmax), 0):r+(prmax) + 1, max(c-(prmax), 0):c+(prmax) + 1][pupile_mask.astype(bool)] = -1

    # Now select only iris block region of eye
    isolated_iris = isolated_iris[max(iris_xy[0]-iris_r, 0):iris_xy[0]+iris_r + 1, max(iris_xy[1]-iris_r, 0):iris_xy[1]+iris_r + 1]
    
    iris_xy_out = (iris_xy_out[0] - max(iris_xy[0]-iris_r, 0), iris_xy_out[1] - max(iris_xy[1]-iris_r, 0))
    pupil_xy_out = (pupil_xy_out[0] - max(iris_xy[0]-iris_r, 0), pupil_xy_out[1] - max(iris_xy[1]-iris_r, 0))

    ############################# EYELID ESTIMATION ##################################

    rru, ccu, rrl, ccl = FindEyeLids(
        img=img,
        iris_xy=iris_xy,
        iris_r=iris_r,
        pupil_xy_out=pupil_xy_out,
        prmax=prmax,
        plot_print=plot_print,
        dilate_eyelid_threshold=dilate_eyelid_threshold
    )
 
    # set pixels above upper eyelid to -1
    for r, c in zip(rru, ccu):
        isolated_iris[:r, c] = -1
    
    # set pixels below lower eyelid to -1
    for r, c in zip(rrl, ccl):
        isolated_iris[r:, c] = -1
    
    # set pixels outside iris to -1
    isolated_iris[iris_mask == 0] = -1

    
    if plot_print:
        fimg = img[max(iris_xy[0]-iris_r, 0):iris_xy[0]+iris_r + 1, max(iris_xy[1]-iris_r, 0):iris_xy[1]+iris_r + 1]
        fimg = np.dstack([fimg, fimg, fimg])
        fimg[rru, ccu] = [0, 0, 255]
        fimg[rrl, ccl] = [0, 0, 255]
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

def FindEyeLids(
        img: np.ndarray,
        iris_xy: Tuple[int, int],
        iris_r: int,
        pupil_xy_out: Tuple[int, int],
        prmax: int,
        plot_print: bool = False,
        dilate_eyelid_threshold: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Finds eyelid contours
    Returns:
        rru, ccu, rrl, ccl
    rr stands for the row index of the contour, u and l stand for upper and lower eyelid, respectively
    cc stands for the column index of the contour
    """
    # copy image and perform thresholding
    lid_imgx = img.copy()
    lid_imgx[iris_xy[0]:, :][lid_imgx[iris_xy[0]:, :] > 130] = 255
    # The darker pixels are set to 255 since they often are around the upper eyelid
    # this often makes it so that the derivative is stronger when coming from below,
    # i.e. from the pupil and iris, which is what we want.
    # Otherwise, the derivative is stronger when coming from above, i.e. from the eyelid,
    # which could mess estimations up.
    lid_imgx[lid_imgx < 60] = 255
    # convert to float and normalize
    lid_imgx = lid_imgx.astype(float)/255

    # Now select only iris block region of eye and blur
    lid_imgx = lid_imgx[max(iris_xy[0]-iris_r, 0):iris_xy[0]+iris_r + 1, max(iris_xy[1]-iris_r, 0):iris_xy[1]+iris_r + 1]
    lid_imgx = cv2.GaussianBlur(lid_imgx, (5, 5), 0)
    
    if dilate_eyelid_threshold:
        # Could add other morphological operations here, but did not seem optimal during testing, not even this dilation
        lid_imgx[:pupil_xy_out[0], :] = cv2.morphologyEx(lid_imgx[:pupil_xy_out[0], :], cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    pup_mask = skim.morphology.disk(prmax).astype(bool)
    rr, cc = skim.draw.circle_perimeter(pupil_xy_out[0], pupil_xy_out[1], prmax + 5)
    rr_new = rr[rr > pupil_xy_out[0]]
    cc_new = cc[rr > pupil_xy_out[0]]

    # set pupil region to mean of surrounding pixels outside in the iris region (hopefully)
    lid_imgx[pupil_xy_out[0] - pup_mask.shape[0]//2:pupil_xy_out[0] + pup_mask.shape[0]//2 + 1, pupil_xy_out[1] - pup_mask.shape[0]//2:pupil_xy_out[1] + pup_mask.shape[0]//2 + 1][pup_mask] = lid_imgx[rr_new, cc_new].mean()
    if plot_print:
        plt.imshow(lid_imgx, cmap="gray")
        plt.show()
    # Find upper eyelid
    amx, rmax, rposmax = FindEyeLidEdge(
                                    img=lid_imgx, 
                                    pupil_rpos=pupil_xy_out[0],
                                    c=pupil_xy_out[1],
                                    rposjump=2,
                                    iris_radius=iris_r,
                                    pupil_radius=prmax,
                                    n_radiusjumps=50,
                                    plot=plot_print,
                                    upper=True,
                                    anglemin=-np.pi/6,
                                    anglemax=np.pi/6,
                                    n_angles=20,
                                    sigma=0.5
                                    )
    # Check if upper eyelid is close to top or not, often needs a shift down from estimations if eyelid is blocking part of iris
    alpha_u = np.linspace(np.pi, 2*np.pi, 700)
    rru, ccu = EyeLidMask(pupil_xy_out[1], rposmax, rmax, iris_r, amx, lid_imgx.shape, True, alpha_u)
    min_rr_upper = rru.min()
    if min_rr_upper > 10:
        rru, ccu = EyeLidMask(pupil_xy_out[1], rposmax+20, rmax, iris_r, amx, lid_imgx.shape, True, alpha_u)

    # Find lower eyelid
    amx, rmax, rposmax= FindEyeLidEdge(
                                    img=lid_imgx, 
                                    pupil_rpos=pupil_xy_out[0],
                                    c=pupil_xy_out[1],
                                    rposjump=2,
                                    n_radiusjumps=30,
                                    iris_radius=iris_r,
                                    pupil_radius=prmax,
                                    plot=plot_print,
                                    upper=False,
                                    anglemin=-np.pi/6,
                                    anglemax=np.pi/6,
                                    n_angles=20,
                                    sigma=0.5
                                    )
    
    # Check if lower eyelid is close to bottom or not, sometimes need to be shifted up for same reason as upper eyelid
    alpha_l = np.linspace(0, np.pi, 700)
    rrl, ccl = EyeLidMask(pupil_xy_out[1], rposmax, rmax, iris_r, amx, lid_imgx.shape, False, alpha_l)
    max_rr_lower = rrl.max()
    if max_rr_lower < lid_imgx.shape[0] - 30:
        rrl, ccl = EyeLidMask(pupil_xy_out[1], rposmax, rmax, iris_r, amx, lid_imgx.shape, False, alpha_l)

    return rru, ccu, rrl, ccl


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
    Find the edge of the eyelid in the image.
    Similar to the process of finding edge of iris as in FindEdge.
    Returns:
        angleval, radiusval, rposmaxval
    """
    if radiusmin is None:
        radiusmin = int(pupil_rpos * 0.3)
    if radiusmax is None:
        if upper:
            radiusmax = int(pupil_rpos * 3)
        else:
            # Lower eyelid can be further away from pupil than upper eyelid and also have a larger radius
            radiusmax = int(pupil_rpos * 3.2)
            radiusmin = int(pupil_rpos * 1)
    
    # Set radius vector to iterate over
    radiusvec = np.round(np.linspace(pupil_radius//2, img.shape[0]*1.3, n_radiusjumps)).astype(int)
    
    # Set angle vector to iterate over
    anglevec = np.linspace(anglemin, anglemax, n_angles)

    # Initialize arrays to store values in grid-search approach
    angleval = np.zeros((anglevec.shape[0], radiusvec.shape[0]))
    radiusval = np.zeros((anglevec.shape[0], radiusvec.shape[0]))
    rposmaxval = np.zeros((anglevec.shape[0], radiusvec.shape[0]))
    maxblurval = np.zeros((anglevec.shape[0], radiusvec.shape[0]))

    # Set alpha vector for integration, depending on upper or lower eyelid and also jump of position
    if upper:
        rposjump = -rposjump
        alpha = np.linspace(np.pi, 2*np.pi, 1000)
    else:
        alpha = np.linspace(0, np.pi, 1000)
    
    # Set functions to get max and min position of radius
    if upper:
        get_max_pos = lambda r: r
        get_min_pos = lambda r: int(r + pupil_rpos - pupil_radius//2)
    else:
        get_max_pos = lambda r: img.shape[0] - r
        get_min_pos = lambda r: int(img.shape[0] - r - pupil_rpos + pupil_radius*1.2)

    # Iterate over radius and angle vectors
    for i, angle in enumerate(anglevec):
        
        for j, radius in enumerate(radiusvec):

            rposmax = get_max_pos(radius)
            rposmin = get_min_pos(radius)

            rposvec = np.arange(rposmin, rposmax + 1, rposjump)
            angleval[i, j] = angle
            radiusval[i, j] = radius
            drLIM = drLineIntegralMultiEyeLid(img, c, rposvec, radius, iris_radius, angle, upper, alpha)
            cgdrLIM = ConvolveGaussiandrLI(drLIM, filter_size=filter_size, sigma=sigma)
            arg_max_blur = np.argmax(cgdrLIM)
            rposmaxval[i, j] = rposvec[arg_max_blur]
            maxblurval[i, j] = cgdrLIM[arg_max_blur]
    
    maxblurval = maxblurval.flatten()
    rposmaxval = rposmaxval.flatten()
    angleval = angleval.flatten()
    radiusval = radiusval.flatten()
    maxbluridx = np.argmax(maxblurval)
    amx = int(angleval[maxbluridx])
    rmax = int(radiusval[maxbluridx])
    rposmax = int(rposmaxval[maxbluridx])

    # if lower eyelid, check if it is too close to pupil
    if not upper:
        top_indices = np.argsort(maxblurval)
        rr, cc = EyeLidMask(c, rposmax, rmax, iris_radius, amx, img.shape, upper, alpha)
        max_rr = rr.max()
        if max_rr - 50 < pupil_rpos + pupil_radius + 5:
            print("Under eyelid too closse to pupil, search for estimation further away:")
            for i in range(1, top_indices.shape[0]):
                top_blur = top_indices[-i]
                rposmax = int(rposmaxval[top_blur])
                amx = int(angleval[top_blur])
                rmax = int(radiusval[top_blur])
                rr, cc = EyeLidMask(c, rposmax, rmax, iris_radius, amx, img.shape, upper, alpha)
                max_rr = rr.max()
                if max_rr - 50 < pupil_rpos + pupil_radius + 5:
                    continue
                else:
                    break
    
    if plot:
        img_plt = img.copy()
        rr, cc = EyeLidMask(c, rposmax, rmax, iris_radius, amx, img.shape, upper, alpha)
        img_plt[rr, cc] = 0
        plt.imshow(img_plt, cmap='gray')
        plt.show()

    return amx, rmax, rposmax

def LineIntegralEyeLid(img: np.ndarray, c: int, r: int, radius: int, iris_radius: int, angle: float, upper: bool, alpha) -> float:
    """
    Same as LineIntegral, but for eyelid. Only difference is that the mask is different. Could just have one function for both, but hard with differerent return values
    """
    rr, cc = EyeLidMask(c, r, radius, iris_radius, angle, img.shape, upper, alpha)
    nan = False
    fsum = rr.shape[0]
    if fsum < (img.shape[0] + img.shape[1]) / 10:
        nan = True
        return np.nan, nan
    return (img[rr, cc]).sum() / fsum, nan

def drLineIntegralMultiEyeLid(img: np.ndarray, c: int, rvec: np.ndarray, radius: int, iris_radius: int, angle: float, upper: bool, alpha) -> np.ndarray:
    """
    Also similar to drLineIntegralMulti, but for eyelid. 
    """
    lint = np.zeros(rvec.shape[0])
    nan_in = False
    for i, r in enumerate(rvec):#:range(rmin, rmax + 2):
        lint[i], nan = LineIntegralEyeLid(img, c, r, radius, iris_radius, angle, upper, alpha)
        nan_in = nan_in or nan
    
    if nan_in:
        # ugly but got some weird error one time so now this is it.
        try:
            mask = np.isnan(lint)
            lint[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), lint[~mask])
        except:
            pass
    
    return np.diff(lint)



def EyeLidMask(c: int, r: int, radius: int, iris_radius: int, angle: float, img_shape: tuple, upper: bool, alpha) -> np.ndarray:
    """
    Returns indices of half ellipses meant to represent eyelid contours, used for line integrals
    with the Daugman integro-differential operator.
    """

    # parameters for ellipse
    theta = angle
    cx = c
    cy = r
    rx = int(iris_radius*1.5)
    ry = radius

    cc = (cx + rx*np.cos(alpha)*np.cos(theta) - ry*np.sin(alpha)*np.sin(theta))
    rr = (cy + rx*np.cos(alpha)*np.sin(theta) + ry*np.sin(alpha)*np.cos(theta))
    # not so simple, need all points on ellipse, then find the ones that are inside the image
    mask = (rr < img_shape[0]) & (rr > 0) & (cc < img_shape[1]) & (cc > 0)
    xfull = np.arange(cc[mask].min(), cc[mask].max()).astype(int)
    rr = np.round(interp1d(cc[mask], rr[mask], fill_value="extrapolate")(xfull)).astype(int)

    mask = (rr < img_shape[0]) & (rr > 0)
    return rr[mask], xfull[mask]


def FindPupil(
    img: np.ndarray,
    filter_size: int,
    rvec: np.ndarray,
    plot_print: bool = False
    ) -> Tuple[int, int, int, zip]:
    """
    Function for finding the pupil in an image.
    Utilizing canny edge detection and hough circle transform at specific thresholds and radii.

    Parameters
    ----------
    img : np.ndarray, image containing the eye
    filter_size : int, size of gaussian filter for preprocess blur
    rvec : np.ndarray, array of radii to search for in hough circle transform
    plot_print : bool, whether to plot the results

    Returns
    -------
    prmax : int, radius of pupil
    c : int, x coordinate of pupil
    r : int, y coordinate of pupil
    hres_left: zip of the hough circle transform results that remain potentially valid
    """

    img_use = cv2.GaussianBlur(img, (filter_size, filter_size), 0)
    
    edges = cv2.Canny(img_use, 40, 50)
    if plot_print:
        plt.imshow(edges, cmap="gray")
        plt.show()
    
    hough_results = skim.transform.hough_circle(edges, rvec)

    ridx, r, c = np.unravel_index(np.argmax(hough_results), hough_results.shape)
    prmax = rvec[ridx]

    # Checking if the pupil is too close to the edge of the image, then
    accums, cx, cy, radii = skim.transform.hough_circle_peaks(hough_results, rvec, total_num_peaks=20)
    if edges.shape[1] - (c + prmax) < 45 or edges.shape[0] - (r + prmax) < 45 or (c - prmax) < 45 or (r - prmax) < 45:
        count = 0
        for center_y, center_x, radius in zip(cy[1:], cx[1:], radii[1:]):
            count += 1
            r = center_y
            c = center_x
            prmax = radius
            if edges.shape[1] - (c + prmax) < 45 or edges.shape[0] - (r + prmax) < 45 or (c - prmax) < 45 or (r - prmax) < 45:
                continue
            else:
                break
    else:
        count = 1
    hres_left = zip(cy[count:], cx[count:], radii[count:])
    return prmax, r, c, hres_left


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
        ) -> Union[float, int]:
    """
    Function for finding the edge loss of an image.
    Can be used to find initial guess for iris radius for quicker search later on.
    Parameters
    ----------
    img : np.ndarray, image containing the eye
    rmin : int, minimum radius to search for
    rmax : int, maximum radius to search for
    filter_size : int, size of gaussian filter for preprocess blur
    sigma : float, sigma of gaussian filter for preprocess blur
    lateral : bool, whether to search for lateral edge only instead of full circle (often iris is not fully visible, or pupil for that matter)
    x0 : int, x coordinate of center of circle
    y0 : int, y coordinate of center of circle
    return_radius : bool, whether to return the radius or the edge loss
    Returns
    -------
    float edge loss of the image
    or
    int radius of the edge loss

    """
    jump = 4
    r_vec = np.arange(rmin, rmax + 1, jump)

    r_vec_current = r_vec.copy()

    rmin = r_vec_current[0]
    rmax = r_vec_current[-1]
    drLIM = drLineIntegralMulti(img, rmin, rmax, x0, y0, lateral=lateral, jump=jump)
    cgdrLIM = ConvolveGaussiandrLI(drLIM, filter_size=filter_size, sigma=sigma)
    arg_max_blur = np.argmax(cgdrLIM)
    if return_radius:
        return int(r_vec_current[arg_max_blur])
    return cgdrLIM[arg_max_blur]



def LineIntegral(img: np.ndarray, r: int, x0: int, y0: int, lateral: bool=False) -> float:
    """
    Calculates the line integral (sum since discrete) of a circle with radius r and center x0, y0.
    Normalized by the number of pixels in the circle.
    """
    circle, ryup, rydown, rxup, rxdown = circle_mask(r, lateral=lateral, xmax=img.shape[1], ymax=img.shape[0], x0r=x0, y0r=y0)
    return (img[y0 - ryup:y0 + rydown + 1, x0 - rxup:x0 + rxdown + 1][circle]).sum() / circle.sum()


def drLineIntegralMulti(img: np.ndarray, rmin: int, rmax: int, x0: int, y0: int, lateral: bool=False, jump: int=1) -> np.ndarray:
    """
    Stands for partial derivative (which is only the difference since the scale is discrete) 
    of line integral with regards to radius r. Multi since it calculates for multiple radii at once.
    Part of Daugman's integro-differential operator.
    """
    r_range = np.arange(rmin, rmax + 2, jump)
    lint = np.zeros(r_range.shape)
    for i, r in enumerate(r_range):#:range(rmin, rmax + 2):
        lint[i] = LineIntegral(img, int(r), x0, y0, lateral=lateral)
    return np.diff(lint)

def ConvolveGaussiandrLI(drLIM: np.ndarray, filter_size: int=3, sigma: float=1.0) -> np.ndarray:
    """
    Convolves the line integral with a gaussian filter.
    Last step in Daugman's algorithm for estimating iris and/or pupil radius.
    """
    gf = np.exp(-(np.arange(filter_size) - filter_size // 2)**2/(2*sigma**2)) / (np.sqrt(2 * np.pi) * sigma)
    return np.convolve(drLIM, gf, mode="same")

def circle_mask(r, xmax: int, ymax: int, x0r: int, y0r: int, lateral: bool=False) -> np.ndarray:
    """
    Creates a circle mask for a given radius and center coordinates.
    Homemade, and not that optimized, but works. Could easily replace with skimage.draw.circle in one line...
    Only used in LineIntegral function and never called directly.
    """
    circle = np.zeros((r + 1, r + 1)).astype(bool)
    circle_x, circle_y = np.meshgrid(np.arange(0, r + 1), np.arange(0, r + 1))
    circle_y = np.flip(circle_y, axis=0)
    dist = np.hstack([np.abs(circle_x ** 2 + circle_y ** 2 - r ** 2), np.ones(r + 1)[:, np.newaxis]*1e10])
    x0 = 0
    y0 = 0
    circle[y0, x0] = True
    incomplete_circle = True
    # Only need to search for the upper right quarter of the circle, because symmetry
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
    # for checking if the circle is out of bounds
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
        ) -> Tuple[Tuple[int, int], int]:
    """
    Basically performs grid search to find optimal centre coordinates and radius.
    Built upon Daugman's integro-differential operator.
    Returns the optimal centre coordinates and radius.
    """
    
    top_index = (y0, x0)
    n = search_radius*2 + 1
    max_blur = np.zeros((n, n))
    opt_r = np.zeros((n, n))
    index_ = np.zeros((n, n, 2))
    r_vec = np.arange(rmin, rmax + 1)

    # Grid search for the optimal radius and center coordinates
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