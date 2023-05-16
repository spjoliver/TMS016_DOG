import numpy as np
import matplotlib.pyplot as plt


"""
Converts a three channel image to greyscale using matplotlib
color recomendations.
"""
def rgbtogray(img):
    try: 
        _ = img.shape[2]
        return np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
    except IndexError:
        raise Exception("Image already grey.")
    return


"""
Transforms the iris into pseudo-polar coordinates.
Cannot utilize information about the eyelid position.
img : greyscale image
pup_center, pup_r : pupil circle center and radius
iris_center, iris_r : iris circle center and outer radius
theta_res, rho_res : how many sample points to be taken theta and radially.
method : method to take the sample points. The one I made is probably easiest 
to adjust to eyelid detection, but I have some other ones that take the non-
concentric pupil and iris into account that could be added.
plot : plots the sampling.
"""
def transform_iris(img, pup_center, pup_r, iris_center, iris_r, theta_res, rho_res,
                  method = "Daug", plot = False):
    if plot:
        fig, ax = plt.subplots()
        ax.imshow(img, cmap="gray", vmin=0., vmax=1.)

    def to_ind(coords):
        return np.array(np.round(coords), dtype="int32")
    
    tranf = np.zeros([rho_res, theta_res])
    
    if method == "Daug":        
        dtheta = 2*np.pi/theta_res
        thetas = np.arange(0., 2*np.pi, dtheta)
        assert(np.abs(thetas[-1] - 2.*np.pi) >1e-5)    
        rhos = np.linspace(0., 1., rho_res)
        for j, theta in enumerate(thetas):
            u = np.array([np.cos(theta), np.sin(theta)])
            pos_pup = pup_center + u*pup_r
            pos_limbus = iris_center + u*iris_r
            x = (1 - rhos)*pos_pup[0] + rhos*pos_limbus[0]
            y = (1 - rhos)*pos_pup[1] + rhos*pos_limbus[1]
            if plot:
                ax.plot(x, y, color="white", lw=0.5)
            #tranf[:, j] = img[np.flip(to_ind(y)), to_ind(x)]
            tranf[:, j] = img[to_ind(y), to_ind(x)]
        return tranf


"""
Transforms the iris into pseudo-polar coordinates.
This method now takes the cropped iris into account where 
the ignored pixels are set to white (1.).
When plotting the limbus is the bottom and the pupil is the upper part.
img : Normalized greyscale image
pup_center, pup_r : pupil circle center and radius
iris_center, iris_r : iris circle center and outer radius
theta_res, rho_res : how many sample points to be taken theta and radially.
method : method to take the sample points. The one I made is probably easiest 
to adjust to eyelid detection, but I have some other ones that take the non-
concentric pupil and iris into account that could be added.
plot : plots the sampling.
"""
def transform_copped_iris(img, pup_center, pup_r, iris_center, iris_r, theta_res, rho_res,
                  method = "Daug", plot = False):    
    assert(img.ndim == 2)
    if np.max(img) > 1.:
        img = img/255.
    if plot:
        fig, ax = plt.subplots()
        ax.imshow(img, cmap="gray", vmin=0., vmax=1.)

    def to_ind(coords):
        return np.array(np.round(coords), dtype="int32")
    
    def outside(img, pos):
        m, n = np.shape(img)
        coord = to_ind(pos)
        if coord[0] < 0 or coord[1] < 0:
            return True
        elif coord[1] >= m or coord[0] >= n:
            return True
        elif (img[coord[1]][coord[0]] - 1.)**2 < 1e-5:
            return True
        else:
            return False
    
    def find_pos_limbus(img, pos, center):
        v = pos - center
        hi = pos
        lo = center
        u = v/np.linalg.norm(v)
        dr = np.sqrt(2)*1.1 # This (should) ensure boundary point
        is_outside = True
        i = 0
        max_iter = 100
        while is_outside and i < max_iter:
            i += 1
            mid = (hi + lo)*0.5
            if outside(img, mid):
                hi = mid
            elif outside(img, mid + u*dr):
                return mid
            else:
                lo = mid
        
        
    tranf = np.zeros([rho_res, theta_res])
    
    if method == "Daug":        
        dtheta = 2*np.pi/theta_res
        thetas = np.arange(0., 2*np.pi, dtheta)
        #assert(np.abs(thetas[-1] - 2.*np.pi) >1e-5)    
        rhos = np.linspace(0., 1., rho_res)
        for j, theta in enumerate(thetas):
            u = np.array([np.cos(theta), np.sin(theta)])
            pos_pup = pup_center + u*pup_r
            
            # There is a high risk that the limbus point is outside of the cropped iris.
            pos_limbus = iris_center + u*iris_r
            if outside(img, pos_limbus):
                pos_limbus = find_pos_limbus(img, pos_limbus, iris_center)
            
            x = (1 - rhos)*pos_pup[0] + rhos*pos_limbus[0]
            y = (1 - rhos)*pos_pup[1] + rhos*pos_limbus[1]
            if plot:
                ax.plot(x, y, color="red", lw=0.5)
            #tranf[:, j] = img[np.flip(to_ind(y)), to_ind(x)]
            
            tranf[:, j] = img[to_ind(y), to_ind(x)]
        return tranf

