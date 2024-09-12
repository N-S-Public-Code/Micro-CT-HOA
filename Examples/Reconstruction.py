"""
Rconstruction of stitched projection
"""
import tomopy
import logging
from skimage import io
import tifffile

# logging
logging.basicConfig(level=logging.INFO)

# read normalized stitched projections 
print('read normalized stitched projections (tiff stack)...')
proj = io.imread('user/example/path_in')

# calculating rotation step
print('calculating rotation step...')
theta = tomopy.angles(proj.shape[0])

# reconstruction
proj = tomopy.minus_log(proj)

# calculating COR
print('calculating COR...')
rot_center = tomopy.find_center(proj, theta, init=1000, ind=0, tol=0.3)

recon = tomopy.recon(proj, theta, center=rot_center, algorithm='gridrec', sinogram_order=False)

recon = tomopy.circ_mask(recon, axis=0, ratio=0.95)

# write tomograms
print('saving reconstructions...')
tifffile.imwrite('user/ecamper/path_out', recon, imagej=True,)
