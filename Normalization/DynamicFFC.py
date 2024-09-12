"""
Dynamic intensity normalization using eigen flat fields.

A Python implementation of the normalization approach proposed in 
"Dynamic intensity normalization using eigen flat fields in X-ray imaging" Vincent Van Nieuwenhove et al, Opt. Express 23.

Developed by Natanel Shubayev.  
"""
import os
import numpy as np
import cv2
from skimage.restoration import denoise_bilateral  # Importing the bilateral filter from scikit-image
from scipy.optimize import minimize
"""
import cv2
import bm3d

import numpy as np
from scipy.linalg import svd
from skimage import img_as_float
from skimage.io import imread, imsave
from skimage.util import dtype
from skimage.exposure import rescale_intensity
from skimage import io
"""

# Directories
readDIR = r'E:\Users\nshubayev\Desktop\107_220422_1213_Disso_8p2_a_ref_Z40_Y4805_24500eV_0p72um_250ms\binned_data\\'
outDIRDFFC = r'E:\Users\nshubayev\Desktop\107_220422_1213_Disso_8p2_a_ref_Z40_Y4805_24500eV_0p72um_250ms\binned_data\DFFC\\'
outDIRFFC = r'E:\Users\nshubayev\Desktop\107_220422_1213_Disso_8p2_a_ref_Z40_Y4805_24500eV_0p72um_250ms\binned_data\conventional\\'

# Create output directories if they don't exist
os.makedirs(outDIRDFFC, exist_ok=True)
os.makedirs(outDIRFFC, exist_ok=True)

# Utility functions
# Parallel analysis function 
def parallelAnalysis(flatFields, repetitions):
    # Calculate standard deviation across columns
    stdEFF = np.std(flatFields, axis=1, ddof=0)

    # Initialize variables
    keepTrack = np.zeros((flatFields.shape[1], repetitions))
    stdMatrix = np.tile(stdEFF[:, np.newaxis], (1, flatFields.shape[1]))

    for ii in range(repetitions):
        print(f"Parallel Analysis: repetition {ii+1}")
        sample = stdMatrix * np.random.randn(*flatFields.shape)
        _, D1 = np.linalg.eig(np.cov(sample))
        D1 = np.real(np.diag(D1))
        keepTrack[:, ii] = D1

    mean_flat_fields_EFF = np.mean(flatFields, axis=1)
    F = flatFields - mean_flat_fields_EFF[:, np.newaxis]
    V1, D1 = np.linalg.eig(np.cov(F))
    D1 = np.real(np.diag(D1))

    selection = np.zeros(flatFields.shape[1], dtype=int)
    selection[D1 > (np.mean(keepTrack, axis=1) + 2 * np.std(keepTrack, axis=1, ddof=0))] = 1
    numberPC = np.sum(selection)

    return V1, D1, numberPC

def condTVmean(projections, meanFF, FF, DF, x, DS):
    # Downsample images
    projections = np.array(projections)
    projections = np.resize(projections, (int(projections.shape[0]/DS), int(projections.shape[1]/DS)))
    
    meanFF = np.array(meanFF)
    meanFF = np.resize(meanFF, (int(meanFF.shape[0]/DS), int(meanFF.shape[1]/DS)))
    
    FF2 = np.zeros((meanFF.shape[0], meanFF.shape[1], FF.shape[2]))
    for ii in range(FF.shape[2]):
        FF2[:,:,ii] = np.resize(FF[:,:,ii], (int(FF[:,:,ii].shape[0]/DS), int(FF[:,:,ii].shape[1]/DS)))
    FF = FF2
    
    DF = np.resize(DF, (int(DF.shape[0]/DS), int(DF.shape[1]/DS)))
    
    # Optimize coefficients
    xNew = minimize(fun, x, args=(projections, meanFF, FF, DF), method='BFGS').x
    
    return xNew

def fun(x, projections, meanFF, FF, DF):
    # Objective function
    FF_eff = np.zeros((FF.shape[0], FF.shape[1]))
    for ii in range(FF.shape[2]):
        FF_eff += x[ii] * FF[:,:,ii]
    
    meanFF_eff = meanFF + FF_eff
    logCorProj = (projections - DF) / meanFF_eff * np.mean(meanFF_eff)
    
    Gx, Gy = np.gradient(logCorProj)
    mag = np.sqrt(Gx**2 + Gy**2)
    cost = np.sum(mag)
    
    return cost

# Parameters
prefixProj = 'proj'
outPrefixDFFC = 'DFFC'
outPrefixFFC = 'FFC'
prefixFlat = 'FF'
prefixDark = 'dark'
fileFormat = '.tif'
nrDark = 1
firstDark = 1
nrWhitePrior = 20
firstWhitePrior = 0
nrWhitePost = 20
firstWhitePost = 20
nrProj = 2400
firstProj = 0
downsample = 2
nrPArepetions = 10
scaleOutputImages = [0, 2]

"""
Load dark and flat fields
"""
print('Load dark and flat fields:')
tmp = cv2.imread(os.path.join(readDIR, prefixProj + "{0:0=4d}".format(firstProj) + fileFormat), cv2.IMREAD_UNCHANGED)
dims = tmp.shape

# Load dark fields
print('Load dark fields ...')
dark = np.zeros((dims[0], dims[1], nrDark))
for ii in range(firstDark, firstDark + nrDark):
    dark[:, :, ii - firstDark] = cv2.imread(os.path.join(readDIR, prefixDark + "{0:0=4d}".format(ii) + fileFormat), cv2.IMREAD_UNCHANGED)
meanDarkfield = np.mean(dark, axis=2)

# Load flat fields
whiteVec = np.zeros((dims[0] * dims[1], nrWhitePrior + nrWhitePost))

print('Load flate fields ...')
k = 0
for ii in range(firstWhitePrior, firstWhitePrior + nrWhitePrior):
    k += 1
    tmp = cv2.imread(os.path.join(readDIR, prefixFlat + "{0:0=4d}".format(ii) + fileFormat), cv2.IMREAD_UNCHANGED) - meanDarkfield
    whiteVec[:, k - 1] = tmp.flatten() - meanDarkfield.flatten()

for ii in range(firstWhitePost, firstWhitePost + nrWhitePost):
    k += 1
    tmp = cv2.imread(os.path.join(readDIR, prefixFlat + "{0:0=4d}".format(ii) + fileFormat), cv2.IMREAD_UNCHANGED) - meanDarkfield
    whiteVec[:, k - 1] = tmp.flatten() - meanDarkfield.flatten()
mn = np.mean(whiteVec, axis=1)

# Subtract mean flat field
Data = whiteVec - np.tile(mn[:, np.newaxis], (1, whiteVec.shape[1]))
N = len(whiteVec) 
del whiteVec, dark

"""
calculate Eigen Flat fields
"""
# Parallel Analysis
print('Parallel Analysis:')
V1, D1, nrEigenflatfields = parallelAnalysis(Data, nrPArepetions)
print(f"{nrEigenflatfields} eigen flat fields selected.")

# Calculation of eigen flat fields
eig0 = np.reshape(mn, dims)
EigenFlatfields = np.zeros((dims[0], dims[1], nrEigenflatfields + 1))
EigenFlatfields[:, :, 0] = eig0

for ii in range(nrEigenflatfields):
    EigenFlatfields[:, :, ii + 1] = np.reshape(Data @ V1[:, N - ii - 1], dims)

# Clearing Data (assuming it's no longer needed)
Data = None

""""
Filter Eigen flat fields using bilateral filtering
"""
print('Filter eigen flat fields ...')
filteredEigenFlatfields = np.zeros_like(EigenFlatfields)
for ii in range(1, nrEigenflatfields + 1):
    print('Filter eigen flat field ' + str(ii))
    tmp = EigenFlatfields[:, :, ii]
    tmp = (tmp - np.min(tmp)) / (np.max(tmp) - np.min(tmp))
    filteredEigenFlatfields[:, :, ii] = denoise_bilateral(tmp, sigma_color=0.05, sigma_spatial=15) * (np.max(tmp) - np.min(tmp)) + np.min(tmp)

"""
Estimate abundance of weights in projections
"""
meanVector = np.zeros(nrProj)
for ii in range(nrProj):
    print('Conventional FFC: ' + str(ii + 1) + '/' + str(nrProj) + '...')
    projection = cv2.imread(os.path.join(readDIR, prefixProj + "{0:0=4d}".format(firstProj + ii) + fileFormat), cv2.IMREAD_UNCHANGED)
    
    tmp = (projection - meanDarkfield) / EigenFlatfields[:, :, 0]
    meanVector[ii] = np.mean(tmp)

    tmp[tmp < 0] = 0
    tmp = -np.log(tmp)
    tmp[np.isinf(tmp)] = 1e5
    tmp = ((tmp - scaleOutputImages[0]) / (scaleOutputImages[1] - scaleOutputImages[0])) * (2**16 - 1)
    tmp = np.uint16(tmp)
    cv2.imwrite(os.path.join(outDIRFFC, outPrefixFFC + "{0:0=4d}".format(firstProj + ii) + fileFormat), tmp)

xArray = np.zeros((nrEigenflatfields, nrProj))
for ii in range(nrProj):
    print('Estimation projection ' + str(ii + 1) + '/' + str(nrProj) + '...')
    projection = cv2.imread(os.path.join(readDIR, prefixProj + "{0:0=4d}".format(firstProj + ii) + fileFormat), cv2.IMREAD_UNCHANGED)

    x = np.zeros(nrEigenflatfields)
    for j in range(nrEigenflatfields):
        FFeff = x[j] * filteredEigenFlatfields[:, :, j + 1]
        x[j] = np.mean(projection - meanDarkfield) / (EigenFlatfields[:, :, 0] + FFeff)

    xArray[:, ii] = x

    # Dynamic flat field correction
    FFeff = np.zeros_like(meanDarkfield)
    for j in range(nrEigenflatfields):
        FFeff += x[j] * filteredEigenFlatfields[:, :, j + 1]

    tmp = (projection - meanDarkfield) / (EigenFlatfields[:, :, 0] + FFeff)
    tmp = tmp / np.mean(tmp) * meanVector[ii]
    tmp[tmp < 0] = 0
    tmp = -np.log(tmp)
    tmp[np.isinf(tmp)] = 1e5
    tmp = ((tmp - scaleOutputImages[0]) / (scaleOutputImages[1] - scaleOutputImages[0])) * (2**16 - 1)
    tmp = np.uint16(tmp)
    cv2.imwrite(os.path.join(outDIRDFFC, outPrefixDFFC + "{0:0=4d}".format(firstProj + ii) + fileFormat), tmp)

np.save(os.path.join(outDIRDFFC, 'parameters.npy'), xArray)
