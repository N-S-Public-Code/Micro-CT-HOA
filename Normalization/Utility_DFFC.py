import numpy as np

import numpy as np
from scipy.linalg import eigh

def parallel_analysis(flat_fields, repetitions):
    """
    Selection of the number of components for PCA using Parallel Analysis.
    Each row in flat_fields is a single observation.

    Parameters:
    flat_fields (np.ndarray): A 2D array where rows are observations.
    repetitions (int): The number of repetitions for parallel analysis.

    Returns:
    V1 (np.ndarray): Eigenvectors of the covariance matrix of the data.
    D1 (np.ndarray): Eigenvalues of the covariance matrix of the data.
    number_pc (int): The number of principal components selected.
    """
    # Compute the standard deviation of each row
    std_eff = np.std(flat_fields, axis=1, ddof=0)
    
    # Initialize matrices
    keep_track = np.zeros((flat_fields.shape[1], repetitions))
    std_matrix = np.tile(std_eff[:, np.newaxis], (1, flat_fields.shape[1]))
    
    # Perform parallel analysis
    for ii in range(repetitions):
        print(f'Parallel Analysis: repetition {ii + 1}')
        sample = std_matrix * np.random.randn(*flat_fields.shape)
        cov_matrix = np.cov(sample, rowvar=False)
        _, D1 = eigh(cov_matrix)
        keep_track[:, ii] = np.flipud(D1)
    
    # Mean and centered data
    mean_flat_fields_eff = np.mean(flat_fields, axis=1)
    F = flat_fields - mean_flat_fields_eff[:, np.newaxis]
    
    # Eigen decomposition
    cov_matrix = np.cov(F, rowvar=False)
    V1, D1 = eigh(cov_matrix)
    D1 = np.flipud(D1)
    
    # Selection criteria
    selection = (D1 > (np.mean(keep_track, axis=1) + 2 * np.std(keep_track, axis=1)))
    number_pc = np.sum(selection)
    
    return V1, D1, number_pc

import numpy as np

def calculate_eigenFF(Data, mn, dims, repetitions, N):
    """
    Calculate eigen flat fields using Parallel Analysis.
    
    Parameters:
    Data (np.ndarray): Input data matrix.
    mn (np.ndarray): Mean vector.
    dims (tuple): Dimensions to reshape arrays.
    repetitions (int): Number of repetitions for parallel analysis.
    N (int): Number of components for the reshaping.
    
    Returns:
    EigenFlatfields (np.ndarray): Array of eigen flat fields.
    """
    # Parallel Analysis
    print('Parallel Analysis:')
    V1, _, nr_eigen_flat_fields = parallel_analysis(Data, repetitions)
    print(f'{nr_eigen_flat_fields} eigen flat fields selected.')
    
    # Calculation of eigen flat fields
    eig0 = np.reshape(mn, dims)
    EigenFlatfields = np.zeros((dims[0], dims[1], nr_eigen_flat_fields + 1))
    EigenFlatfields[:, :, 0] = eig0
    
    for ii in range(nr_eigen_flat_fields):
        EigenFlatfields[:, :, ii + 1] = np.reshape(Data @ V1[:, N - ii - 1], dims)
    
    # Clear Data (not strictly necessary in Python, but included for completeness)
    del Data
    
    return EigenFlatfields

def Filter_EigenFF():
    return

def condTVmean():
    return

def wight_abundance_estimator():
    return