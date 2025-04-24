import numpy as np
import pandas as pd
import scipy.sparse

def load_and_preprocess_data(comparisons):
    """
    Load and preprocess comparison data, compute number of students and comparisons.
    
    Args:
        comparisons (list): List of [student_i, student_j, winner] comparisons.
    
    Returns:
        tuple: (n_students, n_comparisons, comparisons_array)
    """
    comparisons_array = np.array(comparisons)
    n_students = len(np.unique(comparisons_array))
    n_comparisons = len(comparisons)
    print(f'Total students are {n_students},\nTotal comparisons are {n_comparisons}')
    return n_students, n_comparisons, comparisons_array

def create_design_matrix(comparisons, n_students, n_comparisons):
    """
    Create the design matrix X for the model.
    
    Args:
        comparisons (list): List of [student_i, student_j, winner] comparisons.
        n_students (int): Number of unique students.
        n_comparisons (int): Number of comparisons.
    
    Returns:
        scipy.sparse.csr_matrix: Design matrix X.
    """
    X = scipy.sparse.lil_matrix((n_comparisons, n_students))
    for idx, (i, j, k) in enumerate(comparisons):
        X[idx, k] = 1
        loser = i + j - k
        X[idx, loser] = -1
    X = X.tocsr()
    print("Design Matrix:\n", pd.DataFrame(X.toarray()))
    return X