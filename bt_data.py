import numpy as np
import scipy.sparse as sp

def prepare_data():
    """
    Prepare the design matrix for pairwise comparisons and calculate the number of students.

    This function creates a sparse matrix representing pairwise comparisons between students.
    Each row in the matrix corresponds to a comparison, with a value of 1 for the winner and -1 for the loser.
    The function also calculates the total number of students involved in the comparisons.

    Args:
        None

    Returns:
        scipy.sparse.csr_matrix: Sparse matrix representing the pairwise comparisons.
        int: Total number of students.
    """
    comparisons = [[0,1,1],[0,2,2],[0,3,3],[0,4,0],[1,2,2],[1,3,3],[1,4,4],[2,3,2],[2,4,2],[3,4,3]]
    n_students = len(np.unique(np.array(comparisons)))
    n_comparisons = len(comparisons)
    print(f'Total students: {n_students}, Total comparisons: {n_comparisons}')

    X = sp.lil_matrix((n_comparisons, n_students))
    for idx, (i, j, k) in enumerate(comparisons):
        X[idx, k] = 1
        loser = i + j - k
        X[idx, loser] = -1

    return sp.csr_matrix(X), n_students
