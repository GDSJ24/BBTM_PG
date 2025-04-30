import numpy as np
import random
from typing import List, Tuple
import scipy.sparse as sp
from itertools import combinations

class ComparisonGenerator:
    def __init__(self, n_students: int, mu_vector: List[float], sigma_vector: List[float], seed: int = 2013):
        """
        Initialize with student parameters
        Args:
            n_students: Number of students to compare
            mu_vector: List of mean skill levels for students
            sigma_vector: List of standard deviations for students
            seed: Random seed (default: 2013)
        """
        # Validate input lengths
        assert len(mu_vector) == n_students and len(sigma_vector) == n_students, \
            f"mu_vector and sigma_vector must each have a length of {n_students}"
        
        # Set seed for random module and NumPy
        random.seed(seed)
        np.random.seed(seed)
        
        self.n_students = n_students
        self.mus = np.array(mu_vector)
        self.sigmas = np.array(sigma_vector)

    def generate_winner(self, i: int, j: int) -> int:
        """Generate winner using normal distributions -- indices in the list"""
        sample_i = np.random.normal(self.mus[i], self.sigmas[i])
        sample_j = np.random.normal(self.mus[j], self.sigmas[j])
        return i if sample_i > sample_j else j
    
    def generate_winner_01(self, i: int, j: int) -> int:
        """Generate winner using normal distributions -- indices in the list"""
        sample_i = np.random.normal(self.mus[i], self.sigmas[i])
        sample_j = np.random.normal(self.mus[j], self.sigmas[j])
        return 1 if sample_i > sample_j else 0

    def generate_comparisons(self, n_comparisons: int, method=None, args=None) -> List[List[int]]:
        """
        Generate comparisons in [i, j, winner] format
        Args:
            n_comparisons: Number of comparisons to generate
        Returns:
            comparisons: List of [i, j, winner] for each trial
        """
        comparisons = []
        for _ in range(n_comparisons):
            if method is None:
                method = random.sample
                args = (range(self.n_students), 2)
            i, j = method(*args)
            winner = self.generate_winner(i, j)
            comparisons.append([i, j, winner])
        return comparisons

    def get_student_parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (mus, sigmas) for all students"""
        return self.mus.copy(), self.sigmas.copy()

# Design matrix
def build_design_matrix(comparisons: List[List[int]], n_students: int) -> sp.csr_matrix:
    """Design matrix for comparisons"""
    n = len(comparisons)
    X = sp.lil_matrix((n, n_students))
    for idx, (i, j, k, *rest) in enumerate(comparisons):  # Ignore extra fields
        X[idx, k] = 1
        loser = i if k == j else j
        X[idx, loser] = -1
    return X.tocsr()

def get_student_pairs(n_students: int) -> List[Tuple[int, int]]:
    """Student pairs for comparisons"""
    return [(i, j) for i in range(n_students) for j in range(i+1, n_students)]