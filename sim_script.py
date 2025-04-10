
# Imports
import numpy as np
import scipy.sparse as sp
from pypolyagamma import PyPolyaGamma
import multiprocessing as mp
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import kendalltau, spearmanr, norm
from collections import defaultdict
import pandas as pd
from itertools import combinations
import random

def main():
    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
        print(f"You entered: {user_input}")
    else:
        print("No input provided. Please enter something as a command-line argument.")

if __name__ == "__main__":
    main()

