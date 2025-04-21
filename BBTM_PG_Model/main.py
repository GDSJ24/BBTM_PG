from bt_data import prepare_data
from bt_model import run_model, summarize_results
import pandas as pd

def main():
    X, n_students = prepare_data()
    print("Design Matrix:")
    print(pd.DataFrame(X.toarray()))

    print("\nRunning the Gibbs sampling model...")
    samples = run_model(X, n_students)
    print("Sampling complete.")

    results = summarize_results(samples, n_students)
    print("\nRanking Results:")
    print(results)

if __name__ == "__main__":
    main()
