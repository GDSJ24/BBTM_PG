from comp_simulator import simulate_pairwise_comparisons

def main():
    mu_vector = [50.0, 60.0, 70.0, 55.0, 65.0]
    sigma_vector = [5.0, 5.0, 5.0, 5.0, 5.0]

    results = simulate_pairwise_comparisons(mu_vector, sigma_vector, num_trials=1)

    print("Sample Values for each comparison [sample_i, sample_j]:")
    for _, _, _, sample_i, sample_j in results:
        print(f"[{sample_i:.3f}, {sample_j:.3f}]")

    print("\nPairwise comparison results in [i, j, winner]:")
    for i, j, winner, _, _ in results:
        print([i, j, winner])
        

if __name__ == "__main__":
    main()
