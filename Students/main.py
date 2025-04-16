from comp_simulator import simulate_pairwise_comparisons

def main():
    # 5 students with their means and stds
    mu_vector = [50.0, 60.0, 70.0, 55.0, 65.0]
    sigma_vector = [5.0, 5.0, 5.0, 5.0, 5.0]

    results = simulate_pairwise_comparisons(mu_vector, sigma_vector, num_trials=1)

    print("Pairwise Comparison Results in [i,j, winner]:")

    for i, j, winner in results:
        print([i,j,winner])

if __name__ == "__main__":
    main()
