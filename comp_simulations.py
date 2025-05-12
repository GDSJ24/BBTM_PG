import numpy as np
import random
from itertools import combinations

def generate_normal_comparisons(n_students, comparisons_per_student, theta, seed=42):
    """
    Generate unique comparisons for n_students with fixed theta_i, each with exactly comparisons_per_student.
    Winner is determined by higher theta_i (no noise).
    """
    np.random.seed(seed)
    random.seed(seed)
    
    target_counts = np.full(n_students, comparisons_per_student)
    student_counts = defaultdict(int)
    comparisons = []
    used_pairs = set()
    
    total_involvements = n_students * comparisons_per_student
    target_pairs = total_involvements // 2
    if total_involvements % 2 != 0:
        print(f"Note: Total involvements ({total_involvements}) is odd; aiming for {target_pairs} pairs")
    
    # Generate all possible unique pairs
    all_pairs = list(combinations(range(n_students), 2))  # e.g., [(0, 1), (0, 2), ...]
    random.shuffle(all_pairs)
    
    # Select pairs to meet target counts
    for s1, s2 in all_pairs:
        if len(comparisons) >= target_pairs:
            break
        if student_counts[s1] >= target_counts[s1] or student_counts[s2] >= target_counts[s2]:
            continue
        pair = (s1, s2)
        if pair in used_pairs:
            continue
        
        winner = s1 if theta[s1] > theta[s2] else s2
        comparisons.append([s1, s2, winner])
        used_pairs.add(pair)
        student_counts[s1] += 1
        student_counts[s2] += 1
    
    # Verify
    for i in range(n_students):
        if student_counts[i] != target_counts[i]:
            print(f"Warning: Student {i} has {student_counts[i]} comparisons, expected {target_counts[i]}")
    if len(comparisons) != target_pairs:
        print(f"Warning: Generated {len(comparisons)} comparisons, expected {target_pairs}")
    
    return comparisons

# Scenarios with multiple sets
scenarios = [
    (5, [2, 3, 4]),
    (10, [5, 7, 8, 9]),
    (20, [5, 10, 15, 19]),
    (30, [10, 20, 25, 29]),
    (40, [10, 20, 30, 39])
]

for scenario_idx, (n_students, comp_list) in enumerate(scenarios, 1):
    print(f"\nScenario {scenario_idx}: {n_students} students")
    
    # Generate fixed theta_i for this scenario
    seed_theta = 42 + scenario_idx * 100
    np.random.seed(seed_theta)
    theta = np.round(np.random.normal(50, 15, n_students)).astype(int)
    theta = np.clip(theta, 0, 100)
    
    # Sort students by theta
    sorted_indices = np.argsort(-theta)
    sorted_theta = theta[sorted_indices]
    ranks = np.arange(n_students) + 1
    
    # Print fixed marks and ranks once
    print("\nOriginal Marks and Ranks (based on theta_i, fixed across sets):")
    print("Student | Mark (theta_i) | Rank")
    print("-" * 40)
    for student, mark, rank in zip(sorted_indices, sorted_theta, ranks):
        print(f"{student:6d} | {mark:12d} | {rank:4d}")
    print("Theta (unsorted):", theta)
    
    # Generate comparisons for each set
    for set_idx, comp_per_student in enumerate(comp_list):
        seed = 42 + scenario_idx * 10 + set_idx
        print(f"\n  Set {set_idx + 1}: {comp_per_student} comparisons per student")
        comparisons = generate_normal_comparisons(n_students, comp_per_student, theta, seed=seed)
        expected_comps = (n_students * comp_per_student) // 2
        print(f"    Total comparisons: {len(comparisons)} (expected ~{expected_comps})")
        print("    First 5 comparisons:", comparisons[:5] if len(comparisons) >= 5 else comparisons)
        
        
        filename = f"comparisons_scenario_{scenario_idx}_set_{set_idx + 1}.txt"
        with open(filename, 'w') as f:
            formatted_comparisons = ", ".join([f"[{i}, {j}, {w}]" for i, j, w in comparisons])
            f.write(formatted_comparisons)
        print(f"    Saved to {filename}")