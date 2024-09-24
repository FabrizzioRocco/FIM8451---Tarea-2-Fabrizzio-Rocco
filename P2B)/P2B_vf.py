import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from multiprocessing import Pool

# Function to calculate energy and magnetization
def calculate_energy_magnetization(lattice, J=1):
    L = lattice.shape[0]
    energy = 0
    magnetization = np.sum(lattice)
    
    for i in range(L):
        for j in range(L):
            S = lattice[i, j]
            neighbors = lattice[(i+1) % L, j] + lattice[i, (j+1) % L] + lattice[(i-1) % L, j] + lattice[i, (j-1) % L]
            energy -= J * S * neighbors
    
    return energy / 2.0, magnetization  # Each bond is counted twice, so divide energy by 2

# Generate configurations on-the-fly using a generator
def gray_code_generator(L):
    N = L * L
    for config in product([-1, 1], repeat=N):
        yield np.array(config).reshape(L, L)

# Parallel processing version for calculating energy and magnetization
def calculate_energy_magnetization_parallel(config):
    return calculate_energy_magnetization(config)

def compute_histogram_parallel(L, chunk_size=1000):
    energy_magnetization_counts = {}

    def update_histogram(results):
        for energy, magnetization in results:
            key = (energy, magnetization)
            if key in energy_magnetization_counts:
                energy_magnetization_counts[key] += 1
            else:
                energy_magnetization_counts[key] = 1

    pool = Pool()

    config_generator = gray_code_generator(L)

    while True:
        configs_chunk = list(next(config_generator, None) for _ in range(chunk_size))
        configs_chunk = [config for config in configs_chunk if config is not None]
        if not configs_chunk:
            break
        results = pool.map(calculate_energy_magnetization_parallel, configs_chunk)
        update_histogram(results)

    pool.close()
    pool.join()

    return energy_magnetization_counts

def compute_probability_distribution(E_M_histogram, T):
    Z = 0  # Partition function
    probability_distribution = {}

    for (E, M), count in E_M_histogram.items():
        boltzmann_weight = count * np.exp(-E / T)
        Z += boltzmann_weight
        if M in probability_distribution:
            probability_distribution[M] += boltzmann_weight
        else:
            probability_distribution[M] = boltzmann_weight

    for M in probability_distribution:
        probability_distribution[M] /= Z

    return probability_distribution

# Compute Binder cumulant B(T)
def compute_binder_cumulant(probability_distribution):
    m2 = sum([M**2 * prob for M, prob in probability_distribution.items()])
    m4 = sum([M**4 * prob for M, prob in probability_distribution.items()])
    return 1 - (m4 / (3 * m2**2))

def plot_probability_distribution(E_M_histogram, lattice_size, T):
    pi_M = compute_probability_distribution(E_M_histogram, T)
    
    M_values = list(pi_M.keys())
    probabilities = list(pi_M.values())

    plt.plot(M_values, probabilities, label=f'T = {T}')
    plt.xlabel('Total Magnetization (M)')
    plt.ylabel('Probability Distribution π_M')
    plt.title(f'Probability Distribution π_M vs M for Lattice {lattice_size}x{lattice_size}')
    plt.legend()

def main():
    lattice_sizes = [2, 4, 6]
    #lattice_sizes = [2, 4]
    temperature_range = np.arange(0.5, 5.0, 0.25)
    binders = {L: [] for L in lattice_sizes}
    
    for L in lattice_sizes:
        print(f"\nLattice size: {L}x{L}")
        E_M_histogram = compute_histogram_parallel(L)
        
        # Compute Binder cumulants and Probability Distributions
        for T in temperature_range:
            pi_m = compute_probability_distribution(E_M_histogram, T)
            binder_cumulant = compute_binder_cumulant(pi_m)
            binders[L].append(binder_cumulant)
            print(f"T = {T:.2f}, Binder Cumulant B(T) = {binder_cumulant:.4f}")

        # Plot Probability Distribution π_M vs M for T = 2.5 and T = 5.0 for each lattice size
        plt.figure(figsize=(10, 6))
        plot_probability_distribution(E_M_histogram, L, T=2.5)
        plot_probability_distribution(E_M_histogram, L, T=5.0)
        plt.grid(True)
        plt.show()

    # Plot Binder cumulant B(T) for all lattice sizes in one single plot
    plt.figure(figsize=(10, 6))
    
    for L in lattice_sizes:
        plt.plot(temperature_range, binders[L], label=f'L = {L}x{L}')
    
    # Add vertical line at Tc = 2.27
    plt.axvline(x=2.27, color='red', linestyle='--', label='$T_c = 2.27$')
    
    plt.xlabel('Temperature (T)')
    plt.ylabel('Binder Cumulant B(T)')
    plt.title('Binder Cumulant B(T) for Different Lattice Sizes')
    plt.legend()
    plt.grid(True)
    
    plt.show()

if __name__ == "__main__":
    main()
