import numpy as np
from itertools import product
from multiprocessing import Pool

# Helper function for Gray Code flip
def gray_flip(config, i):
    config[i] *= -1
    return config

# Gray code generator based on Alg. 5.2
def gray_code_generator(N):
    # Start with the initial configuration of all spins +1
    config = np.ones(N, dtype=int)
    yield config.copy()

    # Flip one bit at a time using Gray code pattern
    for i in range(1, 2**N):  # There are 2^N configurations
        # Determine which bit to flip (Gray code logic)
        bit_to_flip = i ^ (i >> 1)
        flip_index = (bit_to_flip & -bit_to_flip).bit_length() - 1
        config[flip_index] *= -1
        yield config.copy()


# Function to calculate energy for a given configuration (Ising Model)
def calculate_energy(config, L, J=1, periodic=True):
    lattice = config.reshape(L, L)
    energy = 0
    for i in range(L):
        for j in range(L):
            S = lattice[i, j]
            # Neighbors (with periodic boundary conditions)
            neighbors = 0
            if periodic:
                neighbors += lattice[(i+1) % L, j] + lattice[i, (j+1) % L] + lattice[(i-1) % L, j] + lattice[i, (j-1) % L]
            else:
                if i + 1 < L: neighbors += lattice[i + 1, j]
                if j + 1 < L: neighbors += lattice[i, j + 1]
                if i - 1 >= 0: neighbors += lattice[i - 1, j]
                if j - 1 >= 0: neighbors += lattice[i, j - 1]

            energy -= J * S * neighbors

    return energy / 2.0  # Each bond is counted twice, so divide energy by 2

# Process configuration in parallel (moved outside of calculate_density_of_states)
def process_configuration(config_L_periodic):
    config, L, periodic = config_L_periodic
    energy = calculate_energy(config, L, periodic=periodic)
    energy = round(energy, 5)  # Rounding to avoid floating-point precision issues
    return energy

# Function to calculate density of states using parallelization
def calculate_density_of_states(L, periodic=True):
    N = L * L
    energy_counts = {}

    # Prepare configurations and pass L and periodic as arguments to the worker
    configs = [(config, L, periodic) for config in gray_code_generator(N)]

    # Use multiprocessing pool to parallelize the energy calculation
    pool = Pool()
    results = pool.map(process_configuration, configs)
    pool.close()
    pool.join()

    # Calculate density of states
    for energy in results:
        if energy in energy_counts:
            energy_counts[energy] += 1
        else:
            energy_counts[energy] = 1

    return energy_counts

# Function to print all small N configurations (like Table 5.1)
def print_small_configurations(N):
    print(f"All configurations for N = {N}:")
    for i, config in enumerate(gray_code_generator(N)):
        print(f"{i+1}: {config}")

# Main function
def main():
    # Task 1: Print small configurations for N = 4 (as in Table 5.1 example)
    N_small = 4
    print_small_configurations(N_small)

    # Task 2: Generate the density of states for 2x2, 4x4, 6x6 lattices
    #lattice_sizes = [2, 4]
    lattice_sizes = [2, 4, 6]

    for L in lattice_sizes:
        print(f"\nCalculating density of states for L = {L}x{L} with periodic boundaries:")
        dos_periodic = calculate_density_of_states(L, periodic=True)
        print(f"Density of states (periodic boundaries): {dos_periodic}")

        print(f"\nCalculating density of states for L = {L}x{L} without periodic boundaries:")
        dos_non_periodic = calculate_density_of_states(L, periodic=False)
        print(f"Density of states (non-periodic boundaries): {dos_non_periodic}")

if __name__ == "__main__":
    main()