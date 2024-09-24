import numpy as np
import matplotlib.pyplot as plt

def initialize_lattice(L):
    return np.random.choice([-1, 1], size=(L, L))

def wolff_cluster_step(lattice, T, J=1):
    L = lattice.shape[0]
    visited = np.zeros_like(lattice, dtype=bool)
    i, j = np.random.randint(0, L, size=2)
    cluster_spin = lattice[i, j]
    stack = [(i, j)]
    visited[i, j] = True
    cluster_size = 0
    
    while stack:
        x, y = stack.pop()
        cluster_size += 1
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = (x + dx) % L, (y + dy) % L
            if not visited[nx, ny] and lattice[nx, ny] == cluster_spin:
                if np.random.rand() < 1 - np.exp(-2 * J / T):
                    visited[nx, ny] = True
                    stack.append((nx, ny))
                    lattice[nx, ny] = -lattice[nx, ny]
    
    lattice[i, j] = -lattice[i, j]

def compute_energy(lattice, J=1):
    L = lattice.shape[0]
    energy = 0
    for i in range(L):
        for j in range(L):
            S = lattice[i, j]
            neighbors = lattice[(i+1) % L, j] + lattice[i, (j+1) % L] + lattice[(i-1) % L, j] + lattice[i, (j-1) % L]
            energy -= J * S * neighbors
    return energy / 2.0  # Each pair counted twice

def simulate_wolff(L, T, steps, J=1):
    lattice = initialize_lattice(L)
    energy_list = []
    magnetization_list = []
    
    for step in range(steps):
        wolff_cluster_step(lattice, T, J)
        if step >= steps / 2:  # Consider only the second half of the steps for measurement
            energy = compute_energy(lattice, J)
            energy_list.append(energy)
            magnetization_list.append(np.abs(np.sum(lattice)))
    
    mean_energy = np.mean(energy_list)
    mean_magnetization = np.mean(magnetization_list) / (L * L)
    
    # Specific heat capacity (Cv) calculation
    mean_energy_squared = np.mean(np.array(energy_list) ** 2)
    specific_heat = (mean_energy_squared - mean_energy ** 2) / (T ** 2 * L * L)
    
    return mean_energy, mean_magnetization, specific_heat, magnetization_list

def compute_binder_cumulant(magnetizations, L):
    M2 = np.mean(np.array(magnetizations) ** 2)
    M4 = np.mean(np.array(magnetizations) ** 4)
    return 1 - M4 / (3 * M2 ** 2)

def main():
    temperatures_6x6 = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    temperature_range = np.arange(0.5, 5.25, 0.25)
    lattice_sizes = [6, 16, 32, 64]
    steps = 1000

    # Specific printout for 6x6 lattice
    L = 6
    print(f"\nLattice size {L}x{L}:")
    print(f"{'Temperature':>12} {'Energy':>12} {'Specific Heat':>15}")
    for T in temperatures_6x6:
        energy, magnetization, specific_heat, _ = simulate_wolff(L, T, steps)
        print(f"{T:12.1f} {energy / (L * L):12.4f} {specific_heat:15.4f}")

    # Collect data for plotting for other lattice sizes
    binder_cumulants = {L: [] for L in lattice_sizes}
    magnetization_histograms = {L: [] for L in lattice_sizes}
    
    for L in lattice_sizes:
        for T in temperature_range:
            _, magnetization, _, magnetizations = simulate_wolff(L, T, steps)
            binder_cumulants[L].append(compute_binder_cumulant(magnetizations, L))
            magnetization_histograms[L].append(magnetizations)

    # Plotting
    fig, axes = plt.subplots(len(lattice_sizes) + 1, 1, figsize=(10, 18))
    
    # Combined Binder Cumulant vs Temperature
    for L in lattice_sizes:
        axes[0].plot(temperature_range, binder_cumulants[L], label=f'{L}x{L}')
    axes[0].axvline(x=2.27, color='red', linestyle='--', label='T_c = 2.27')
    axes[0].set_xlabel('Temperature')
    axes[0].set_ylabel('Binder Cumulant')
    axes[0].set_title('Binder Cumulant vs Temperature')
    axes[0].legend()

    # Separate Magnetization Histograms
    for idx, L in enumerate(lattice_sizes):
        for i, T in enumerate(temperatures_6x6):
            axes[idx + 1].hist(magnetization_histograms[L][i], bins=30, alpha=0.5, label=f'T={T}')
        axes[idx + 1].set_xlabel('Magnetization')
        axes[idx + 1].set_ylabel('Frequency')
        axes[idx + 1].set_title(f'{L}x{L} Magnetization Histogram')
        axes[idx + 1].legend()

    # Adjust the spacing between subplots
    plt.subplots_adjust(hspace=2.5)  # Increase hspace to make more room between subplots

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
