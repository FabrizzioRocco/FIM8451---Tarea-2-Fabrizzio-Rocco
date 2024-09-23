import numpy as np
import matplotlib.pyplot as plt

def initialize_lattice(L):
    return np.random.choice([-1, 1], size=(L, L))

def compute_energy(lattice, J=1):
    L = lattice.shape[0]
    energy = 0
    for i in range(L):
        for j in range(L):
            S = lattice[i, j]
            neighbors = lattice[(i+1) % L, j] + lattice[i, (j+1) % L] + lattice[(i-1) % L, j] + lattice[i, (j-1) % L]
            energy -= J * S * neighbors
    return energy / 2.0  # Each pair counted twice

def metropolis_step(lattice, T, J=1):
    L = lattice.shape[0]
    for _ in range(L * L):
        i, j = np.random.randint(0, L, size=2)
        S = lattice[i, j]
        neighbors = lattice[(i+1) % L, j] + lattice[i, (j+1) % L] + lattice[(i-1) % L, j] + lattice[i, (j-1) % L]
        dE = 2 * J * S * neighbors
        if dE < 0 or np.random.rand() < np.exp(-dE / T):    
            lattice[i, j] = -S

def simulate_ising(L, T, steps, J=1):
    lattice = initialize_lattice(L)
    energy_list = []
    magnetization_list = []
    for step in range(steps):
        metropolis_step(lattice, T, J)
        if step >= steps / 2:               
            energy = compute_energy(lattice, J)
            energy_list.append(energy)
            magnetization_list.append(np.abs(np.sum(lattice)))
    
    mean_energy = np.mean(energy_list)
    mean_magnetization = np.mean(magnetization_list) / (L * L)
    
    # Specific heat capacity (Cv) calculation
    mean_energy_squared = np.mean(np.array(energy_list) ** 2)
    specific_heat = (mean_energy_squared - mean_energy ** 2) / (T ** 2 * L * L)
    
    return mean_energy, mean_magnetization, specific_heat

def main():
    temperatures_6x6 = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]  
    temperature_range = np.arange(0.25, 5.25, 0.25)  
    lattice_sizes = [4, 8, 16, 32]  
    steps = 1000  # Number of Metropolis steps

    # Data storage for plotting
    magnetizations = {L: [] for L in lattice_sizes}
    specific_heats = {L: [] for L in lattice_sizes}

    # Specific printout for 6x6 lattice
    L = 6
    print(f"\nLattice size {L}x{L}:")
    print(f"{'Temperature':>12} {'Energy':>12} {'Specific Heat':>15}")
    for T in temperatures_6x6:
        energy, magnetization, specific_heat = simulate_ising(L, T, steps)
        # Print results with at least 4 decimal places
        print(f"{T:12.1f} {energy / (L * L):12.5f} {specific_heat:15.5f}")

    # Collect data for plotting for other lattice sizes
    for L in lattice_sizes:
        for T in temperature_range:
            _, magnetization, specific_heat = simulate_ising(L, T, steps)
            magnetizations[L].append(magnetization)
            specific_heats[L].append(specific_heat)

    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    for L in lattice_sizes:
        ax[0].plot(temperature_range, magnetizations[L], label=f'{L}x{L}')
        ax[1].plot(temperature_range, specific_heats[L], label=f'{L}x{L}')

    ax[0].set_xlabel('Temperature')
    ax[0].set_ylabel('Magnetization')
    ax[0].set_title('Magnetization vs Temperature')
    ax[0].legend()

    ax[1].set_xlabel('Temperature')
    ax[1].set_ylabel('Specific Heat per Spin')
    ax[1].set_title('Specific Heat vs Temperature')
    ax[1].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
