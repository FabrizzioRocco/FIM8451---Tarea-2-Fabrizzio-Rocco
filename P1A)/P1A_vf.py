import numpy as np
import matplotlib.pyplot as plt

def direct_pi(N):
    """Monte Carlo estimation of π using N random points."""
    x = np.random.uniform(-1, 1, N)
    y = np.random.uniform(-1, 1, N)
    inside_circle = (x**2 + y**2) <= 1
    Nhits = np.sum(inside_circle)
    pi_estimate = 4 * Nhits / N
    return pi_estimate, Nhits

def mean_square_deviation(N, runs=20):
    """Calculate the mean square deviation of the π estimate for given N over multiple runs."""
    deviations = []
    pi_estimates = []
    
    for _ in range(runs):
        pi_estimate, Nhits = direct_pi(N)
        pi_estimates.append(pi_estimate)
        deviation = (Nhits / N - np.pi / 4)**2
        deviations.append(deviation)
    
    mean_pi_estimate = np.mean(pi_estimates)
    mean_deviation = np.mean(deviations)
    return mean_pi_estimate, mean_deviation

# Parameters
N_values = [10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000]
runs = 20
mean_square_devs = []
pi_estimates = []

# Running the simulations and calculating mean square deviations and pi estimates
for N in N_values:
    mean_pi_estimate, msd = mean_square_deviation(N, runs)
    pi_estimates.append(mean_pi_estimate)
    mean_square_devs.append(msd)
    print(f"N = {N}: Pi Estimate (4*N/Nhits) = {mean_pi_estimate}, Mean square deviation = {msd}, Mean square deviation/N = {msd/N}")

# Creating subplots
plt.figure(figsize=(14, 6))

# Subplot 1: Pi Estimate vs N
plt.subplot(1, 2, 1)
plt.plot(N_values, pi_estimates, '-o')
plt.xscale('log')
plt.xlabel('N (log scale)')
plt.ylabel('Pi Estimate')
plt.title('Pi Estimate as a Function of N')
plt.grid(True)

# Subplot 2: Mean Square Deviation vs N
plt.subplot(1, 2, 2)
plt.plot(N_values, mean_square_devs, '-o')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('N (log scale)')
plt.ylabel('Mean Square Deviation (log scale)')
plt.title('Mean Square Deviation of π Estimate as a Function of N')
plt.grid(True)

plt.tight_layout()
plt.show()
