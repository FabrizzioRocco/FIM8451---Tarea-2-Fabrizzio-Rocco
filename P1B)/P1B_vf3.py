import numpy as np
import matplotlib.pyplot as plt

def markov_pi(N, delta):
    """Markov Chain Monte Carlo estimation of π using N steps and step size delta."""
    x, y = 0.0, 0.0  # Start at the "clubhouse" (origin)
    Nhits = 0
    rejection_count = 0
    
    for _ in range(N):
        x_new = x + np.random.uniform(-delta, delta)
        y_new = y + np.random.uniform(-delta, delta)
        
        if x_new**2 + y_new**2 <= 1:  # Accept the move
            x, y = x_new, y_new
            Nhits += 1
        else:
            rejection_count += 1
    
    pi_estimate = 4 * Nhits / N
    rejection_rate = rejection_count / N
    
    return pi_estimate, Nhits, rejection_rate

def mean_square_deviation_markov(N, delta, runs=20):
    """Calculate the mean square deviation, pi estimate, and rejection rates for a fixed delta over multiple runs."""
    deviations = []
    pi_estimates = []
    rejection_rates = []
    
    for _ in range(runs):
        pi_estimate, Nhits, rejection_rate = markov_pi(N, delta)
        deviation = (pi_estimate - np.pi)**2
        deviations.append(deviation)
        pi_estimates.append(pi_estimate)
        rejection_rates.append(rejection_rate)
    
    return np.mean(deviations), np.mean(rejection_rates), np.mean(pi_estimates)

# Part 1: Fixed delta = 0.3, print for different N values
delta_fixed = 0.3
#N_values = [10, 100, 1000, 10000, 100000, 1000000]
N_values = [10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000]


print("Fixed delta =", delta_fixed)
for N in N_values:
    msd, rejection_rate, pi_estimate = mean_square_deviation_markov(N, delta_fixed)
    print(f"N = {N:>10}: Pi estimate = {pi_estimate:.6f}, Mean square deviation = {msd:.6e}")

# Part 2: Plotting Mean Square Deviation and Rejection Rate vs Delta
deltas = np.linspace(0.01, 3, 30)  # Range of delta values from 0.01 to 3
N_large = 1000000  # Fixed large N

mean_square_devs = []
rejection_rates = []

# Running the simulations for different delta values
for delta in deltas:
    msd, rejection_rate, _ = mean_square_deviation_markov(N_large, delta)
    mean_square_devs.append(msd)
    rejection_rates.append(rejection_rate)

# Plotting Mean Square Deviation as a Function of delta
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(deltas, mean_square_devs, '-o')
plt.xlabel('Delta')
plt.ylabel('Mean Square Deviation')
plt.title('Mean Square Deviation vs Delta (N = 1,000,000)')
plt.grid(True)

# Plotting Rejection Rate as a Function of delta
plt.subplot(1, 2, 2)
plt.plot(deltas, rejection_rates, '-o')
plt.xlabel('Delta')
plt.ylabel('Rejection Rate')
plt.title('Rejection Rate vs Delta (N = 1,000,000)')
plt.grid(True)

plt.tight_layout()
plt.show()
