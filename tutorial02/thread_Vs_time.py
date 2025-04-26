#%%
import matplotlib.pyplot as plt

# Manually input the thread count and corresponding times
threads = [1, 2, 4, 8, 16, 32, 64]
times = [0.295900, 0.133871, 0.080889, 0.053715, 0.043149, 0.041698, 0.026964]

# Plotting the data
plt.figure(figsize=(10, 6))
plt.plot(threads, times, marker='o', color='b', label='Execution Time')

# Adding labels and title
plt.xlabel('Number of Threads')
plt.ylabel('Execution Time (seconds)')
plt.title('Threads vs Execution Time')
plt.grid(True)
plt.xscale('log')  # Log scale for x-axis to show the performance improvement clearly
plt.yscale('log')  # Log scale for y-axis to clearly visualize performance speedup
plt.legend()

# Show plot
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Given data
times = [0.295900, 0.133871, 0.080889, 0.053715, 0.043149, 0.041698, 0.026964]
threads = [1, 2, 4, 8, 16, 32, 64]

# Calculate the observed speedup (S_observed)
T1 = times[0]  # Time for 1 thread
S_observed = [T1 / t for t in times]

# Define Amdahl's Law function for fitting
def amdahls_law(p, f):
    return 1 / ((1 - f) + f / p)

# Fit the observed data to Amdahl's law
popt, _ = curve_fit(lambda p, f: amdahls_law(p, f), threads[1:], S_observed[1:], bounds=(0, 1))
f_estimate = popt[0]  # The estimated parallel fraction

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(threads[1:], S_observed[1:], marker='o', label='Observed Speedup')

# Calculate the theoretical speedup using the estimated parallel fraction
S_theoretical = [amdahls_law(p, f_estimate) for p in threads[1:]]
plt.plot(threads[1:], S_theoretical, marker='x', linestyle='--', label=f'Amdahl\'s Law (f={f_estimate:.4f})')

# Add labels and title
plt.xlabel('Number of Threads')
plt.ylabel('Speedup')
plt.title('Threads vs Speedup')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()

# Return the estimated parallel fraction
f_estimate

# %%
