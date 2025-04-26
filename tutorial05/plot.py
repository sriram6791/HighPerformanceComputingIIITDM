
#%%
import matplotlib.pyplot as plt

# Given data
threads = [1, 2, 4, 8, 16, 32, 64]
times = [4.706, 2.043, 1.101, 0.826, 0.575, 0.588, 0.610]

# Compute speedup (Speedup = Serial Time / Parallel Time)
serial_time = times[0]
speedup = [serial_time / t for t in times]

# Plot Speedup vs. Threads
plt.figure(figsize=(8, 5))
plt.plot(threads, speedup, marker='o', linestyle='-', color='b', label="Speedup")

# Labels and title
plt.xlabel("Number of Threads")
plt.ylabel("Speedup")
plt.title("Speedup vs. Number of Threads for Matrix Multiplication")
plt.xticks(threads)  # Set x-axis ticks to thread values
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()

# Show the plot
plt.show()

# %%
