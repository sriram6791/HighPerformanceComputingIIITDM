#%%
import matplotlib.pyplot as plt

# Hardcoded data from your results
num_threads = [1, 2, 4, 8, 16, 32, 64]
time_taken = [0.252104, 0.131160, 0.076296, 0.046388, 0.040050, 0.049185, 0.036436]

# Calculate Speedup
speedup = [time_taken[0] / t for t in time_taken]

# Plotting
plt.figure(figsize=(8,6))
plt.plot(num_threads, speedup, marker='o', linestyle='-', color='b', label="Speedup")

# Labels and Title
plt.xlabel("Number of Threads")
plt.ylabel("Speedup")
plt.title("Speedup vs Number of Threads")
plt.xticks(num_threads)  # Set x-axis ticks
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()

# Show the plot
plt.show()

# %%
