#%%
import matplotlib.pyplot as plt

# Given data
num_threads = [1, 2, 4, 8, 16, 32, 64]
time_taken = [0.405624, 0.077494, 0.047199, 0.034554, 0.033767, 0.037875, 0.029812]

# Calculate speedup (Speedup = Time with 1 thread / Time with N threads)
speedup = [time_taken[0] / t for t in time_taken]

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(num_threads, speedup, marker='o', linestyle='-', color='b', label="Speedup")

# Labels and Title
plt.xlabel("Number of Processors (Threads)")
plt.ylabel("Speedup")
plt.title("Speedup vs Number of Threads")
plt.xticks(num_threads)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.show()

# %%
