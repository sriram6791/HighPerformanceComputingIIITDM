#%%
import matplotlib.pyplot as plt

# Data from the report
threads = [1, 2, 4, 6, 8, 10, 12, 16, 20, 32, 64]
time_taken = [29.293817, 16.329146, 9.175332, 8.166987, 7.553509, 6.586831, 6.122119, 7.061283, 7.068791, 8.161721, 10.251109]

plt.figure(figsize=(10, 6))
plt.plot(threads, time_taken, marker='o', linestyle='-', color='b')
plt.xlabel("Number of Threads")
plt.ylabel("Time Taken (seconds)")
plt.title("Threads vs Time Taken")
plt.grid(True)
plt.savefig("threads_vs_time.png")
plt.show()

# %%
