import sys
import json
from datetime import datetime
import matplotlib.pyplot as plt

# Step 1: Read and parse the log file
log_file = sys.argv[1]
eval_losses = []
timestamps = []

with open(log_file, "r") as file:
    for line in file:
        log_entry = json.loads(line)
        message = log_entry.get("message", {})
        if "eval_loss" in message:
            # Extract eval_loss and timestamp
            eval_losses.append(message["eval_loss"])
            timestamps.append(datetime.fromisoformat(log_entry["timestamp"]))

# Step 2: Calculate relative time elapsed
if timestamps:
    start_time = timestamps[0]
    elapsed_times = [(ts - start_time).total_seconds() for ts in timestamps]

    # Step 3: Plot eval_loss vs elapsed time
    plt.figure(figsize=(8, 5))
    plt.plot(elapsed_times, eval_losses, marker="o", label="Eval Loss")
    plt.title("Evaluation Loss Over Time")
    plt.xlabel("Time Elapsed (seconds)")
    plt.ylabel("Eval Loss")
    plt.yscale("log")
    # plt.ylim(bottom=1e-4)
    plt.grid()
    plt.legend()
    plt.show()
else:
    print("No eval_loss values found in the log.")

