import utils.logging
import matplotlib.pyplot as plt
import numpy as np
import sys

if len(sys.argv) > 1:
    filename = str(sys.argv[1])
else:
    filename = 'timestamp_log.txt'

parser = utils.logging.ModuleLogParser()
parser.read(filename)

timestamps = parser.get_timestamps()
accelerations = [parser.get(timestamp).get('acceleration', 0) for timestamp in timestamps]
actual_distances = [parser.get(timestamp).get('actual_distance', 0) for timestamp in timestamps]
perceived_distances = [parser.get(timestamp).get('perceived_distance', 0) for timestamp in timestamps]

# Compute change in acceleration (acc_change)
acc_changes = [(accelerations[i+1] - accelerations[i]) / (float(timestamps[i+1]) - float(timestamps[i])) if i < len(accelerations) - 1 else 0 for i in range(len(accelerations))]

# Plot actual vs. perceived distance
plt.figure(figsize=(12, 6))

xs = np.arange(len(timestamps))
ads = np.array(actual_distances).astype(np.double)
ads_mask = np.isfinite(ads)
pds = np.array(perceived_distances).astype(np.double)
pds_mask = np.isfinite(pds)

#plt.plot(xs[ads_mask], ads[ads_mask], linestyle='-', marker='o')
#plt.plot(xs[pds_mask], pds[pds_mask], linestyle='-', marker='o')

plt.plot(timestamps, actual_distances, label='Actual Distance')
plt.plot(timestamps, perceived_distances, label='Perceived Distance', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Distance')
plt.title('Actual vs. Perceived Distance over Time')
plt.legend()
plt.show()

# Plot change in acceleration
plt.figure(figsize=(12, 6))
plt.plot(timestamps[:-1], acc_changes[:-1], label='Acceleration Change', marker='o')  # Exclude the last point (no change)
plt.xlabel('Time')
plt.ylabel('Acceleration Change')
plt.title('Change in Acceleration over Time')
plt.legend()
plt.show()
