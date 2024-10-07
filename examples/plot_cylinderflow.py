import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import numpy as np

fig, ax = plt.subplots(nrows=2, ncols=2)

strouhal_freq = 0.1727  # Hz
strouhal_per = 1 / strouhal_freq

drag_data = np.genfromtxt("results/drag_over_time.csv", delimiter=",", skip_header=1)

tt = drag_data[:, 0]
drag = drag_data[:, 1]

tt = np.divide(tt, strouhal_per)

ax[0, 0].plot(tt, drag, color="orange")
ax[0, 0].set_xlabel("Time (# of Strouhal periods)")
ax[0, 0].set_ylabel("Drag Coefficient")
ax[0, 0].set_xlim(0, 30)
ax[0, 0].set_ylim(1, 2.5)

lift_data = np.genfromtxt("results/lift_over_time.csv", delimiter=",", skip_header=1)

tt = lift_data[:, 0]
lift = lift_data[:, 1]

N = len(tt)
dt = tt[-1] - tt[-2]

tt = np.divide(tt, strouhal_per)

# Do an FFT and get the freq bins that FFT lives on
lift_hat = fft(lift)
freq = fftfreq(N, dt)

ax[0, 1].plot(tt, lift, color="orange")
ax[0, 1].set_xlabel("Time (# of Strouhal periods)")
ax[0, 1].set_ylabel("Lift Coefficient")
ax[0, 1].set_xlim(0, 30)
ax[0, 1].set_ylim(-1.5, 1.5)

half_N = int(np.floor(N / 2))
freq = np.divide(freq, strouhal_freq)

ax[1, 0].plot(freq[:half_N], np.abs(lift_hat[:half_N]))
ax[1, 0].set_xlabel("Frequency (ratio to Strouhal frequency)")
ax[1, 0].set_ylabel("Energy")
ax[1, 0].set_xlim(0, 2)
ax[1, 0].set_ylim(0, 3000)
max_val_idx = np.argmax(np.abs(lift_hat[:half_N]))
in_strouhals = freq[max_val_idx]
print(f"Found max component in lift signal at {in_strouhals} Strouhal frequencies")

y_data = np.genfromtxt("results/y_over_time.csv", delimiter=",", skip_header=1)[:, 1]

cutoff = 900
ax[1, 1].plot(y_data[cutoff:], lift[cutoff:])
ax[1, 1].set_xlabel("y, vertical position shift")
ax[1, 1].set_ylabel("Lift Coefficient")

CLmax = np.max(lift[1500:])
print(f"Found max lift coefficient after stabilization of {CLmax}")
CDmean = np.mean(drag[1500:])
print(f"Found mean drag coefficient after stabilization of {CDmean}")

plt.tight_layout()
plt.savefig("results/comparison.jpg")
