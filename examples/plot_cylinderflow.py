import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import numpy as np

strouhal_freq = 0.1727  # Hz
strouhal_per = 1 / strouhal_freq

drag_data = np.genfromtxt("results/drag_over_time.csv", delimiter=",", skip_header=1)

tt = drag_data[:, 0]
drag = drag_data[:, 1]

tt = np.divide(tt, strouhal_per)

plt.plot(tt, drag, color='orange')
plt.xlabel("Time (# of Strouhal periods)")
plt.ylabel("Drag Coefficient")
plt.xlim(0,30)
plt.show()

lift_data = np.genfromtxt("results/lift_over_time.csv", delimiter=",", skip_header=1)

tt = lift_data[:, 0]
lift = lift_data[:, 1]

N = len(tt)
dt = tt[-1] - tt[-2]

tt = np.divide(tt, strouhal_per)

# Do an FFT and get the freq bins that FFT lives on
lift_hat = fft(lift)
freq = fftfreq(N, dt)

plt.plot(tt, lift, color='orange')
plt.xlabel("Time (# of Strouhal periods)")
plt.ylabel("Lift Coefficient")
plt.xlim(0,30)
plt.show()

half_N = int(np.floor(N/2))
freq = np.divide(freq, strouhal_freq)

plt.plot(freq[:half_N], np.abs(lift_hat[:half_N]))
plt.xlabel("Frequency (ratio to Strouhal frequency)")
plt.ylabel("Energy")
plt.xlim(0, 2)
plt.ylim(0,3000)
max_val_idx = np.argmax(np.abs(lift_hat[:half_N]))
in_strouhals = freq[max_val_idx]
print(f"Found max component in lift signal at {in_strouhals} Strouhal frequencies")
plt.show()