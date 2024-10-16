import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-2 * np.pi, 4 * np.pi, 500)

y = np.sin(x)

plt.plot(x, y, label="sin(x)")

plt.title("Sinusoidal Wave")
plt.xlabel("x")
plt.ylabel("sin(x)")

plt.grid(True)
plt.legend()
plt.savefig("sin_wave.jpg")