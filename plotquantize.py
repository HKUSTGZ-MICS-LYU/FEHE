import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# --------------------------------------------------------
#  Example: Quantization of a Gaussian (N(0,1)) distribution
#           into 2^b intervals (b bits).
# --------------------------------------------------------

# 1. Setup: define the number of bits b and thus the number of intervals L
b = 3              # for example, 3 bits
L = 2 ** b         # number of intervals (8 intervals)

# 2. Create x-axis and the PDF of a standard normal distribution
x_min, x_max = -4, 4
x = np.linspace(x_min, x_max, 400)
pdf = norm.pdf(x, loc=0, scale=1)

# 3. Define the partition boundaries for L intervals in [x_min, x_max].
#    A simple approach: equally spaced intervals in this example.
#    (In practice, you might use non-uniform spacing, e.g., Lloyd-Max quantizer.)
boundaries = np.linspace(x_min, x_max, L+1)

# 4. Plot the PDF
plt.figure(figsize=(8, 5), dpi=100)
plt.plot(x, pdf, 'k-', label='N(0,1) PDF')

# 5. Fill each interval with a different color and annotate
colors = plt.cm.viridis(np.linspace(0.1, 0.9, L))  # a colormap with L different shades

for i in range(L):
    # Interval: [boundaries[i], boundaries[i+1])
    left = boundaries[i]
    right = boundaries[i+1]
    mask = (x >= left) & (x < right)
    
    # Fill the region under the PDF curve
    plt.fill_between(x[mask], pdf[mask], color=colors[i], alpha=0.3)
    
    # Compute a simple representative value (e.g., midpoint)
    q_i = 0.5 * (left + right)
    
    # Label text: interval index i, b-bit code, representative value
    # We place the text near the top of the PDF in this interval
    x_mid = q_i
    y_mid = norm.pdf(q_i, 0, 1) + 0.02
    
    plt.text(x_mid, y_mid,
             f"Interval {i}\nCode: {i:0{b}b}\nq={q_i:.2f}",
             ha='center', va='bottom', fontsize=8, color='black')

# 6. Cosmetics: axis, title, legend, etc.
plt.axhline(y=0, color='black', linewidth=0.8)
plt.title(f"Gaussian Quantization: b={b} bits, L={L} intervals", fontsize=11)
plt.xlabel("x")
plt.ylabel("PDF")
plt.ylim(0, 0.45)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
