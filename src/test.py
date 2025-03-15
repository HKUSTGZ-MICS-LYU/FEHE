from utils import quantization

# Test the sigma quantization

# step 1: Generate three sequences, each value is a random number between -1 and 1, the length of the sequence is 100
import numpy as np
import matplotlib.pyplot as plt
import time
np.random.seed(int(time.time()))
size = 100
seq1 = np.random.uniform(-1, 1, size=size)
seq2 = np.random.uniform(-1, 1, size=size)
seq3 = np.random.uniform(-1, 1, size=size)


# step 2: Calulate the sigma of the three sequences
concensus = np.concatenate([seq1, seq2, seq3])
max_val = np.max(concensus)
min_val = np.min(concensus)
sigma = np.std(concensus)
mu = np.mean(concensus)
print(f"sigma: {sigma}, mu: {mu}")

# step 3: Quantize the three sequences
quantizer = quantization.Quantizer()
q_weight1, q_params1 = quantizer.quantize_weights_unified(
                seq1,
                n_bits=8,
                sigma_bits=[8,8,8,8],
                method="sigma",
                global_max=max_val,
                global_min=min_val,
                global_mu=mu,
                global_sigma=sigma
            )
q_weight2, q_params2 = quantizer.quantize_weights_unified(
                seq2,
                n_bits=8,
                sigma_bits=[8,8,8,8],
                method="sigma",
                global_max=max_val,
                global_min=min_val,
                global_mu=mu,
                global_sigma=sigma
            )
q_weight3, q_params3 = quantizer.quantize_weights_unified(
                seq3,
                n_bits=8,
                sigma_bits=[8,8,8,8],
                method="sigma",
                global_max=max_val,
                global_min=min_val,
                global_mu=mu,
                global_sigma=sigma
            )   

dq_weight1 = quantizer.dequantize_weights_unified(q_weight1, q_params1)
dq_weight2 = quantizer.dequantize_weights_unified(q_weight2, q_params2)
dq_weight3 = quantizer.dequantize_weights_unified(q_weight3, q_params3)

# step 4: Calculate the error
error1 = np.mean(np.abs(seq1 - dq_weight1))
error2 = np.mean(np.abs(seq2 - dq_weight2))
error3 = np.mean(np.abs(seq3 - dq_weight3))
print(f"error1: {error1}, error2: {error2}, error3: {error3}")

# step 5: Plot the three original and quantized sequences 
plt.figure()
plt.plot(seq1, label="seq1")
plt.plot(dq_weight1, label="q_seq1")
plt.legend()
plt.savefig("seq1.png")

plt.figure()
plt.plot(seq2, label="seq2")
plt.plot(dq_weight2, label="q_seq2")
plt.legend()
plt.savefig("seq2.png")

plt.figure()
plt.plot(seq3, label="seq3", color="red")
plt.plot(dq_weight3, label="q_seq3")
plt.legend()
plt.savefig("seq3.png")

for i in range(len(seq1)):
    print(f"seq1: {seq1[i]}, q_seq1: {dq_weight1[i]}")



