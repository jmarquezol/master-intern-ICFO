import numpy as np
import matplotlib.pyplot as plt
from peps import PEPS

# --- PARAMETERS ---
L = 13         # LxL lattice
J = 1.0

# range of temp values to see behaviuor near T_critical
beta_values = np.linspace(0.1, 0.8, 100)    

d_phys = 2
D = 2           # bond dim of the PEPS
D_bound = 10    # max bond dim for the boundary MPS

z_approx_list = []
z_exact_list = []
error_list = []

print(f"Benchmarking Ising 2D on {L}x{L} grid...")

for beta in beta_values:
    # Create Ising PEPS
    ising_peps = PEPS.create_ising_2d(Lx=L, Ly=L, beta=beta, J=J)
    
    # 1. Approximate Contraction (Boundary MPS)
    val_approx = ising_peps.contract_2d(D_bound=D_bound)
    z_approx_list.append(val_approx)
    
    # 2. Exact Contraction (opt_einsum w/o truncation)
    val_exact = ising_peps.contract_2d_exact()
    z_exact_list.append(val_exact)
    
    error = abs(val_approx - val_exact) / val_exact
    error_list.append(error)
    print(f"beta={beta:.2f} | Error: {error:.2e}")

# Plotting
plt.figure(figsize=(10, 5))
plt.semilogy(beta_values, error_list, 'r-')
plt.axvline(0.4407, color='k', linestyle=':', label='Critical Temperature')
plt.xlabel(r' Temperature, $\beta$')
plt.ylabel('Relative Error')
plt.title(f'Error (Approx vs Exact) vs Temperature (grid {L}x{L})')
plt.legend()

plt.tight_layout()
plt.show()


# ------------------------------------------------------------------
# COMPARISON: Z FROM BRUTE FORCE VS Z FROM EXACT PEPS CONTRACTION
# ------------------------------------------------------------------
beta = 0.3
L_small = 4

ising_peps = PEPS.create_ising_2d(Lx=L_small, Ly=L_small, beta=beta, J=J)
z_exact = ising_peps.contract_2d_exact()

# Brute force Z
z_brute = PEPS.compute_Z_brute_force(Lx=L_small, Ly=L_small, beta=beta, J=J)

print(f"\nBrute Force Z: {z_brute:.6e}")
print(f"Exact PEPS Z:   {z_exact:.6e}")

# Relative Error:
rel_error = abs(z_brute - z_exact) / z_exact
print(f"Relative Error: {rel_error:.2e}")

# ------------------------------------------------------------------
# COMPARISON: Z FROM BRUTE FORCE VS Z FROM APPROX PEPS CONTRACTION
# ------------------------------------------------------------------beta = 0.3

z_approx = ising_peps.contract_2d(D_bound=D_bound)

print(f"\nBrute Force Z: {z_brute:.6e}")
print(f"Approx PEPS Z:   {z_approx:.6e}")

# Relative Error:
rel_error = abs(z_brute - z_approx) / z_exact
print(f"Relative Error: {rel_error:.2e}")


