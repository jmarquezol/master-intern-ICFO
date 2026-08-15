import matplotlib.pyplot as plt
import time
from peps import PEPS

# --- PARAMETERS ---
L = 11
d_phys = 2
D = 2           # bond dim of the PEPS
D_bound = 4     # max bond dim for the boundary MPS

print(f"--- PEPS SIMULATION: EXACT vs APPROXIMATE (Grid {Lx}x{Ly}, D={D}) ---")

# 1. GENERATE RANDOM PEPS: TNs with bond dim D
peps_obj = PEPS.create_random_2d_peps(Lx, Ly, d_phys, D, seed=42)

# 2. RUN BENCHMARKS
val_approx1 = peps_obj.contract_2d(D_bound=D_bound)
val_exact1 = peps_obj.contract_2d_exact()

print(f"    Approx Result: {val_approx1:.10f}")
print(f"    Exact Result:  {val_exact1:.10f}")
print(f"    Rel Error:     {abs(val_approx1 - val_exact1)/abs(val_exact1):.2e}")

# --- REL. ERROR VS SYSTEM SIZE ---
lattice_sizes = range(2, 11)

errors = []
times_approx = []
times_exact = []

for L in lattice_sizes:
    print(f"Calculating for grid {L}x{L} with D={D}...")
    # 1. Initialize random PEPS
    peps_obj = PEPS.create_random_2d_peps(L, L, d_phys, D, seed=42)

    # 2. Approx Benchmark
    t0 = time.time()
    val_approx = peps_obj.contract_2d(D_bound=D_bound)
    t_approx = time.time() - t0

    # 3. Exact Benchmark
    t0 = time.time()
    val_exact = peps_obj.contract_2d_exact()
    t_exact = time.time() - t0

    # 4. Results
    rel_err = abs(val_approx - val_exact) / (abs(val_exact) + 1e-15)

    errors.append(rel_err)
    times_approx.append(t_approx)
    times_exact.append(t_exact)

    print(f"Finished. Relative error: {rel_err:.2e}")

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 6))

# Axis 1: Relative error (log plot scale)
color1 = 'red'
ax1.set_xlabel('Size of the grid (L x L)')
ax1.set_ylabel('Relative Error', color=color1)
ax1.semilogy(lattice_sizes, errors, marker='o', color=color1, linewidth=2, label='Relative Error')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, which="both", ls="-", alpha=0.3)

# Axis 2: Execution time
ax2 = ax1.twinx()
color2 = 'blue'
ax2.set_ylabel('Time (s)', color=color2)
ax2.plot(lattice_sizes, times_approx, marker='s', linestyle='--', color='tab:blue', label='Approx. Time')
ax2.plot(lattice_sizes, times_exact, marker='^', linestyle='--', color='tab:cyan', label='Exact Time')
ax2.tick_params(axis='y', labelcolor=color2)

# Estética final
plt.title(f'PEPS Benchmark: Error vs System Size (D={D}, D_bound={D_bound})')
fig.tight_layout()
plt.legend(loc='upper left')
plt.show()