import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from peps import PEPS

# HELPER FUNCTIONS
def measure_magnetization(config):
    """
    Computes average magnetization per site m = (1/N) * sum(spins).
    Input config is 0 (Up) and 1 (Down).
    We map: 0 -> +1, 1 -> -1
    """
    # We convert 0/1 to +1/-1
    spins = np.where(config == 0, 1, -1)
    
    return np.mean(spins)

def measure_energy(config, J=1.0):
    """
    Computes energy per site E/N for the 2D Ising model.
    H = -J * sum(<ij> s_i s_j)
    """
    Lx, Ly = config.shape
    spins = np.where(config == 0, 1, -1)
    
    energy = 0.0
    
    # Horizontal interactions (i, i+1):
    #   Element-wise multiply columns 0..L-2 with columns 1..L-1
    horiz_pairs = spins[:, :-1] * spins[:, 1:]
    energy += -J * np.sum(horiz_pairs)
    
    # Vertical interactions (j, j+1):
    #   Element-wise multiply rows 0..L-2 with rows 1..L-1
    vert_pairs = spins[:-1, :] * spins[1:, :]
    energy += -J * np.sum(vert_pairs)
    
    # Energy per site
    return energy / (Lx * Ly)


# PARAMETERS
N_samples = 1000 
D_bound = 8
Lx, Ly = 4, 4
beta = 0.4
J = 1.0

# Create the PEPS Object
ising = PEPS.create_ising_2d(Lx, Ly, beta=beta, J=J)

# Sample a random configuration from the PEPS distribution
config, log_prob = ising.sample_configuration(D_bound=10)

print("\nGenerated Configuration:")
print(config)

# Storage for observables
mag_history = []
energy_history = []

print(f"Starting MC Independent Sampling (Beta={beta}, Samples={N_samples})...")

for i in tqdm(range(N_samples)):
    # For Independent Sampling, we generate every sample from scratch.
    config, lop_prob = ising.sample_configuration(D_bound)
    
    m = measure_magnetization(config)
    e = measure_energy(config, J)
    
    mag_history.append(m)
    energy_history.append(e)


# Statistics Analysis
avg_mag = np.mean(mag_history)
avg_energy = np.mean(energy_history)

std_mag = np.std(mag_history) / np.sqrt(N_samples)
std_energy = np.std(energy_history) / np.sqrt(N_samples)

print("\nRESULTS:")
print(f"Average Magnetization: {avg_mag:.5f} +/- {std_mag:.5f}")
print(f"Average Energy:        {avg_energy:.5f} +/- {std_energy:.5f}")

# Plotting Convergence of Mag and Energy
fig, ax1 = plt.subplots(figsize=(10, 6))

# Magnetization Plot (Left Axis)
color_mag = 'tab:blue'
ax1.set_xlabel('Sample Number')
ax1.set_ylabel('Magnetization <M>', color=color_mag, fontweight='bold')
ax1.plot(np.cumsum(mag_history) / np.arange(1, N_samples + 1), 
         color=color_mag, lw=2, label='Magnetization')
ax1.tick_params(axis='y', labelcolor=color_mag)
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()  

# Energy Plot (Right Axis)
color_eng = 'tab:red'
ax2.set_ylabel('Energy <E>', color=color_eng, fontweight='bold')
ax2.plot(np.cumsum(energy_history) / np.arange(1, N_samples + 1), 
         color=color_eng, linestyle='--', lw=2, label='Energy')
ax2.tick_params(axis='y', labelcolor=color_eng)

# 3. Final Touches
plt.title(f"Convergence of MC Indep Sampling (beta={beta}, Grid {Lx}x{Ly})")
fig.tight_layout()
plt.show()