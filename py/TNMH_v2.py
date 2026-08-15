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
    Computes total energy for the 2D Ising model.
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
    
    return energy


# PARAMETERS
N_samples = 15000 
D_bound = 8
Lx, Ly = 4, 4
beta = 0.44
J = 1.0

print(f"Starting TNMH Algorithm (Beta={beta}, Samples={N_samples}, Grid={Lx}x{Ly})...")

# Create PEPS Object
ising = PEPS.create_ising_2d(Lx, Ly, beta=beta, J=J)

# Initialize Markov Chain (Step t=0) using the OPTIMIZED method
current_config, current_log_prob = ising.sample_config_opt(D_bound=D_bound)
current_energy = measure_energy(current_config, J)

mag_history = []
energy_history = []
acceptance_count = 0

pbar = tqdm(range(N_samples), desc="MCMC Evolution")

for t in pbar:
    # 1. PROPOSE (sample a new candidate config from the PEPS distribution)
    # MUST USE sample_config_opt HERE
    new_config, new_log_prob = ising.sample_config_opt(D_bound=D_bound)
    new_energy = measure_energy(new_config, J)

    # 2. ACCEPT/REJECT 
    log_TN_ratio = current_log_prob - new_log_prob
    log_boltzmann_ratio = -beta * (new_energy - current_energy)
    log_acceptance = log_TN_ratio + log_boltzmann_ratio

    if np.log(np.random.rand()) < log_acceptance:
        current_config = new_config
        current_log_prob = new_log_prob
        current_energy = new_energy
        acceptance_count += 1

    # 3. MEASURE
    m = measure_magnetization(current_config)
    e = current_energy / (Lx * Ly) 
    
    mag_history.append(m)
    energy_history.append(e)

    if t % 100 == 0 or t == N_samples - 1:
        running_acc_rate = acceptance_count / (t + 1)
        pbar.set_postfix({"Acc": f"{running_acc_rate:.1%}"})

# ANALYSIS 
acceptance_rate = acceptance_count / N_samples

burn_in = int(0.2 * N_samples)
mag_history = mag_history[burn_in:]
energy_history = energy_history[burn_in:]

N_eff = len(mag_history)
avg_mag = np.mean(mag_history)
avg_energy = np.mean(energy_history)
std_mag = np.std(mag_history) / np.sqrt(N_eff)
std_energy = np.std(energy_history) / np.sqrt(N_eff)

print("\nRESULTS:")
print(f"Acceptance Rate:     {acceptance_rate:.2%}")
print(f"Average Magnetization: {avg_mag:.5f} +/- {std_mag:.5f}")
print(f"Average Energy:        {avg_energy:.5f} +/- {std_energy:.5f}")

# PLOTTING
fig, ax1 = plt.subplots(figsize=(10, 6))
x_axis = np.arange(1, N_eff + 1)

color_mag = 'tab:blue'
ax1.set_xlabel('Sample Number')
ax1.set_ylabel('Magnetization <M>', color=color_mag, fontweight='bold')
ax1.plot(x_axis, np.cumsum(mag_history) / x_axis, color=color_mag, lw=2, label='Magnetization')
ax1.tick_params(axis='y', labelcolor=color_mag)
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()  
color_eng = 'tab:red'
ax2.set_ylabel('Energy <E>', color=color_eng, fontweight='bold')
ax2.plot(x_axis, np.cumsum(energy_history) / x_axis, color=color_eng, linestyle='--', lw=2, label='Energy')
ax2.tick_params(axis='y', labelcolor=color_eng)

plt.title(f"Convergence of MC Indep Sampling (optimized) (beta={beta}, Grid {Lx}x{Ly})")
fig.tight_layout()
plt.show()