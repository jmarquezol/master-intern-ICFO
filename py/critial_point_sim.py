import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from peps import PEPS

# HELPER FUNCTIONS

def measure_energy(config, J=1.0):
    spins = np.where(config == 0, 1, -1)
    horiz_pairs = spins[:, :-1] * spins[:, 1:]
    vert_pairs = spins[:-1, :] * spins[1:, :]
    return -J * (np.sum(horiz_pairs) + np.sum(vert_pairs))

def run_mcmc_sweep(Lx, Ly, beta, D_bound, N_samples, J=1.0):
    """
    Runs the optimized MCMC for a specific grid and temperature.
    Returns the acceptance rate and average energy per site.
    """
    ising = PEPS.create_ising_2d(Lx, Ly, beta=beta, J=J)
    
    # Initialize
    current_config, current_log_prob = ising.sample_config_opt(D_bound=D_bound)
    current_energy = measure_energy(current_config, J)
    
    acceptance_count = 0
    energy_history = []
    
    for _ in range(N_samples):
        # 1. Propose
        new_config, new_log_prob = ising.sample_config_opt(D_bound=D_bound)
        new_energy = measure_energy(new_config, J)
        
        # 2. Metropolis Filter
        log_TN_ratio = current_log_prob - new_log_prob
        log_boltzmann_ratio = -beta * (new_energy - current_energy)
        log_acceptance = log_TN_ratio + log_boltzmann_ratio
        
        if np.log(np.random.rand()) < log_acceptance:
            current_config = new_config
            current_log_prob = new_log_prob
            current_energy = new_energy
            acceptance_count += 1
            
        energy_history.append(current_energy / (Lx * Ly))
        
    # Discard 20% burn-in
    burn_in = int(0.2 * N_samples)
    acc_rate = acceptance_count / N_samples
    avg_e = np.mean(energy_history[burn_in:])
    
    return acc_rate, avg_e

# MAIN SIMULATION
if __name__ == "__main__":
    # Parameters to sweep
    L_values = [16, 32, 64]  # Test different grid sizes (N = L x L)
    # Create a beta array with higher density near the critical point (0.44)
    beta_values = np.concatenate([
        np.linspace(0.3, 0.4, 2),
        np.linspace(0.41, 0.48, 4),
        np.linspace(0.5, 0.6, 2)
    ])
    
    D_bound = 2
    N_samples = 2000 # Keep it relatively low for the sweep to finish in reasonable time
    
    # Dictionaries to store results
    results_acc = {L: [] for L in L_values}
    results_e = {L: [] for L in L_values}
    
    print(f"Starting Phase Transition Sweep for D_bound={D_bound}")
    print(f"Betas: {beta_values}")
    
    for L in L_values:
        print(f"\n--- Running for Grid {L}x{L} ---")
        for beta in tqdm(beta_values, desc=f"L={L} Sweep"):
            acc, e = run_mcmc_sweep(Lx=L, Ly=L, beta=beta, D_bound=D_bound, N_samples=N_samples)
            results_acc[L].append(acc)
            results_e[L].append(e)

    # PLOTTING

    beta_c_analytical = np.log(1 + np.sqrt(2)) / 2  # ~ 0.44068

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    
    for idx, L in enumerate(L_values):
        ax1.plot(beta_values, results_acc[L], marker='o', label=f'L={L}', color=colors[idx])
        ax2.plot(beta_values, results_e[L], marker='s', label=f'L={L}', color=colors[idx])
        
    # Format Acceptance Rate Plot
    ax1.axvline(x=beta_c_analytical, color='red', linestyle='--', label=r'$\beta_c \approx 0.441$')
    ax1.set_title(f"Acceptance Rate vs Inverse Temperature (D={D_bound})")
    ax1.set_xlabel(r"Inverse Temperature $\beta$")
    ax1.set_ylabel("MCMC Acceptance Rate")
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.4)
    ax1.legend()
    
    # Format Energy Plot
    ax2.axvline(x=beta_c_analytical, color='red', linestyle='--', label=r'$\beta_c \approx 0.441$')
    ax2.set_title("Average Energy per Site vs Inverse Temperature")
    ax2.set_xlabel(r"Inverse Temperature $\beta$")
    ax2.set_ylabel(r"$\langle E \rangle / N$")
    ax2.grid(True, alpha=0.4)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()