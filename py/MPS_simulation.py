
import numpy as np
from matrix_product_states import MPS

N, d_phys, b_dim = 10, 2, 10
print(f"--- SIMULATION FOR N={N} PARTICLES ---")

#################################################
# TEST 1: GHZ STATE
#################################################
print("/nTEST 1: GHZ STATE (|0..0> + |1..1>)")
psi_ghz = MPS.create_ghz(N, d_phys=d_phys, b_dim=b_dim)

# VERIFY AMPLITUDES
print("\nChecking Amplitudes...")

# Check state |00...0>
state_zeros = [0]*N
amp_zeros = psi_ghz.compute_amplitude(state_zeros)
print(f"Amplitude |00...0>:  {amp_zeros:.4f} (Expected ~1.0 unnormalized)")

# Check state |11...1>
state_ones = [1]*N
amp_ones = psi_ghz.compute_amplitude(state_ones)
print(f"Amplitude |11...1>:  {amp_ones:.4f} (Expected ~1.0 unnormalized)")

# Check a mixed state |010...0>
state_mixed = [0]*N
state_mixed[1] = 1
amp_mixed = psi_ghz.compute_amplitude(state_mixed)
print(f"Amplitude |01...0>:  {amp_mixed:.4f} (Expected 0.0)")

# VERIFY NORMS
print("\nChecking Norm...")
print(f"Norm (Canonical method):  {psi_ghz.norm_canonical():.4f}")
print(f"Norm (General method):  {psi_ghz.norm_general():.4f}")
print(f"Expected Norm:  {np.sqrt(2):.4f}") # state |0...0> + |1...1> has length sqrt(1^2 + 1^2) = sqrt(2)

#################################################
# TEST 2: W STATE
#################################################
print("\nTEST 2: W STATE (|100..> + |010...> + ... + |00..1>)")
psi_w = MPS.create_w_state(N, d_phys=d_phys, b_dim=b_dim)
expected_amp = 1.0 / np.sqrt(N) # 1/sqrt(10) = 0.3162

# Verify Amplitudes of valid states
print(f"\n(Expected Amplitude: 1/sqrt({N}) = {expected_amp:.4f})")

# State |10...0> (First particle excited)
s1 = [0]*N; s1[0] = 1
amp_s1 = psi_w.compute_amplitude(s1)
print(f"|10...0>: {amp_s1:.4f}")

# State |00...1> (Last particle excited)
sN = [0]*N; sN[-1] = 1
amp_sN = psi_w.compute_amplitude(sN)
print(f"|00...1>: {amp_sN:.4f}")

# Verify Amplitudes of forbidden states)
print("\nChecking Forbidden States (Expected: 0.0)")

# Vacuum |00...0>
s_vac = [0]*N
amp_vac = psi_w.compute_amplitude(s_vac)
print(f"Vacuum |0...0>: {amp_vac:.4f}")

# Double Excitation |110...0>
s_double = [0]*N; s_double[0]=1; s_double[1]=1
amp_double = psi_w.compute_amplitude(s_double)
print(f"Double |11...0>: {amp_double:.4f}")

# Verify Norms
print("\nChecking Norms")
norm_can_w = psi_w.norm_canonical()
norm_gen_w = psi_w.norm_general()

print(f"Norm (Canonical): {norm_can_w:.4f}")
print(f"Norm (General):   {norm_gen_w:.4f}")
print(f"Expected Norm:    1.0000")

if np.isclose(norm_can_w, 1.0) and np.isclose(norm_gen_w, 1.0):
    print(">> SUCCESS: W State is valid.")
else:
    print(">> FAILURE: W State Norm mismatch.")


#################################################
# TEST 3: 1D Ising Model MPS
# Target: |Psi> = sum_sigma exp(-beta * H(sigma)) |sigma>
# H = sum sigma_i * sigma_{i+1}
#################################################

print("\nTEST 3: CLASSICAL ISING STATE")

beta = 0.5
print(f"Parameters: beta = {beta}, N = {N}")
psi_ising = MPS.create_ising_state(N, beta=beta, d_phys=d_phys, b_dim=b_dim)

# Check Amplitudes vs Boltzmann Weights
print("\nChecking Amplitudes (Boltzmann Factors)")

# State |00...0> (All Spins -1)
# Energy H = (-1)(-1) * (N-1) terms = N-1
# Expected Amplitude = exp(-beta * (N-1))
s_ferro = [0]*N
energy_ferro = (N - 1)
expected_amp_ferro = np.exp(-beta * energy_ferro)
amp_ferro = psi_ising.compute_amplitude(s_ferro)

print(f"State |00...0> (All -1):")
print(f"  Calculated Amp: {amp_ferro:.6f}")
print(f"  Expected Amp:   {expected_amp_ferro:.6f}")
print(f"  Match:          {np.isclose(amp_ferro, expected_amp_ferro)}")

# State |0101...> (Antiferromagnetic: -1, +1, -1, +1...)
# Energy H = (-1)(1) * (N-1) terms = -(N-1)
# Expected Amplitude = exp(-beta * -(N-1)) = exp(beta * (N-1))
s_af = [i%2 for i in range(N)]
energy_af = -(N - 1)
expected_amp_af = np.exp(-beta * energy_af)
amp_af = psi_ising.compute_amplitude(s_af)

print(f"State |0101...> (Alternating):")
print(f"  Calculated Amp: {amp_af:.6f}")
print(f"  Expected Amp:   {expected_amp_af:.6f}")

# Check Norm vs Partition Function
print("\nChecking Norm (Partition Function)")
# The Norm of Psi is sqrt(<Psi|Psi>)
# <Psi|Psi> = sum |exp(-beta H)|^2 = sum exp(-2*beta H) = Z(2*beta)
# For 1D Open Ising: Z(K) = 2 * (2 cosh(K))^(N-1)
# Here K = 2*beta

K = 2 * beta
Z_2beta = 2 * (2 * np.cosh(K))**(N-1)
expected_norm = np.sqrt(Z_2beta)

norm_can_ising = psi_ising.norm_canonical()
norm_gen_ising = psi_ising.norm_general()

print(f"Norm (Canonical): {norm_can_ising:.6f}")
print(f"Norm (General):   {norm_gen_ising:.6f}")
print(f"Expected Norm:    {expected_norm:.6f} (sqrt(Z(2beta)))")

if np.isclose(norm_can_ising, expected_norm):
    print(">> SUCCESS: Ising State Norm matches Partition Function.")
else:
    print(">> FAILURE: Norm mismatch.")



###############################################
# COMPRESSION TEST
###############################################

print(f"--- COMPRESSION TEST (N={N}) ---")

# Let us create a GHZ with Excess Memory: the state only need bond dim 2, but we give it 10
print("\nCreating GHZ with Bond Dim = 10...")
psi = MPS.create_ghz(N, d_phys=d_phys, b_dim=10)

amp_0 = psi.compute_amplitude([0]*N)
print(f"Original Amplitude |0...0>: {amp_0:.4f}")
print(f"Original Norm: {psi.norm_canonical():.4f}")

# 2. Compress to 2
print("\nCompressing to Bond Dim = 2...")
psi.compress(max_bond_dim=2)

amp_0_comp = psi.compute_amplitude([0]*N)
print(f"Compressed Amplitude |0...0>: {amp_0_comp:.4f}")
print(f"Compressed Norm: {psi.norm_canonical():.4f}")

if np.isclose(amp_0, amp_0_comp):
    print(">> SUCCESS: Compression to D=2 preserved the state.")
else:
    print(">> FAILURE: Compression damaged the state.")