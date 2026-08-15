using ITensors, ITensorMPS, Plots, LinearAlgebra

""" Boundary Contraction & Sampling """
# Note: this implementation does not allow to study frustated systems yet

# 1. Create a PEPS grid
function create_ising_peps(Lx::Int, Ly::Int, beta::Float64, J::Float64=1.0)
    s = [Index(2, "Site,s=$x,$y") for x in 1:Lx, y in 1:Ly]   # Physical spins
    h = [Index(2, "Link,h=$x,$y") for x in 1:Lx, y in 1:Ly-1] # Horizontal bonds
    v = [Index(2, "Link,v=$x,$y") for x in 1:Lx-1, y in 1:Ly] # Vertical bonds
    
    Q = [exp(beta*J) exp(-beta*J); 
         exp(-beta*J) exp(beta*J)]
    evals, evecs = eigen(Q)
    M = evecs * diagm(sqrt.(evals)) * transpose(evecs)
    
    A = Matrix{ITensor}(undef, Lx, Ly)
    for x in 1:Lx, y in 1:Ly
        # Collect only the bounds that exist for this specific (x,y) coordinate
        inds_xy = Index[s[x,y]]
        if y > 1  push!(inds_xy, h[x, y-1]) end
        if y < Ly push!(inds_xy, h[x, y])   end
        if x > 1  push!(inds_xy, v[x-1, y]) end
        if x < Lx push!(inds_xy, v[x, y])   end
        
        T = ITensor(inds_xy...)
        
        for spin in 1:2
            vL = y > 1 ? M[spin, :] : [1.0]
            vR = y < Ly ? M[spin, :] : [1.0]
            vU = x > 1 ? M[spin, :] : [1.0]
            vD = x < Lx ? M[spin, :] : [1.0]
            
            # Populate tensor by iterating over all combinations of the existing bounds
            for iL in eachindex(vL), iR in eachindex(vR), iU in eachindex(vU), iD in eachindex(vD)
                val = vL[iL] * vR[iR] * vU[iU] * vD[iD]
                
                assign = Pair{Index, Int}[s[x,y] => spin]
                if y > 1  push!(assign, h[x, y-1] => iL) end
                if y < Ly push!(assign, h[x, y] => iR)   end
                if x > 1  push!(assign, v[x-1, y] => iU) end
                if x < Lx push!(assign, v[x, y] => iD)   end
                
                T[assign...] = val
            end
        end
        A[x, y] = T
    end
    return A, s, v # Return vertical indices so we can track them
end

# 2. Bottom environment pre-computation
function compute_bottom_envs(A::Matrix{ITensor}, s::Matrix{Index{Int64}}, v::Matrix{Index{Int64}}, Lx::Int, Ly::Int, D_bound::Int)
    bottom_envs = Vector{Union{MPS, Nothing}}(undef, Lx)
    bottom_envs[Lx] = nothing

    b = [Index(2, "Site,b=$y") for y in 1:Ly]
    
    # Base case: Row Lx (Trace out physical spins to form an MPS)
    tensors_Lx = ITensor[]
    for y in 1:Ly
        # Multiplying by [1.0, 1.0] sums over the physical 's' index natively
        T_traced = A[Lx, y] * ITensor([1.0, 1.0], s[Lx, y])
        T_traced = replaceinds(T_traced, [v[Lx-1, y]] => [b[y]])
        push!(tensors_Lx, T_traced)
    end
    curr_mps = MPS(tensors_Lx)

    # Save, reverting dummy b back to vertical v
    bottom_envs[Lx] = MPS([replaceinds(curr_mps[y], b[y] => v[Lx-1, y]) for y in 1:Ly])

    # Sweep upwards to build the remaining environments
    for x in Lx-1:-1:2
        tensors_mpo = ITensor[]
        for y in 1:Ly
            T_traced = A[x, y] * ITensor([1.0, 1.0], s[x, y])
            # v_bottom -> b (to connect to curr_mps). v_top -> b' (new open leg)
            T_traced = replaceinds(T_traced, [v[x, y], v[x-1, y]] => [b[y], b[y]'])
            push!(tensors_mpo, T_traced)
        end
        MPO_x = MPO(tensors_mpo)
        
        # Apply row MPO to bottom environment and compress
        curr_mps = apply(MPO_x, curr_mps; maxdim=D_bound, cutoff=1e-10)
        curr_mps = noprime(curr_mps) # Strip the prime
        normalize!(curr_mps)

        # Save
        bottom_envs[x] = MPS([replaceinds(curr_mps[y], b[y] => v[x-1, y]) for y in 1:Ly])
    end
    return bottom_envs
end

# 3. 1D classical sampler
function sample_classical_1d(row_mps::MPS, s_inds::Vector{Index{Int64}})
    Ly = length(row_mps)
    sampled_spins = zeros(Int, Ly)
    log_prob_row = 0.0
    
    # Precompute Right environments (sweeping right to left)
    R = Vector{ITensor}(undef, Ly)
    temp = ITensor(1.0)
    for y in Ly:-1:1
        T_traced = row_mps[y] * ITensor([1.0, 1.0], s_inds[y])
        temp *= T_traced
        n = norm(temp)
        if n > 0 temp ./= n end
        R[y] = temp
    end
    
    # Sample Left to Right
    L_env = ITensor(1.0)
    for y in 1:Ly
        T = row_mps[y]
        
        # Calculate classical weight for spin 0 (Up -> Index 1)
        proj0 = ITensor([1.0, 0.0], s_inds[y])
        w0_tensor = L_env * (T * proj0) * (y < Ly ? R[y+1] : ITensor(1.0))
        w0 = max(0.0, scalar(w0_tensor))
        
        # Calculate classical weight for spin 1 (Down -> Index 2)
        proj1 = ITensor([0.0, 1.0], s_inds[y])
        w1_tensor = L_env * (T * proj1) * (y < Ly ? R[y+1] : ITensor(1.0))
        w1 = max(0.0, scalar(w1_tensor))
        
        total_w = w0 + w1
        p0 = total_w < 1e-15 ? 0.5 : w0 / total_w
        
        # Choose spin based on raw classical probability
        chosen_spin = rand() < p0 ? 0 : 1
        sampled_spins[y] = chosen_spin
        
        p_chosen = chosen_spin == 0 ? p0 : (1.0 - p0)
        log_prob_row += p_chosen > 1e-15 ? log(p_chosen) : -Inf
        
        # Update Left Environment for next iteration
        chosen_proj = chosen_spin == 0 ? proj0 : proj1
        L_env *= (T * chosen_proj)
        n = norm(L_env)
        if n > 0 L_env ./= n end
    end
    
    return sampled_spins, log_prob_row
end

# 4. Full 2D sampler
function sample_config_opt(Lx::Int, Ly::Int, beta::Float64, D_bound::Int, J::Float64=1.0)
    A, s_inds, v_inds = create_ising_peps(Lx, Ly, beta, J)
    bottom_envs = compute_bottom_envs(A, s_inds, v_inds, Lx, Ly, D_bound)

    top_env = nothing
    b = [Index(2, "Site,b=$y") for y in 1:Ly]
    
    current_config = zeros(Int, Lx, Ly)
    log_prob_tot = 0.0
    
    for x in 1:Lx
        # 1. Sandwich row x to create effective 1D MPS
        eff_tensors = ITensor[]
        for y in 1:Ly
            T = A[x, y]
            if top_env !== nothing 
                # Temporarily revert top_env dummy 'b' to actual vertical index
                T_top = replaceinds(top_env[y], b[y] => v_inds[x-1, y])
                T *= T_top
            end
            if x < Lx T *= bottom_envs[x+1][y] end
            push!(eff_tensors, T)
        end
        row_mps = MPS(eff_tensors)
        
        # 2. Sample classical 1D
        sampled_spins, log_p_row = sample_classical_1d(row_mps, s_inds[x, :])
        current_config[x, :] = sampled_spins
        log_prob_tot += log_p_row
        
        # 3. Project physical spins to create a fixed MPO
        mpo_tensors = ITensor[]
        for y in 1:Ly
            spin_val = sampled_spins[y]
            # Convert 0/1 back to projection vectors
            proj = ITensor(spin_val == 0 ? [1.0, 0.0] : [0.0, 1.0], s_inds[x, y])
            push!(mpo_tensors, A[x, y] * proj)
        end
        
        # 4. Push Top Environment Down
        if top_env === nothing
            top_tensors = ITensor[]
            for y in 1:Ly
                T = replaceinds(mpo_tensors[y], [v_inds[1, y]] => [b[y]])
                push!(top_tensors, T)
            end
            top_env = MPS(top_tensors)
            normalize!(top_env)
        else
            if x < Lx
                mpo_tensors_replaced = ITensor[]
                for y in 1:Ly
                    T = mpo_tensors[y]
                    T = replaceinds(T, [v_inds[x-1, y], v_inds[x, y]] => [b[y], b[y]'])
                    push!(mpo_tensors_replaced, T)
                end
                MPO_fixed = MPO(mpo_tensors_replaced)
                
                top_env = apply(MPO_fixed, top_env; maxdim=D_bound, cutoff=1e-10)
                top_env = noprime(top_env)
                normalize!(top_env)
            end
        end
    end
    
    # Returns the 2D grid of 0s and 1s, exactly like your Python code
    return current_config, log_prob_tot
end


""" TNMH MCMC Sweep """

function measure_energy(config::Matrix{Int}, J::Float64=1.0)
    # Convert 0/1 configuration array to +1/-1 spins
    # Using broadcast: 1 .- 2*(0) = 1, 1 .- 2*(1) = -1
    spins = 1 .- 2 .* config
    
    # Calculate horizontal bonds (multiply each column by the column to its right)
    horiz_pairs = spins[:, 1:end-1] .* spins[:, 2:end]
    
    # Calculate vertical bonds (multiply each row by the row below it)
    vert_pairs = spins[1:end-1, :] .* spins[2:end, :]
    
    # Total energy is the sum of all bond interactions
    return -J * (sum(horiz_pairs) + sum(vert_pairs))
end

function run_mcmc_sweep(Lx::Int, Ly::Int, beta::Float64, D_bound::Int, N_samples::Int)
    J = 1.0
    
    # propose initial state
    curr_config, curr_log_prob = sample_config_opt(Lx, Ly, beta, D_bound, J)
    curr_energy = measure_energy(curr_config, J)
    
    acceptance_count = 0
    energy_history = Float64[]
    
    for step in 1:N_samples
        # propose new state
        new_config, new_log_prob = sample_config_opt(Lx, Ly, beta, D_bound, J)
        new_energy = measure_energy(new_config, J)
        
        # Metropolis-Hastings filter
        log_TN_ratio = curr_log_prob - new_log_prob
        log_boltzmann_ratio = -beta * (new_energy - curr_energy)
        log_acceptance = log_TN_ratio + log_boltzmann_ratio
        
        # Accept/Reject
        if log(rand()) < log_acceptance
            curr_config = new_config
            curr_log_prob = new_log_prob
            curr_energy = new_energy
            acceptance_count += 1
        end
        
        push!(energy_history, curr_energy / (Lx * Ly))
    end
    
    burn_in = div(N_samples, 5) # 20% burn-in
    acc_rate = acceptance_count / N_samples
    
    history_kept = energy_history[burn_in+1:end]
    avg_e = sum(history_kept) / length(history_kept)
    
    return acc_rate, avg_e
end