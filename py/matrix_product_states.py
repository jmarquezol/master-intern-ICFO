import numpy as np

class MPS:

    def __init__(self, N, d_phys, A):
        """
        initialize MPS object
        
        :param self: -
        :param N: number of sites
        :param d_phys: physical dimension (leg)
        :param A: list of tensors: A[n].shape = (d_phys, b_dim_left, b_dim_right)
        """

        self.N, self.d_phys = N, d_phys
        self.A = [item for item in A] 

        self.left_canonical_form()
        self.canonical_center = self.N-1
        
    def left_canonical_form(self):
        """
        Converts the MPS into left canonical form using QR decomposition
        """

        for n in range(self.N-1):

            shape = self.A[n].shape

            # QR decomposition: Q is an orthogonal matrix, R is an upper triangular matrix
            # reshape A[n] into a matrix of shape (d_phys*b_dim_left, b_dim_right)
            q, r = np.linalg.qr(self.A[n].reshape(shape[0]*shape[1], shape[2]), 
                                mode = 'reduced')
            # shape of q is (d * b_dim_left, K)
            # shape of r is (K, b_dim_right)
            # K = b_dim_right

            self.A[n] = q.reshape(shape[0], shape[1], -1) # store Q back in A[n] in the original shape (with updated b_dim_right represented by "-1")
            self.A[n+1] = np.tensordot(self.A[n+1], r, axes = (1, 1)) # contract (right leg of) r and (left leg of) A[n+1]; axes=( [legs_of_A], [legs_of_r] )
            self.A[n+1] = np.transpose(self.A[n+1], axes = (0, 2, 1)) # shape adjustment after contraction wih tesor product

        self.canonical_center = self.N-1 # Every tensor from 0 to N-2 is a "Left Isometry" (A[n]^\dagger A[n] = I), while A[N-1] holds the normalization information of the wf

    def compute_amplitude(self, idx):
        """
        Probability amplitude coefficient for a given configuration idx (for a specific computational basis state)

        Procedure: take the tensor at site 0, slice it with idx[0] and shape (d, 1, R) -> (1, R) row vector
        then itereate for every next site n until the last tensor: (1, R) @ (D, D) @ ... @ (D, 1) -> scalar
        
        :param idx: list of integers of length N representing the state at each site
        """

        if len(idx) != self.N:
            raise ValueError("Length of idx must be equal to N")
        
        res = self.A[0][idx[0]] # shape (1, R)

        for n in range(1, self.N): # loop from site 1 to N-1
            matrix = self.A[n][idx[n]]
            
            # Matrix multiplication: (1, D) @ (D, D) -> (1, D)
            res = res @ matrix
        
        return res[0,0]


    def norm_canonical(self):
        """
        Computes normm using the canonical form of the MPS (valid only if the MPS is in right canonical form)
        """

        last_tensor = self.A[-1] # last tensor in the MPS where all normalization is stored

        return np.linalg.norm(last_tensor.flatten())
    
    def norm_general(self):
        """
        Computes norm by contracting the full network from left to right for a general MPS
        """

        env = np.eye(1) # initial environment (scalar 1 represented as a 1x1 identity matrix with shape (bra_L, ket_L))

        for n in range(self.N):
            A = self.A[n]
            # both with shape: (d, L, R)

            # 1st step: contract env with ket A over the left bond
            temp = np.tensordot(env, self.A[n], axes = (1, 1)) # shape: (bra_L, d, R) after contraction

            # 2nd step: contract the result with bra A_conj over physical index and left bond
            env = np.tensordot(self.A[n].conj(), temp, axes = ([0,1], [1,0])) # shape: (R_bra, R_ket) after contraction
        return np.sqrt(np.real(env[0,0]))

    def normalize_tensors(self):
        """
        Divides each tensor by its maximum absolute value.
        """
        for n in range(self.N):
            norm_factor = np.max(np.abs(self.A[n]))
            if norm_factor > 0:
                self.A[n] /= norm_factor
    
    def expectation_value(self, op, site_idx):
        """
        Computes the expectation value <Psi|op|Psi> at a single site.
        
        :param op: Single site operator (d x d matrix)
        :param site_index: Index of the site to apply the operator
        """
        env = np.eye(1) # initial environment

        for n in range(self.N):
            tensor = self.A[n] # Shape (d, L, R)
            
            # If this is the target site, we apply the operator to the ket
            if n == site_idx:
                # Contract op with physical leg of tensor
                # op is (d, d), tensor is (d, L, R) -> result (d, L, R)
                tensor = np.tensordot(op, tensor, axes=(1, 0))
            
            # Standard contraction (same as norm_general)
            # 1. Contract env with ket tensor
            temp = np.tensordot(env, tensor, axes=(1, 1)) # (bra_L, d, R)
            
            # 2. Contract with bra tensor (conjugate)
            env = np.tensordot(self.A[n].conj(), temp, axes=([0, 1], [1, 0])) # (R_bra, R_ket)

        return np.real(env[0, 0])

    def apply_mpo(self, mpo):
        """
        Applies an MPO to the current MPS state: |Psi'> = O |Psi>
        Returns a NEW MPS object
        
        mpo: list of Rank-4 tensors with shape: (d_out, d_in, b_dim_L_mpo, b_dim_R_mpo)
        """
        if len(mpo) != self.N:
            raise ValueError("MPO length must match MPS length")

        new_A = []

        for n in range(self.N):
            # MPS Tensor A[n]: (d_in, L, R)
            # MPO Tensor W[n]: (d_out, d_in, L_w, R_w) where convection is: group physical legs first, then bond legs
            
            # 1st connect MPO axis 1 (d_in) with MPS axis 0 (d_in)
            # result shape: (d_out, L_w, R_w, L, R)
            temp = np.tensordot(mpo[n], self.A[n], axes=([1], [0]))
            
            # 2nd transpose to group lefts and rights legs together
            # result shape: (d_out, L_w, L, R_w, R)
            temp = np.transpose(temp, (0, 1, 3, 2, 4))
            shape = temp.shape
            
            # 3rd fuse the resulting bond indices
            # New Left Bond Dim = L_w * L
            # New Right Bond Dim = R_w * R
            # final shape: (d_out, New_L, New_R)
            new_tensor = temp.reshape(shape[0], shape[1]*shape[2], shape[3]*shape[4])

            new_A.append(new_tensor)

        return MPS(self.N, self.d_phys, new_A)
    
    def compress(self, max_bond_dim):
        """
        Compresses the MPS to a maximum bond dimension using SVD.
        Sweeps Right -> Left (N-1 to 0).
        """
        # 1st start with QR decomposition (weights are at the right end)
        self.left_canonical_form() 
        
        # 2nd iterate backwards from N-1 down to 1 (site 0 doesn't need compression since we are compressing the left bonds)
        for n in range(self.N - 1, 0, -1):

            # A[n] shape: (d, L, R), and we want to compress the left bond L
            # Reshape to Matrix M: Rows=L, Cols=(d, R)    +     Transpose (d, L, R) -> (L, d, R)
            temp = np.transpose(self.A[n], (1, 0, 2))
            shape = temp.shape # (L, d, R)
            matrix = temp.reshape(shape[0], shape[1] * shape[2])
            
            # 3rd SVD decomposition: M = U * S * Vh
            U, s, Vh = np.linalg.svd(matrix, full_matrices=False)
            
            # 4th Truncate
            dim_keep = min(len(s), max_bond_dim)
            
            U_trunc = U[:, :dim_keep]        # Shape: (L, dim_keep)
            s_trunc = s[:dim_keep]           # Shape: (dim_keep)
            Vh_trunc = Vh[:dim_keep, :]      # Shape: (dim_keep, d*R)
            
            # 5th Update A[n] with Vh, and reshape back to tensor form -> (dim_keep, d, R) -> Transpose to (d, dim_keep, R)
            self.A[n] = Vh_trunc.reshape(dim_keep, shape[1], shape[2]).transpose(1, 0, 2)
            
            # 6th Push T = U_trunc * s_trunc (which has shape (L, dim_keep)) to the left neighbor A[n-1] with shape (d, L_prev, R_prev) 
            T = U_trunc * s_trunc[None, :] 
            
            # Note that here R_prev == L, so we contract over that axis
            self.A[n-1] = np.tensordot(self.A[n-1], T, axes=([2], [0]))
        
        self.canonical_center = 0 # now the center of orthogonality is at site 0



    
    @classmethod
    def create_ghz(cls, N, d_phys=2, b_dim=2):
        """
        GHZ state |00...0> + |11...1>
        
        :param N: Number of sites
        :param d_phys: Physical dimension (default is 2: spin up and down)
        :param b_dim: Bond dimension
        """
        if b_dim < 2:
            raise ValueError("Bond dimension must be at least 2 for GHZ state.") # Note: if b_dim=1, we can only represent product states

        # Left Tensor: (d_phys, 1, b_dim) which acts as a row vector (1, b_dim) after "slicing"
        A_L = np.zeros((d_phys, 1, b_dim))
        A_L[0, 0, 0] = 1.0 # particle in state 0 has input (left) 0 and output (right) 0
        A_L[1, 0, 1] = 1.0 # particle in state 1 has input (left) 0 by default and output (right) 1

        # Bulk Tensor: (d_phys, b_dim, b_dim) which acts as a matrix (b_dim, b_dim) after "slicing"
        A_B = np.zeros((d_phys, b_dim, b_dim))
        A_B[0, 0, 0] = 1.0 # particle in state 0 has input (left) 0 and output (right) 0
        A_B[1, 1, 1] = 1.0 # particle in state 1 has input (left) 1 and output (right) 1

        # Right Tensor: (d_phys, b_dim, 1) which acts as a column vector (b_dim, 1) after "slicing"
        A_R = np.zeros((d_phys, b_dim, 1))
        A_R[0, 0, 0] = 1.0 # particle in state 0 has input (left) 0 and output (right) 0
        A_R[1, 1, 0] = 1.0 # particle in state 1 has input (left) 1 and output (right) 0 (final of the chain)

        A_list = [A_L.copy()] + [A_B.copy() for _ in range(N-2)] + [A_R.copy()]

        return cls(N, d_phys, A_list)
    
    @classmethod
    def create_w_state(cls, N, d_phys=2, b_dim=2):
        """
        W-State: |100..> + |010..> + ... + |00..1>
        Represents a superposition of a single excitation.
        """
        if b_dim < 2:
            raise ValueError("Bond dimension must be at least 2 for W state.")
        
        # Normalization
        norm_factor = 1.0 / np.sqrt(N)

        # Left Tensor (by default, input is 0)
        A_L = np.zeros((d_phys, 1, b_dim))
        A_L[0, 0, 0] = 1.0 * norm_factor # Phys 0 -> Output 0 (no excitation yet)
        A_L[1, 0, 1] = 1.0 * norm_factor # Phys 1 -> Output 1 (excitation placed here)
        # note: norm factor position is arbitrary, can be placed in any tensor

        # Bulk Tensor
        A_B = np.zeros((d_phys, b_dim, b_dim))
        # Case A: Input is 0 (No excitation yet)
        A_B[0, 0, 0] = 1.0 # Phys 0 -> Output 0 (still no excitation)
        A_B[1, 0, 1] = 1.0 # Phys 1 -> Output 1 (excitation placed here)
        
        # Case B: Input is 1 (Excitation already exists)
        A_B[0, 1, 1] = 1.0 # Phys 0 -> Output 1 (passing it along)
        # Note: A_B[1, 1, ?] is 0.0 because we can't have two excitations.

        # Right Tensor
        A_R = np.zeros((d_phys, b_dim, 1))
        # Input 0: we reached the end but have 0 excitations, so we must pick as the right tensor |1>
        A_R[1, 0, 0] = 1.0 
        # Input 1: we already have an excitation, so we must pick |0> to avoid double excitation
        A_R[0, 1, 0] = 1.0

        A_list = [A_L.copy()] + [A_B.copy() for _ in range(N-2)] + [A_R.copy()]
        return cls(N, d_phys, A_list)
    
    @classmethod
    def create_ising_state(cls, N, beta, d_phys=2, b_dim=2):
        """
        It created a state encoding the classical 1D Ising model Boltzmann weights.
        Psi = sum_{sigma} exp(-beta * H(sigma)) |sigma>
        H(sigma) = sum_{i} sigma_i * sigma_{i+1}  (sigma in {-1, +1} mapping to indices {0, 1})
        """
        if b_dim < 2:
            raise ValueError("Bond dimension must be at least 2 for Ising state.")

        # convert index 0/1 to spin -1/+1
        def get_spin(idx):
            return 1.0 if idx == 1 else -1.0

        # Left Tensor
        # No previous interaction, just set the outgoing bond to current spin
        A_L = np.zeros((d_phys, 1, b_dim))
        for s in range(d_phys):
            # s is current physical spin. Outgoing bond becomes s: if s=0 -> bond=0 (spin -1), if s=1 -> bond=1 (spin +1)
            # Weight is 1.0 (boundary condition) because no previous spin to interact with
            A_L[s, 0, s] = 1.0

        # Bulk Tensor
        # interaction between previous (left bond) and current (phys bond)
        A_B = np.zeros((d_phys, b_dim, b_dim))
        for s in range(d_phys):         # Current Spin (Physical)
            for prev in range(d_phys):  # Previous Spin (Left Bond)
                # Outgoing bond MUST match Current Spin to propagate history to the next site
                spin_curr = get_spin(s)
                spin_prev = get_spin(prev)
                weight = np.exp(-beta * spin_prev * spin_curr)
                
                A_B[s, prev, s] = weight # we add the weight if input bond = prev spin and output bond = current spin

        # Right Tensor
        # interaction between previous (left bond) and current (phys bond). Close bond.
        A_R = np.zeros((d_phys, b_dim, 1))
        for s in range(d_phys):
            for prev in range(d_phys):
                spin_curr = get_spin(s)
                spin_prev = get_spin(prev)
                weight = np.exp(-beta * spin_prev * spin_curr)
                
                A_R[s, prev, 0] = weight

        A_list = [A_L.copy()] + [A_B.copy() for _ in range(N-2)] + [A_R.copy()]
        return cls(N, d_phys, A_list)
    

    @staticmethod
    def create_pauli_x_mpo(N, d_phys=2):
        """
        Creates an MPO representing the global Pauli-X operator: X^{\otimes N}

        This MPO is a product operator that acts as the Pauli-X operator on each site independently.
        :param N: Number of sites
        """
        sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]])
        W = np.zeros((d_phys, d_phys, 1, 1)) # shape: (d_out, d_in, b_dim_L, b_dim_R), where bond dims are 1 for product operators (no entanglement between sites)
        W[:, :, 0, 0] = sigma_x
        return [W.copy() for _ in range(N)]