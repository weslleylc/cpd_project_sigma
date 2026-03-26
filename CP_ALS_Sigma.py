import torch
import argparse
import os
import math
import cupy as cp
import numpy as np
import random
from torch.utils.dlpack import from_dlpack, to_dlpack
from cupyx.scipy.sparse.linalg import minres, LinearOperator
from pathlib import Path

# =============================================================================
# MATHEMATICAL DERIVATION FOR YOUR PROFESSOR (Equation 30)
# =============================================================================
"""
Goal: Minimize || Vec(K * Sigma^1/2) - P_T * Vec(U_T) ||^2

1. Definition of P_T (from Equation 30):
   P_T = [ (Sigma^1/2).T * M ] \otimes Id(T)
   where M = (U_S \odot U_H \odot U_W) is the Khatri-Rao product of fixed factors.

2. Deriving the Right-Hand Side (Vector 'b'):
   The standard Least Squares solution for Ax = b gives:
   b = (P_T).T * Vec(K * Sigma^1/2)

   Substitute P_T:
   b = [ (M.T * Sigma^1/2) \otimes Id(T) ] * Vec(K * Sigma^1/2)

3. Using the property (A \otimes B) Vec(C) = Vec(B * C * A.T):
   Let B = Id(T)
   Let C = K * Sigma^1/2
   Let A.T = M.T * Sigma^1/2  => A = (Sigma^1/2).T * M

   b = Vec( Id(T) * (K * Sigma^1/2) * (Sigma^1/2).T * M )

4. Resulting Simplification:
   Since (Sigma^1/2 * (Sigma^1/2).T) = Sigma (The full Covariance Matrix):
   b = Vec( K * Sigma * M )

Implementation Note:
We use the full Sigma instead of Sigma^1/2 because the "half" terms from 
the Target and the Projection Matrix merge during the derivation. 
This is numerically more stable than computing matrix square roots.
"""


# =============================================================================
# UTILS & REPRODUCIBILITY
# =============================================================================

def set_seed(seed):
    """Sets random seeds for reproducibility."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def torch_to_cupy(t):
    return cp.from_dlpack(t)


def cupy_to_torch(c):
    return from_dlpack(c)


# =============================================================================
# SUPPORT FUNCTIONS
# =============================================================================

def matvec_full_sigma(x_cp, A_core, dim_size, rank):
    """
    Generalized Matrix-Vector multiplication for Sigma-Aware systems.
    This applies the 'A' matrix logic for any mode.
    """
    x_torch = cupy_to_torch(x_cp).view(dim_size, rank).to(torch.float32)

    # A_core is the (Dim x Rank x Dim x Rank) interaction tensor
    # We contract the input vector across the last two dimensions
    res = torch.einsum('iajb,jb->ia', A_core, x_torch)

    return torch_to_cupy(res.flatten())

def reconstruct_tensor_from_factors(factors):
    """
    Reconstructs the CP model.
    Expected format: [In, H, W, Out]
    """
    U_S, U_H, U_W, U_T = factors
    return torch.einsum('sr,hr,wr,tr->shwt', U_S, U_H, U_W, U_T)


def calcul_err_sigma(tensor, sigma, tensor2):
    """
    Computes error in the Sigma functional norm.
    Expects tensors in shape [In, H, W, Out].
    """
    X_diff = tensor - tensor2
    # Reshape to [In*H*W, Out] to multiply by Sigma matrix [InHW, InHW]
    weighted_diff = torch.matmul(sigma, X_diff.reshape((-1, tensor.size()[3])))
    return torch.norm(weighted_diff)


def calcul_err(tensor, tensor2):
    """Standard Frobenius norm error."""
    return torch.norm(tensor - tensor2)


def matvec_M_cp(x_cp, Z_sigma_Z, out_c, rank):
    """Matrix-vector operator for the linear system A * u = b."""
    x_torch = cupy_to_torch(x_cp).view(out_c, rank).to(torch.float32)
    # The A matrix for Mode-T is effectively: Id(T) \otimes (M.T * Sigma * M)
    # Calculated as: U_T * Z_sigma_Z
    res = torch.matmul(x_torch, Z_sigma_Z)
    return torch_to_cupy(res.flatten())


# =============================================================================
# MAIN ALGORITHM
# =============================================================================

def cp_als_full_sigma(tensorT, rank, sigma_half, n_iter_max=100, tol=1e-6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. FORMATTING [In, H, W, Out]
    if len(tensorT.shape) == 3:
        out_c, in_c, hw = tensorT.shape
        h = w = int(math.sqrt(hw))
        tensor = torch.moveaxis(tensorT.view(out_c, in_c, h, w), 0, -1).to(device)
    else:
        tensor = torch.moveaxis(tensorT, 0, -1).to(device)

    in_c, h, w, out_c = tensor.shape
    sigma_half = sigma_half.to(device)
    sigma = torch.matmul(sigma_half, sigma_half.t())
    sigma_6d = sigma.view(in_c, h, w, in_c, h, w)

    # 2. INITIALIZATION
    factors = [torch.randn([in_c, rank], device=device),  # Mode 0: S
               torch.randn([h, rank], device=device),  # Mode 1: H
               torch.randn([w, rank], device=device),  # Mode 2: W
               torch.randn([out_c, rank], device=device)]  # Mode 3: T

    for f in factors: f /= (torch.norm(f, dim=0) + 1e-9)

    norm_sig_tensor = torch.norm(torch.matmul(sigma, tensor.reshape((-1, out_c))))
    errors = []

    # 3. THE FULL SIGMA LOOP
    for iteration in range(n_iter_max):
        for mode in [3, 0, 1, 2]:
            U_S, U_H, U_W, U_T = factors

            if mode == 3:  # Update Output Channels (T)
                # b = Vec( K * Sigma * (S_kr_H_kr_W) )
                K_sig = torch.matmul(sigma, tensor.reshape((-1, out_c))).view(in_c, h, w, out_c)
                b_target = torch.einsum('shwt,sr,hr,wr->tr', K_sig, U_S, U_H, U_W)

                # A_core = (S_kr_H_kr_W).T * Sigma * (S_kr_H_kr_W) -> Result: [Rank, Rank]
                # Because T is outside Sigma, A is diagonal in blocks (Identity_T @ A_core)
                A_core_small = torch.einsum('sr,hr,wr,shwabc,aq,bq,cq->rq', U_S, U_H, U_W, sigma_6d, U_S,
                                            U_H, U_W)

                # Optimization: For Mode T, A is simpler because T is independent
                def matvec_T(v_cp):
                    v_torch = cupy_to_torch(v_cp).view(out_c, rank)
                    return torch_to_cupy(torch.matmul(v_torch, A_core_small).flatten())

                A_op = LinearOperator((out_c * rank, out_c * rank), matvec=matvec_T, dtype=cp.float32)

            elif mode == 0:  # Update Input Channels (S)
                # This is the "Full Sigma" part. We contract everything except S.
                # b_target_S: [S, Rank]
                b_target = torch.einsum('shwt,shwabc,tr,hr,wr->sr', tensor, sigma_6d, U_T, U_H, U_W)

                # A_core_S: [S, Rank, S_prime, Rank_prime]
                A_core = torch.einsum('tr,tq,hr,wr,shwabc,hq,wq->s r a q', U_T, U_T, U_H, U_W, sigma_6d,
                                      U_H, U_W)
                A_op = LinearOperator((in_c * rank, in_c * rank),
                                      matvec=lambda v: matvec_full_sigma(v, A_core, in_c, rank),
                                      dtype=cp.float32)

            elif mode == 1:  # Update Height (H)
                b_target = torch.einsum('shwt,shwabc,tr,sr,wr->hr', tensor, sigma_6d, U_T, U_S, U_W)
                A_core = torch.einsum('tr,tq,sr,wr,shwabc,sq,wq->h r b q', U_T, U_T, U_S, U_W, sigma_6d,
                                      U_S, U_W)
                A_op = LinearOperator((h * rank, h * rank),
                                      matvec=lambda v: matvec_full_sigma(v, A_core, h, rank),
                                      dtype=cp.float32)

            elif mode == 2:  # Update Width (W)
                b_target = torch.einsum('shwt,shwabc,tr,sr,hr->wr', tensor, sigma_6d, U_T, U_S, U_H)
                A_core = torch.einsum('tr,tq,sr,hr,shwabc,sq,hq->w r c q', U_T, U_T, U_S, U_H, sigma_6d,
                                      U_S, U_H)
                A_op = LinearOperator((w * rank, w * rank),
                                      matvec=lambda v: matvec_full_sigma(v, A_core, w, rank),
                                      dtype=cp.float32)

            # SOLVE
            b_cp = torch_to_cupy(b_target.flatten())
            u_flat_cp, _ = minres(A_op, b_cp, tol=1e-9, maxiter=500)
            factors[mode] = cupy_to_torch(u_flat_cp).view(factors[mode].shape).to(torch.float32)

        # Balancing & Convergence
        rec = reconstruct_tensor_from_factors(factors)
        err = calcul_err_sigma(tensor, sigma, rec, out_c) / norm_sig_tensor
        errors.append(err.item())
        print(f"Iter {iteration + 1} | Full Sigma Error: {err:.6f}")
        if iteration > 0 and abs(errors[-2] - errors[-1]) < tol: break

    return [factors[3], factors[0], factors[1], factors[2]]


# =============================================================================
# EXECUTION
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sigma-Aware CP-ALS Decomposition")
    parser.add_argument('--tensor_path', type=str, required=True, help="Path to the kernel tensor .pt file")
    parser.add_argument('--sigma_path', type=str, required=True,
                        help="Path to the Sigma^1/2 covariance .pt file")
    parser.add_argument('--rank', type=int, required=True, help="Target CP rank")
    parser.add_argument('--n_iter_max', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    # Load weights and covariance root
    W = torch.load(args.tensor_path, weights_only=False)
    S_half = torch.load(args.sigma_path, weights_only=False)

    # Run decomposition
    factors = cp_als_sigma(W, args.rank, S_half, n_iter_max=args.n_iter_max)

    # Save output
    os.makedirs('outputs', exist_ok=True)
    clean_name = Path(args.tensor_path).stem
    save_path = f"outputs/cp_factors_{clean_name}_rank{args.rank}.pt"
    torch.save(factors, save_path)

    print(f"✅ Factors saved to: {save_path}")