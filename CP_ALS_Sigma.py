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
    # Calculated as: U_T * Z_sigma_Z
    res = torch.matmul(x_torch, Z_sigma_Z)
    return torch_to_cupy(res.flatten())


# =============================================================================
# MAIN ALGORITHM
# =============================================================================

def cp_als_sigma(tensorT, rank, sigma_half, n_iter_max=100, tol=1e-6, verbose=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. PREPARATION (Move Output axis to the end for Sigma alignment)
    if len(tensorT.shape) == 3:
        out_c, in_c, hw = tensorT.shape
        h = w = int(math.sqrt(hw))
        # [Out, In, H, W] -> [In, H, W, Out]
        tensor = torch.moveaxis(tensorT.view(out_c, in_c, h, w), 0, -1).to(device)
    else:
        tensor = torch.moveaxis(tensorT, 0, -1).to(device)

    tsize = tensor.size()
    in_c, h, w, out_c = tsize

    # Convert Sigma_half (Σ^{1/2}) to full Sigma (Σ) for the ALS update
    sigma_half = sigma_half.to(device)
    sigma = torch.matmul(sigma_half, sigma_half.t())

    # Calculate normalization factor for relative error
    norm_sigma_tensor = torch.norm(torch.matmul(sigma, tensor.reshape((-1, out_c))))

    # 2. INITIALIZATION
    # Shape: [Dimension, Rank]
    factors = [torch.randn([in_c, rank], device=device),  # U_S
               torch.randn([h, rank], device=device),  # U_H
               torch.randn([w, rank], device=device),  # U_W
               torch.randn([out_c, rank], device=device)]  # U_T

    # Initial unit normalization
    for f in factors:
        f /= (torch.norm(f, dim=0, keepdim=True) + 1e-9)

    rec_errors = [
        calcul_err_sigma(tensor, sigma, reconstruct_tensor_from_factors(factors)) / norm_sigma_tensor]

    # 3. ALS LOOP
    for iteration in range(n_iter_max):
        # Update sequence: Mode 3 (Target/Output), then S, H, W
        for mode in [3, 0, 1, 2]:

            if mode == 3:
                # --- SIGMA-AWARE UPDATE (U_T) ---
                U_S, U_H, U_W = factors[0], factors[1], factors[2]

                # --- SIGMA-AWARE UPDATE (U_T) ---
                # Math: b = Vec( K * Sigma * M )

                # 1. Flatten the PyTorch tensor to match mathematical K
                # Shape: [T, S*H*W]. We use reshape() to handle non-contiguous memory safely.
                K = tensor.reshape(out_c, -1)

                # 2. Apply Sigma exactly as the formula says: K * Sigma
                # Shape: [T, S*H*W] @ [S*H*W, S*H*W] -> [T, S*H*W]
                K_sigma = torch.matmul(K, sigma)

                # 3. Reshape back to 4D so we can project the factors
                # Shape: [T, S, H, W]
                K_sigma_tensor = K_sigma.reshape(out_c, in_c, h, w)

                # 4. Multiply by M (Projection onto the rank space)
                # Math: einsum(Target_K_Sigma, U_S, U_H, U_W)
                # Shape: [T, R], T=out_c
                b_target = torch.einsum('tshw,sr,hr,wr->tr', K_sigma_tensor, U_S, U_H, U_W)

                # 5. Vectorize (Vec)
                b_cp = torch_to_cupy(b_target.flatten())

                # A = M.T * Sigma * M (Size: Rank x Rank)
                sigma_6d = sigma.view(in_c, h, w, in_c, h, w)
                # This performs the projection of the 6D Sigma tensor onto the rank space
                MT_sigma_M = torch.einsum('sr,hr,wr,shwabc,aq,bq,cq->rq',
                                         U_S, U_H, U_W, sigma_6d, U_S, U_H, U_W)

                # Solve Ax = b using Matrix-Free MINRES
                n_params = out_c * rank
                A_op = LinearOperator((n_params, n_params),
                                      matvec=lambda v: matvec_M_cp(v, MT_sigma_M, out_c, rank),
                                      dtype=cp.float32)

                u_flat_cp, _ = minres(A_op, b_cp, tol=1e-10, maxiter=500)
                factors[3] = cupy_to_torch(u_flat_cp).view(out_c, rank).to(torch.float32)

            else:
                # --- STANDARD ALS UPDATE (Modes S, H, W) ---
                # These modes follow standard Frobenius CP-ALS logic
                others = [f for i, f in enumerate(factors) if i != mode]

                # Compute Khatri-Rao Product Z
                Z = others[0]
                for f in others[1:]:
                    Z = (Z.unsqueeze(1) * f.unsqueeze(0)).reshape(-1, rank)

                # Solve normal equation with Tikhonov regularization
                M_reg = torch.t(Z) @ Z + 1e-6 * torch.eye(rank, device=device)
                W_m = tensor.permute(mode, *[i for i in range(4) if i != mode]).reshape(tsize[mode], -1)
                factors[mode] = torch.t(torch.linalg.solve(M_reg, torch.t(Z) @ torch.t(W_m)))

        # 4. SCALE BALANCING (Geometric Mean normalization)
        norms = [torch.norm(f, dim=0) for f in factors]
        gm_alpha = torch.pow(torch.prod(torch.stack(norms), dim=0), 1.0 / 4.0)
        for n in range(4):
            factors[n] *= (gm_alpha / (norms[n] + 1e-9))

        # 5. CONVERGENCE CHECKING
        current_rec = reconstruct_tensor_from_factors(factors)
        err_sig = calcul_err_sigma(tensor, sigma, current_rec) / norm_sigma_tensor
        err_frob = calcul_err(tensor, current_rec)
        rel_frob = err_frob / torch.norm(tensor)

        rec_errors.append(err_sig.item())

        if verbose:
            print(
                f"Iter {iteration + 1} | Sigma Err: {err_sig:.6f} | Rel Frob Err: {rel_frob:.6f} | Delta: {rec_errors[-2] - rec_errors[-1]:.4e}")

        if iteration > 0 and (rec_errors[-2] - rec_errors[-1]) < tol:
            if verbose: print("Convergence reached.")
            break

    # Revert to original PyTorch format [Out, In, H, W]
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