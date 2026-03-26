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
# UTILITÁRIOS E REPRODUTIBILIDADE
# =============================================================================

def set_seed(seed):
    """Configura sementes para reprodutibilidade[cite: 1209]."""
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
# FUNÇÕES DE APOIO
# =============================================================================

def reconstruct_tensor_from_factors(factors):
    r"""
    Reconstrói o modelo CP[cite: 14, 1302].
    Formato após moveaxis: [In, H, W, Out]
    """
    U_S, U_H, U_W, U_T = factors
    return torch.einsum('sr,hr,wr,tr->shwt', U_S, U_H, U_W, U_T)


def calcul_err_sigma(tensor, sigma, tensor2):
    r"""
    Erro na norma funcional Sigma[cite: 468, 586, 918].
    Expects [In, H, W, Out]
    """
    X_diff = tensor - tensor2
    # Reshape para [In*H*W, Out] para alinhar com a matriz Sigma
    return torch.norm(torch.matmul(sigma, X_diff.reshape((-1, tensor.size()[3]))))

def calcul_err(tensor, tensor2):
    return torch.norm(tensor - tensor2)

def matvec_M_cp(x_cp, Z_sigma_Z, out_c, rank):
    """Operador de matriz-vetor para o sistema linear AU = b[cite: 997]."""
    x_torch = cupy_to_torch(x_cp).view(out_c, rank).to(torch.float32)
    # A atualização do Modo-0 é dada por: U_T * (Z^T Sigma Z)
    res = torch.matmul(x_torch, Z_sigma_Z)
    return torch_to_cupy(res.flatten())


# =============================================================================
# ALGORITMO PRINCIPAL
# =============================================================================

def cp_als_sigma(tensorT, rank, sigma_half, n_iter_max=100, tol=1e-6, verbose=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. PREPARAÇÃO (Lógica do autor: move Output axis para o final)
    if len(tensorT.shape) == 3:
        out_c, in_c, hw = tensorT.shape
        h = w = int(math.sqrt(hw))
        # [Out, In, H, W] -> [In, H, W, Out]
        tensor = torch.moveaxis(tensorT.view(out_c, in_c, h, w), 0, -1).to(device)
    else:
        tensor = torch.moveaxis(tensorT, 0, -1).to(device)

    tsize = tensor.size()
    in_c, h, w, out_c = tsize

    # Sigma half (Σ^{1/2}) para Sigma total (Σ) [cite: 585, 905]
    sigma_half = sigma_half.to(device)
    sigma = torch.matmul(sigma_half, sigma_half.t())

    norm_sigma_tensor = torch.norm(torch.matmul(sigma, tensor.reshape((-1, out_c))))

    # 2. INICIALIZAÇÃO (Eq. 324)
    # Importante: [Dimensão, Rank] para evitar erro de broadcast no einsum
    factors = [torch.randn([in_c, rank], device=device),  # U_S
               torch.randn([h, rank], device=device),  # U_H
               torch.randn([w, rank], device=device),  # U_W
               torch.randn([out_c, rank], device=device)]  # U_T

    # Normalização de unidade inicial [cite: 334]
    for f in factors:
        f /= (torch.norm(f, dim=0, keepdim=True) + 1e-9)

    rec_errors = [
        calcul_err_sigma(tensor, sigma, reconstruct_tensor_from_factors(factors)) / norm_sigma_tensor]

    # 3. ALS LOOP
    for iteration in range(n_iter_max):
        # Sweeping: Modo 3 (Target/Output), então S, H, W
        for mode in [3, 0, 1, 2]:

            if mode == 3:
                # --- ATUALIZAÇÃO MODO-SIGMA (U_T) ---
                U_S, U_H, U_W = factors[0], factors[1], factors[2]

                # Cálculo de b = (P^T)^T * Vec(K * Sigma) [cite: 999]
                K_sigma = torch.matmul(sigma, tensor.reshape((-1, out_c)))
                b_target = torch.einsum('sr,hr,wr,shwt->tr', U_S, U_H, U_W,
                                        K_sigma.view(in_c, h, w, out_c))
                b_cp = torch_to_cupy(b_target.flatten())

                # Cálculo de A = Z^T Sigma Z (R x R)
                sigma_6d = sigma.view(in_c, h, w, in_c, h, w)
                Z_sigma_Z = torch.einsum('sr,hr,wr,shwabc,aq,bq,cq->rq',
                                         U_S, U_H, U_W, sigma_6d, U_S, U_H, U_W)

                # Solver MINRES Matrix-Free [cite: 997, 1006]
                n_params = out_c * rank
                A_op = LinearOperator((n_params, n_params),
                                      matvec=lambda v: matvec_M_cp(v, Z_sigma_Z, out_c, rank),
                                      dtype=cp.float32)

                u_flat_cp, _ = minres(A_op, b_cp, tol=1e-10, maxiter=500)
                factors[3] = cupy_to_torch(u_flat_cp).view(out_c, rank).to(torch.float32)

            else:
                # --- ATUALIZAÇÃO ALS PADRÃO (S, H, W) ---
                others = [f for i, f in enumerate(factors) if i != mode]
                Z = others[0]
                for f in others[1:]:
                    Z = (Z.unsqueeze(1) * f.unsqueeze(0)).reshape(-1, rank)

                # Eq. Normal com Tikhonov [cite: 17, 256]
                M = torch.t(Z) @ Z + 1e-6 * torch.eye(rank, device=device)
                W_m = tensor.permute(mode, *[i for i in range(4) if i != mode]).reshape(tsize[mode], -1)
                factors[mode] = torch.t(torch.linalg.solve(M, torch.t(Z) @ torch.t(W_m)))

        # 4. BALANCEMENTO DE ESCALA (Média Geométrica) [cite: 232, 314, 442]
        norms = [torch.norm(f, dim=0) for f in factors]
        gm_alpha = torch.pow(torch.prod(torch.stack(norms), dim=0), 1.0 / 4.0)
        for n in range(4):
            factors[n] *= (gm_alpha / (norms[n] + 1e-9))

        # 5. CONVERGÊNCIA
        err = calcul_err_sigma(tensor, sigma, reconstruct_tensor_from_factors(factors)) / norm_sigma_tensor
        err_recon = calcul_err(tensor,  reconstruct_tensor_from_factors(factors))
        err_recon_rel = err_recon/torch.norm(tensor)

        rec_errors.append(err.item())

        if verbose:
            print(
                f"Iter {iteration + 1}, Erro Sigma: {err:.6f}, Erro Recon: {err_recon:.6f}, Erro Recon: {err_recon_rel:.6f},Redução: {rec_errors[-2] - rec_errors[-1]:.4e}")

        if iteration > 0 and (rec_errors[-2] - rec_errors[-1]) < tol:
            if verbose: print("Convergência atingida.")
            break

    # Retorna para o formato original PyTorch [Out, In, H, W]
    U_T = factors[3]
    return [U_T, factors[0], factors[1], factors[2]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensor_path', type=str, required=True)
    parser.add_argument('--sigma_path', type=str, required=True)
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--n_iter_max', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    W = torch.load(args.tensor_path, weights_only=False)
    # O código agora espera Sigma^1/2 e calcula Sigma internamente para estabilidade
    S_half = torch.load(args.sigma_path, weights_only=False)

    factors = cp_als_sigma(W, args.rank, S_half, n_iter_max=args.n_iter_max)

    os.makedirs('outputs', exist_ok=True)



    save_path = f"outputs/cp_factors_{Path(args.tensor_path.replace("weights/", "").replace(".pt", "")).name}_rank{args.rank}.pt"
    torch.save(factors, save_path)
    print("✅ Processo concluído com sucesso.")