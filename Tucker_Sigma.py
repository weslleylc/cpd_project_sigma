"""
Tucker2-ALS-Sigma Algorithm Implementation
Re-written with detailed English comments explaining the author's mathematical steps.
"""

import torch
import cupy as cp
from cupyx.scipy.sparse.linalg import minres, LinearOperator
import argparse

def reconstruct_tensor_from_factors(core, factors):
    """
    Reconstructs the full approximated tensor from the compressed core and factors.
    Math: W_hat = Core x_1 Factor_1 x_2 Factor_2
    """
    # tensordot contracts specific dimensions. Here it multiplies the core by the factors
    # to project the compressed core back to the original input/output channel sizes.
    return torch.tensordot(torch.tensordot(factors[0], core, dims=[[1], [0]]), factors[1], dims=[[3], [1]])


def calcul_err_sigma(tensor, sigma, tensor2):
    """
    Calculates the Frobenius norm of the error, weighted by the Sigma matrix.
    Math: || Sigma * (W_original - W_reconstructed) ||_F
    """
    X_diff = tensor - tensor2
    # Reshape the 4D difference into a 2D matrix before multiplying by the 2D Sigma matrix
    return torch.norm(torch.matmul(sigma, X_diff.reshape((-1, tensor.size()[3]))))


def matvec_M(x, A, D, r, b):
    """
    Custom Matrix-Vector multiplication for the MINRES solver.
    Instead of calculating a massive (A^t A) * (D^t D) matrix which would crash the RAM,
    this function calculates the product 'on the fly' using Einstein Summation (einsum).
    """
    A = cp.asarray(A)
    D = cp.asarray(D)
    b = cp.asarray(b)

    # Extract dimensions from the author's specific tensor shapes
    r, n2, n3, s, n4, n5 = D.shape
    n1, n2, n3, n7, n8, n9 = A.shape

    # Ensure 'x' is a CuPy array and reshape it to match the expected input factor
    x = cp.asarray(x).reshape(n1, r)

    # Einsum performs complex multi-dimensional multiplication efficiently on the GPU
    result = cp.einsum('ijkumn,qjksmn,us -> iq', A, D, x)
    return result.ravel() # Flatten back to a 1D vector for the MINRES solver


def solve_M_B(A, D, b, r, n):
    """
    Solves the linear system (M * U = b) to find the Factor matrix using CuPy MINRES.
    MINRES (Minimum Residual Method) is an iterative solver for large sparse/symmetric systems.
    """
    A = cp.asarray(A)
    D = cp.asarray(D)
    b = cp.asarray(b)

    # We tell CuPy to use our custom 'matvec_M' function instead of a real matrix
    M_operator = LinearOperator((n, n), matvec=lambda v: matvec_M(v, A, D, r, b), dtype=cp.float32)

    # Run the solver until the tolerance (1e-10) is met or 2000 iterations pass
    x, istop = minres(M_operator, b, tol=1e-10, maxiter=2000)[:2]
    return x.astype(cp.float32)


def partial_tucker_sigma(
    tensorT: torch.Tensor,
    rank,
    sigma: torch.Tensor,
    init='svd',
    tinit=None,
    n_iter_max=int,
    tol=1e-6,
    verbose=1,
    cvg_criterion="abs_rec_error",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. PREPARE THE SIGMA MATRIX
    # The author transposes Sigma initially to match their specific mathematical derivations
    sigma = torch.t(sigma)
    sigma = sigma.to(device)

    # 2. PREPARE THE WEIGHT TENSOR
    # The author moves the first axis (Output Channels) to the very end.
    # If input is [Out, In, H, W], it becomes [In, H, W, Out].
    tensor = torch.moveaxis(tensorT, 0, -1)

    # Adds a dummy dimension. It becomes a 5D tensor: [In, H, W, 1, Out]
    tensor = tensor[:, :, None, :]
    tensor = tensor.to(device)

    # Invert the user-provided ranks to match the author's internal logic
    rank = [rank[1], rank[0]]
    tsize = tensor.size()

    # Pre-calculate the total target energy of the original tensor (used for percentage error later)
    norm_sigma_tensor = torch.norm(torch.flatten(torch.matmul(sigma, tensor.reshape((-1, tsize[3])))))

    # 3. INITIALIZATION
    if tinit != None:
        # If the user provides an existing core and factors, orthonormalize them using QR decomposition
        core, factors = tinit
        core, factors = core[:], factors[:]

        Q, R = torch.linalg.qr(factors[0])
        factors[0] = Q
        core = torch.tensordot(R, core, dims=[[1], [0]])

        Q, R = torch.linalg.qr(factors[1])
        factors[1] = Q
        core = torch.tensordot(core, R, dims=[[-1], [1]])
    else:
        # Default SVD (Singular Value Decomposition) Initialization
        if init == 'svd':
            # Initialize the First Factor Matrix
            U, _, _ = torch.linalg.svd(tensor.reshape((tsize[0], -1)), full_matrices=False)
            factors = [U[:, :rank[0]].to(device)]

            # Initialize the Second Factor Matrix
            U, _, _ = torch.linalg.svd(torch.t(tensor.reshape((-1, tsize[3]))), full_matrices=False)
            factors += [U[:, :rank[1]]]

            # Initialize the Core Tensor by solving a Least Squares problem
            sigma2 = sigma.reshape((-1,) + tuple(tsize[i] for i in range(3)))
            sigma2 = torch.tensordot(sigma2, factors[0], dims=[[1], [0]]).moveaxis(-1, 1).reshape((sigma2.size()[0], -1))

            tensor2 = tensor.reshape((-1, tsize[3]))
            sigmatensor = sigma @ tensor2

            # lstsq calculates the exact mathematical pseudo-inverse
            DX = torch.linalg.lstsq(torch.t(sigma2) @ sigma2, torch.t(sigma2) @ sigmatensor)[0]
            core = torch.t(torch.linalg.solve(torch.t(factors[1]) @ factors[1], torch.t(factors[1]) @ torch.t(DX)))
            core = core.reshape([rank[0], tsize[1], tsize[2], rank[1]]).to(device)

    core_factors_checkpoint = [core, factors]
    rec_errors = []

    # Calculate initial error before any ALS iterations happen
    unnorml_rec_error = calcul_err_sigma(tensor, sigma, reconstruct_tensor_from_factors(core, factors))
    rec_error = unnorml_rec_error / norm_sigma_tensor
    rec_errors.append(rec_error)

    # 4. THE ALS (ALTERNATING LEAST SQUARES) LOOP
    for iteration in range(n_iter_max):
        if verbose > 1:
            print("Starting iteration", iteration + 1)

        # The author's weird loop: 0 = First Factor, 3 = Second Factor, 4 = Core Tensor
        for mode in [0, 3, 4]:
            if verbose > 1:
                print("Mode", mode, "of", 3)

            # --- MODE 0: UPDATE THE FIRST FACTOR MATRIX ---
            if mode == 0:
                # Prepare Sigma contractions
                sigmaTsigma = torch.tensordot(sigma, sigma, dims=[[0], [0]])
                sigmaTsigma_reshaped = sigmaTsigma.reshape(list(tsize[:3]) + list(tsize[:3]))

                # Contract the core with the second factor
                Core_factor = torch.tensordot(core, factors[1], dims=[[-1], [1]]).to(device)
                Corefactor_2 = torch.tensordot(Core_factor, Core_factor, dims=[[-1], [-1]])

                # Prepare the target side of the equation
                tensor_reshaped = tensor.reshape((-1, tsize[3])) @ factors[1]
                sigmatensor = sigma @ tensor_reshaped
                sigma2tensor = torch.tensordot(sigma, sigmatensor, dims=[[0], [0]]).reshape(list(tsize)[:3] + [-1])
                sigmatensorcore = torch.tensordot(sigma2tensor, core, dims=[[1, 2, 3], [1, 2, 3]]).flatten()

                # This is the heaviest mathematical step. The author uses CuPy MINRES here
                # to avoid out-of-memory errors on the first factor.
                factor = solve_M_B(sigmaTsigma_reshaped, Corefactor_2, sigmatensorcore, rank[0], sigmatensorcore.size(0))
                factor = factor.reshape(tsize[mode], rank[0])

                # Safely convert CuPy array back to PyTorch tensor without DLPack crash
                factors[0] = torch.as_tensor(factor, device=device)

            # --- MODE 3: UPDATE THE SECOND FACTOR MATRIX ---
            elif mode == 3:
                # Because the second factor is usually smaller (Output Channels),
                # the author uses standard PyTorch exact solver (linalg.solve) instead of CuPy MINRES.
                sigma_reshaped = sigma.reshape((-1,) + tuple(tsize[i] for i in range(3)))
                sigma_factor = torch.tensordot(sigma_reshaped, factors[0], dims=[[1], [0]]).moveaxis(-1, 1)
                sigma_factorcore = torch.tensordot(sigma_factor, core, dims=[[1, 2, 3], [0, 1, 2]])

                b = torch.t(sigma_factorcore) @ sigma @ tensor.reshape((-1, tsize[3]))
                factors[1] = torch.t(torch.linalg.solve(torch.t(sigma_factorcore) @ sigma_factorcore, b)).reshape((-1, rank[1]))

            # --- MODE 4: UPDATE THE CORE TENSOR ---
            else:
                sigma_reshaped = sigma.reshape((-1,) + tuple(tsize[i] for i in range(3)))
                sigma_factor = torch.tensordot(sigma_reshaped, factors[0], dims=[[1], [0]]).moveaxis(-1, 1).reshape((sigma2.size()[0], -1))

                tensor_reshaped = tensor.reshape((-1, tsize[3]))
                sigmatensor = sigma @ tensor_reshaped

                # Solve for the core tensor using standard exact PyTorch solver
                DX = torch.linalg.solve(torch.t(sigma_factor) @ sigma_factor, torch.t(sigma_factor) @ sigmatensor)
                core = torch.t(torch.linalg.solve(torch.t(factors[1]) @ factors[1], torch.t(factors[1]) @ torch.t(DX))).reshape(core.size())

        # 5. ERROR CHECKING & EARLY STOPPING
        if tol:
            unnorml_rec_error = calcul_err_sigma(tensor, sigma, reconstruct_tensor_from_factors(core, factors))
            rec_error = unnorml_rec_error / norm_sigma_tensor
            rec_errors.append(rec_error)

            if iteration >= 1:
                # Check how much the error dropped in this iteration
                rec_error_decrease = rec_errors[-2] - rec_errors[-1]

                if verbose:
                    print(f"iteration {iteration}, reconstruction error: {rec_error:.6f}, decrease = {rec_error_decrease:.6f}, unnormalized = {unnorml_rec_error:.6f}")

                # If error went up instead of down, the algorithm diverged. Rollback to last good state.
                if rec_error_decrease < 0:
                    core, factors = core_factors_checkpoint[:]
                    print('divergence, stopped before')
                    break

                if cvg_criterion == "abs_rec_error":
                    stop_flag = abs(rec_error_decrease) < tol
                elif cvg_criterion == "rec_error":
                    stop_flag = rec_error_decrease < tol
                else:
                    raise TypeError("Unknown convergence criterion")

                # If the decrease is smaller than the tolerance, we are done!
                if stop_flag:
                    if verbose:
                        print(f"Tucker converged after {iteration} iterations")
                    break

            else:
                if verbose:
                    print(f"reconstruction error={rec_errors[-1]}")

        # Save a checkpoint of the current iteration in case of divergence
        core_factors_checkpoint = [core[:], factors[:]]

    return (core, [factors[1], factors[0]])


def main():
    parser = argparse.ArgumentParser(description="Run Tucker2-ALS-Sigma Algorithm.")

    parser.add_argument('--tensor_path', type=str, required=True, help='Path to the input tensor (.pt file)')
    parser.add_argument('--rank', type=int, nargs='+', required=True, help='Target rank(s) for decomposition')
    parser.add_argument('--sigma_path', type=str, required=True, help='Path to the sigma tensor (.pt file)')
    parser.add_argument('--init', type=str, default='svd', choices=['svd', 'random'], help='Initialization method')
    parser.add_argument('--tinit', type=str, default=None, help='Optional tensor initialization (.pt file)')
    parser.add_argument('--n_iter_max', type=int, default=100, help='Maximum number of iterations')
    parser.add_argument('--tol', type=float, default=1e-6, help='Tolerance for convergence')
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity level')
    parser.add_argument('--cvg_criterion', type=str, default='abs_rec_error', help='Convergence criterion')

    args = parser.parse_args()

    # Disable weights_only to allow loading legacy author tensors without warnings
    tensorT = torch.load(args.tensor_path, weights_only=False)
    sigma = torch.load(args.sigma_path, weights_only=False)
    tinit = torch.load(args.tinit, weights_only=False) if args.tinit else None

    # Run the main algorithm
    core, factors = partial_tucker_sigma(
        tensorT=tensorT,
        rank=args.rank,
        sigma=sigma,
        init=args.init,
        tinit=tinit,
        n_iter_max=args.n_iter_max,
        tol=args.tol,
        verbose=args.verbose,
        cvg_criterion=args.cvg_criterion
    )

    # Save the output factors (I added this so the script actually saves the result of the computation!)
    import os
    os.makedirs('outputs', exist_ok=True)
    base_name = os.path.basename(args.tensor_path).replace('.pt', '')
    save_path = f"outputs/tucker_{base_name}_rank{args.rank[0]}_{args.rank[1]}.pt"

    torch.save({
        'core': core.cpu(),
        'factor_out': factors[0].cpu(),
        'factor_in': factors[1].cpu()
    }, save_path)

    print(f"✅ Tucker decomposition complete! Factors saved to: {save_path}")

if __name__ == "__main__":
    main()