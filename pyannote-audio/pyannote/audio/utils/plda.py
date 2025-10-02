#!/usr/bin/env python

# @Authors: Pálka Petr
# @Emails: xpalka07@stud.fit.vutbr.cz
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
from numpy.linalg import inv 
from scipy.linalg import eigh


def compute_scatters(X_train_lda, y):
    """
    Compute global mean, between-class scatter Sb, and within-class scatter Sw.

    Given samples X ∈ R^{N×D} and integer labels y, this computes:
        class counts n_c and means μ_c,
        global mean μ,
        M = ∑_c n_c μ_c μ_cᵀ,
        XtX = Xᵀ X,
        Sb = ∑_c n_c (μ_c - μ)(μ_c - μ)ᵀ = M − N μ μᵀ,
        Sw = ∑_c ∑_{x∈c} (x − μ_c)(x − μ_c)ᵀ = XtX − M.

    Parameters
    ----------
    X_train_lda : ndarray, shape (N, D)
        Data matrix (e.g., LDA-preprocessed x-vectors). Use float64 for stability.
    y : ndarray, shape (N,)
        Integer class labels; values need not be contiguous or start at zero.

    Returns
    -------
    glob_mu : ndarray, shape (D,)
        Global mean vector μ (of preprocessed embeddings).
    Sb : ndarray, shape (D, D)
        Between-class scatter matrix (unnormalized).
    Sw : ndarray, shape (D, D)
        Within-class scatter matrix (unnormalized).
    """
    
    # Map labels to 0..C-1
    classes, class_idx = np.unique(y, return_inverse=True)
    C = len(classes)
    N, D = X_train_lda.shape

    # Class means
    counts = np.bincount(class_idx).astype(X_train_lda.dtype)  # (C,)
    sums = np.zeros((C, D), dtype=X_train_lda.dtype)
    np.add.at(sums, class_idx, X_train_lda)  # accumulate row-wise
    means = sums / counts[:, None]  # (C, D)

    # Global mean and uncentered second moment X^T X
    glob_mu = X_train_lda.mean(axis=0, keepdims=True) # (1, D)
    XtX = X_train_lda.T @ X_train_lda  # (D, D)

    # Sum_c n_c mu_c mu_c^T
    M = (means.T * counts) @ means  # (D, D)

    Sb = M - (N * (glob_mu.T @ glob_mu))
    Sw = XtX - M
    return glob_mu.squeeze(), Sb, Sw


def compute_lda(Sb_cov, Sw_cov, lda_dim=128, floor_abs=5.8745e-06): 
    """
    Compute the LDA projection matrix by solving the generalized eigenproblem
    Sb v = λ Sw v via the whitening trick.

    The routine eigendecomposes Sw, floors its eigenvalues by `floor_abs`,
    forms Sw^{-1/2}, then eigendecomposes S = Sw^{-1/2} Sb Sw^{-1/2}.
    The returned transform is W = Sw^{-1/2} V_k, where columns are ordered
    by decreasing generalized eigenvalue (discriminability).

    Parameters
    ----------
    Sb_cov : (D, D) ndarray
        Between-class scatter/covariance matrix (symmetric).
    Sw_cov : (D, D) ndarray
        Within-class scatter/covariance matrix (symmetric, PSD/PD).
    lda_dim : int, default=128
        Number of LDA directions to keep (k <= D). Excess is clipped to D.
    floor_abs : float, default=5.8745e-06
        Absolute floor applied to eigenvalues of Sw before inversion.

    Returns
    -------
    lda_tr : (D, k) ndarray
        LDA transform matrix. Columns are generalized eigenvectors,
        orthonormal in the Sw metric (i.e., W^T Sw W ≈ I), ordered by
        decreasing eigenvalue.
    """
    
    # symm
    Sb_cov = 0.5*(Sb_cov + Sb_cov.T)
    Sw_cov = 0.5*(Sw_cov + Sw_cov.T)

    # eigendecompose Sw and floor its eigenvalues
    s, U = eigh(Sw_cov)
    s_f = np.maximum(s, floor_abs)

    # Sw^{-1/2}
    Sw_mhalf = (U / np.sqrt(s_f)) @ U.T  
    
    # whitened Sb
    S = Sw_mhalf @ Sb_cov @ Sw_mhalf
    S = 0.5*(S + S.T)

    # eigendecompose whitened Sb, take top-k
    w, V = eigh(S)           
    V = V[:, ::-1]
    w = w[::-1]

    k = min(lda_dim, Sb_cov.shape[0])
    lda_tr = Sw_mhalf @ V[:, :k]
    return lda_tr

    
def em_plda(X_train_plda, y, mu=None, Sigma_ac=None, Sigma_wc=None, niters=20):
    """
    EM training for (two-covariance) PLDA model.

    Parameters
    ----------
    X_train_plda : ndarray of shape (N, D)
        Matrix of x-vectors (embeddings), one row per sample.
    y : ndarray of shape (N,)
        Integer label per sample.
    mu : ndarray of shape (D,), optional
        Initial global mean. If None, uses the empirical mean of X.
    Sigma_ac : ndarray of shape (D, D), optional
        Initial across-class covariance. If None, uses identity.
    Sigma_wc : ndarray of shape (D, D), optional
        Initial within-class covariance. If None, uses identity.
    niters : int, default=20
        Number of EM iterations.

    Returns
    -------
    mu : ndarray of shape (D,)
        Estimated global mean.
    Sigma_ac : ndarray of shape (D, D)
        Estimated across-class covariance.
    Sigma_wc : ndarray of shape (D, D)
        Estimated within-class covariance.

    Notes
    -----
    - Per iteration, we solve for Σ_s = (Σ_ac^{-1} + N_s Σ_wc^{-1})^{-1} once per N_s bucket,
      then compute μ_s for all labels in the bucket in a single batched GEMM.
    """
    
    N, D = X_train_plda.shape
    assert N == len(y)
    mu       = X_train_plda.mean(axis=0) if mu is None else mu
    Sigma_ac = np.eye(D, dtype=X_train_plda.dtype) if Sigma_ac is None else Sigma_ac
    Sigma_wc = np.eye(D, dtype=X_train_plda.dtype) if Sigma_wc is None else Sigma_wc
    
    # Prepare groups to speed up the computation for very large S
    # Group by N_s (count per label), then by label id
    labels, class_idx, counts = np.unique(y, return_inverse=True, return_counts=True)
    counts_per_sample = counts[class_idx]  # (N,)
    S = labels.size

    # reindex  primary key: N_s, secondary: label
    perm_samples = np.lexsort((y, counts_per_sample))  
    X_grouped = X_train_plda[perm_samples]
    y_grouped = y[perm_samples]

    # Per-label boundaries in grouped order
    label_change          = np.r_[True, y_grouped[1:] != y_grouped[:-1]]
    label_starts_samples  = np.flatnonzero(label_change)         # (S,)
    label_ends_samples    = np.r_[label_starts_samples[1:], len(y_grouped)]
    N_                    = label_ends_samples - label_starts_samples  # (S,)

    # Buckets of labels by identical N_s
    Ns_groups = np.unique(N_, return_index=True, return_counts=True)
    
    assert N == N_.sum()
    assert S == len(label_starts_samples)
    
    
    # Per-label sums
    X_sum = np.add.reduceat(X_grouped, label_starts_samples, axis=0)  # (S, D)
    
    # Global Gram once (avoid per-label)
    XtX = X_grouped.T @ X_grouped  # (D, D)

    # accumulators
    μ_acc       = np.empty(D, dtype=X_grouped.dtype)
    Σ_ac_acc    = np.empty((D, D), dtype=X_grouped.dtype)
    NsΣs_acc    = np.empty_like(Σ_ac_acc)
    C_acc       = np.empty_like(Σ_ac_acc)
    musmus_acc  = np.empty_like(Σ_ac_acc)

    for i in range(niters):
        μ_acc.fill(0), Σ_ac_acc.fill(0), 
        NsΣs_acc.fill(0), C_acc.fill(0), musmus_acc.fill(0)

        invΣ_ac = inv(Sigma_ac)
        invΣ_wc = inv(Sigma_wc)
        base    = invΣ_ac @ mu

        for Nval, start, g_size in zip(*Ns_groups):
            # E-step
            Xsum_g = X_sum[start:start+g_size]  # (G, D)

            Prec = invΣ_ac + Nval * invΣ_wc  # (D, D)
            Σ_s  = inv(Prec)                 # (D, D)  
            μ_g = Σ_s @ (base[:, None] + invΣ_wc @ Xsum_g.T)  # (D, G)

            # M-step (accumulate)
            μ_acc += μ_g.sum(axis=1)  ##

            d_g = μ_g - mu[:, None]
            Σ_ac_acc += g_size * Σ_s + (d_g @ d_g.T)  ##

            NsΣs_acc   += g_size * Nval * Σ_s
            C_acc      += Xsum_g.T @ μ_g.T
            musmus_acc += Nval * (μ_g @ μ_g.T)
            
        Σ_wc_acc = NsΣs_acc + XtX - C_acc - C_acc.T + musmus_acc  ##
        
        # Finalize M-step
        mu = μ_acc / S
        Sigma_ac = Σ_ac_acc / S
        Sigma_wc = Σ_wc_acc / N

        # Symm + jitter
        eps = 1e-10
        Sigma_ac = 0.5*(Sigma_ac + Sigma_ac.T) + eps*np.eye(D)
        Sigma_wc = 0.5*(Sigma_wc + Sigma_wc.T) + eps*np.eye(D)
    
    return mu, Sigma_ac, Sigma_wc


def l2norm(vec_or_matrix): 
    """L2 normalization of vector array.

    Args:
        vec_or_matrix (np.array): one vector or array of vectors

    Returns:
        np.array: normalized vector or array of normalized vectors
    """
    if len(vec_or_matrix.shape) == 1:
        # linear vector
        return vec_or_matrix / np.linalg.norm(vec_or_matrix)
    elif len(vec_or_matrix.shape) == 2:
        return (
            vec_or_matrix / np.linalg.norm(vec_or_matrix, axis=1, ord=2)[:, np.newaxis]
        )
    else:
        raise ValueError(
            "Wrong number of dimensions, 1 or 2 is supported, not %i."
            % len(vec_or_matrix.shape)
        )

