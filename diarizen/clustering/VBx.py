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
#
# Revision History
#   L. Burget   20/1/2021 1:00AM - original version derived from the more
#   P. Pálka    28/4/2025 17:00PM - original version derived from the more
#                                  complex VB_diarization.py avaiable at
# https://github.com/BUTSpeechFIT/VBx/blob/e39af548bb41143a7136d08310765746192e34da/VBx/VB_diarization.py
#

import numpy as np
from scipy.special import logsumexp

from scipy.linalg import eigh
from scipy.special import softmax


def VBx(X, Phi, loopProb=0.9, Fa=1.0, Fb=1.0, pi=10, gamma=None, maxIters=10,
        epsilon=1e-4, alphaQInit=1.0, ref=None, plot=False,
        return_model=False, alpha=None, invL=None):
    """
    Inputs:
    X           - T x D array, where columns are D dimensional feature vectors
                  (e.g. x-vectors) for T frames
    Phi         - D array with across-class covariance matrix diagonal.
                  The model assumes zero mean, diagonal across-class and
                  identity within-class covariance matrix.
    loopProb    - Probability of not switching speakers between frames
    Fa          - Scale sufficient statiscits
    Fb          - Speaker regularization coefficient Fb controls the final number of speakers
    pi          - If integer value, it sets the maximum number of speakers
                  that can be found in the utterance.
                  If vector, it is the initialization for speaker priors (see Outputs: pi)
    gamma       - An initialization for the matrix of responsibilities (see Outputs: gamma)
    maxIters    - The maximum number of VB iterations
    epsilon     - Stop iterating, if the obj. fun. improvement is less than epsilon
    alphaQInit  - Dirichlet concentraion parameter for initializing gamma
    ref         - T dim. integer vector with per frame reference speaker IDs (0:maxSpeakers)
    plot        - If set to True, plot per-frame marginal speaker posteriors 'gamma'
    return_model- Return also speaker model parameter
    alpha, invL - If provided, these are speaker model parameters used in the first iteration

    Outputs:
    gamma       - S x T matrix of responsibilities (marginal posteriors)
                  attributing each frame to one of S possible speakers
                  (S is defined by input parameter pi)
    pi          - S dimensional column vector of ML learned speaker priors.
                  This allows us to estimate the number of speaker in the
                  utterance as the probabilities of the redundant speaker
                  converge to zero.
    Li          - Values of auxiliary function (and DER and frame cross-entropy
                  between gamma and reference, if 'ref' is provided) over iterations.
    alpha, invL - Speaker model parameters returned only if return_model=True

    Reference:
      Landini F., Profant J., Diez M., Burget L.: Bayesian HMM clustering of
      x-vector sequences (VBx) in speaker diarization: theory, implementation
      and analysis on standard tasks
    """
    """
    The comments in the code refers to the equations from the paper above. Also
    the names of variables try to be consistent with the symbols in the paper.
    """
    D = X.shape[1]  # feature (e.g. x-vector) dimensionality

    if type(pi) is int:
        pi = np.ones(pi)/pi

    if gamma is None:
        # initialize gamma from flat Dirichlet prior with
        # concentration parameter alphaQInit
        gamma = np.random.gamma(alphaQInit, size=(X.shape[0], len(pi)))
        gamma = gamma / gamma.sum(1, keepdims=True)

    assert(gamma.shape[1] == len(pi) and gamma.shape[0] == X.shape[0])

    G = -0.5*(np.sum(X**2, axis=1, keepdims=True) + D*np.log(2*np.pi))  # per-frame constant term in (23)
    V = np.sqrt(Phi)  # between (5) and (6)
    rho = X * V  # (18)
    Li = []
    ELBO = None ##
    for ii in range(maxIters):
        # Do not start with estimating speaker models if those are provided
        # in the argument
        if ii > 0 or alpha is None or invL is None:
            invL = 1.0 / (1 + Fa/Fb * gamma.sum(axis=0, keepdims=True).T*Phi)  # (17) for all speakers
            alpha = Fa/Fb * invL * gamma.T.dot(rho)  # (16) for all speakers
        log_p_ = Fa * (rho.dot(alpha.T) - 0.5 * (invL+alpha**2).dot(Phi) + G)  # (23) for all speakers

        if loopProb <= 0.:
            # use GMM update instead as it is much faster for large amount of speakers
            eps = 1e-8
            lpi = np.log(pi + eps) 
            log_p_x = logsumexp(log_p_ + lpi, axis=-1)  # marginal LLH of each data point
            log_pX_ = np.sum(log_p_x, axis=0)  # total LLH over all data points (to monitor ELBO)

            gamma = np.exp(log_p_ + lpi - log_p_x[:, None])  # responsibilities
            pi = np.sum(gamma, axis=0)
        else:
            # HMM (original code)         
            tr = np.eye(len(pi)) * loopProb + (1-loopProb) * pi  # (1) transition probability matrix
            gamma, log_pX_, logA, logB = forward_backward(log_p_, tr, pi)  # (19) gamma, (20) logA, (21) logB, (22) log_pX_
            pi = gamma[0] + (1-loopProb)*pi * np.sum(np.exp(logsumexp(
                logA[:-1], axis=1, keepdims=True) + log_p_[1:] + logB[1:] - log_pX_
            ), axis=0)  # (24)

        pi = pi / pi.sum()

        ELBO = log_pX_ + Fb * 0.5 * np.sum(np.log(invL) - invL - alpha**2 + 1)  # (25)
        Li.append([ELBO])

        if ii > 0 and ELBO - Li[-2][0] < epsilon:
            if ELBO - Li[-2][0] < 0:
                print('WARNING: Value of auxiliary function has decreased!')
            break
    return (gamma, pi, Li) + ((alpha, invL) if return_model else ())

def cluster_vbx(ahc_init, fea, Phi, Fa, Fb, loopProb=0.0, maxIters=20, init_smoothing=7.0):
    """ahc_init (T x N_clusters) """
    qinit = np.zeros((len(ahc_init), ahc_init.max() + 1))
    qinit[range(len(ahc_init)), ahc_init.astype(int)] = 1.0
    qinit = qinit if init_smoothing < 0 else softmax(qinit * init_smoothing, axis=1)
    gamma, pi, _, _, _ = VBx(
        fea, Phi, 
        loopProb=loopProb, 
        Fa=Fa, Fb=Fb, 
        pi=qinit.shape[1], gamma=qinit, 
        maxIters=maxIters, return_model=True
    )
    return gamma, pi

def l2_norm(vec_or_matrix):
    """ L2 normalization of vector array.

    Args:
        vec_or_matrix (np.array): one vector or array of vectors

    Returns:
        np.array: normalized vector or array of normalized vectors
    """
    if len(vec_or_matrix.shape) == 1:
        # linear vector
        return vec_or_matrix / np.linalg.norm(vec_or_matrix)
    elif len(vec_or_matrix.shape) == 2:
        return vec_or_matrix / np.linalg.norm(vec_or_matrix, axis=1, ord=2)[:, np.newaxis]
    else:
        raise ValueError('Wrong number of dimensions, 1 or 2 is supported, not %i.' % len(vec_or_matrix.shape))

def vbx_setup(tf_dir):
    """ 
    Loads the transformation pipeline for x-vectors into the PLDA space for VBx.

    Args:
        tf_dir (str): Directory path containing 'xvec_transform.npz' and 'plda.npz' files.

    Returns:
        xvec_tf (function): Transformation function to preprocess x-vectors (centering, whitening, LDA).
        plda_tf (function): Transformation function to map x-vectors into the PLDA latent space.
        plda_psi (np.ndarray): Eigenvalues of the between-class covariance in the PLDA space (reordered).
    """
    
    x = np.load(f'{tf_dir}/xvec_transform.npz')
    mean1, mean2, lda = x['mean1'], x['mean2'], x['lda']
    
    p = np.load(f'{tf_dir}/plda.npz')
    plda_mu, plda_tr, plda_psi = p['mu'], p['tr'], p['psi']
    
    # within-class, between-class matrices (W, B)
    W = np.linalg.inv(plda_tr.T.dot(plda_tr))
    B = np.linalg.inv((plda_tr.T / plda_psi).dot(plda_tr))
    
    # Solve the generalized eigenvalue problem for whitening and sort eigenvalues
    acvar, wccn = eigh(B, W)
    plda_psi = acvar[::-1]
    plda_tr = wccn.T[::-1]
    
    # tf for preprocessing x-vectors (centering, whitening, LDA)
    xvec_tf = lambda x: np.sqrt(lda.shape[1]) * l2_norm(
        lda.T.dot(
            np.sqrt(lda.shape[0]) * l2_norm(x - mean1).T
        ).T - mean2
    )   
    # tf to map x-vectors to the PLDA latent space (center, apply transform, optional truncate to 'lda_dim')
    plda_tf = lambda x0, lda_dim=lda.shape[1]: (x0 - plda_mu).dot(plda_tr.T)[:, :lda_dim]
    return xvec_tf, plda_tf, plda_psi


