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
# @Authors: Pálka Petr
# @Emails: xpalka07@stud.fit.vutbr.cz

import argparse
import os
from pathlib import Path
import numpy as np
from numpy.linalg import inv
from scipy.linalg import eigh

from pyannote.audio.utils.plda import l2norm, compute_scatters, compute_lda, em_plda


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed", type=str, required=True,
                        help="Path to embedding matrix (N, D) in .npy")
    parser.add_argument("--label", type=str, required=True,
                        help="Path to label vector (N) in .npy") 
    parser.add_argument("--out_dir", type=str, default=".", 
                        help="Directory where xvec_transform.npz and plda.npz will be stored.")
    parser.add_argument("--lda_dim", type=int, default=128, 
                        help="LDA projection to lda_dim before PLDA.")
    parser.add_argument("--iters", type=int, default=20, 
                        help="EM iterations for PLDA.")
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading embedding and labels...")
    X = np.load(args.embed)
    y = np.load(args.label)
    
    N, D = X.shape
    assert len(X) == len(y)
    lda_dim = min(int(args.lda_dim), D)
    niters = int(args.iters)  # em-plda iters


    # LDA
    mean1 = X.mean(0)
    X_train_lda = np.sqrt(D) * l2norm(X - mean1)  # preprocessing

    print("Computing LDA...")
    glob_mu, Sb, Sw = compute_scatters(X_train_lda, y)
    lda = compute_lda(Sb / N, Sw / N, lda_dim=lda_dim)
    mean2 = -lda.T @ glob_mu
    
    path_xtf = os.path.join(args.out_dir, "xvec_transform.npz")
    np.savez(path_xtf, mean1=mean1, lda=lda, mean2=mean2)
    print(f"Saved preprocessing and LDA to `{path_xtf}`")
    
    
    # PLDA
    X_train_plda = np.sqrt(lda_dim) * l2norm(X_train_lda @ lda - mean2)
    
    # init plda params
    plda_mu = X_train_plda.mean(axis=0)
    Sigma_ac = np.identity(lda_dim)
    Sigma_wc = np.identity(lda_dim)
    
    print("Computing PLDA...")
    plda_mu, Sigma_ac, Sigma_wc = em_plda(X_train_plda, y, plda_mu, Sigma_ac, Sigma_wc, niters)
    
    acvar, wccn = eigh(Sigma_ac, Sigma_wc)
    plda_psi = acvar[::-1]
    plda_tr = wccn.T[::-1]
    
    path_plda = os.path.join(args.out_dir, "plda.npz")
    np.savez(path_plda, mu=plda_mu, tr=plda_tr, psi=plda_psi)
    print(f"Saved PLDA params to `{path_plda}`")
    
    print("Done!")
    # VBx_input_features = np.sqrt(lda_dim) * l2norm(np.sqrt(D) * l2norm(X - mean1) @ lda - mean2)
    
if __name__ == "__main__":
    main()

