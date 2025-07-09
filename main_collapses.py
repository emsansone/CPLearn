# Copyright 2025 CPLearn team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE. 
 
import os
import torch
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import numpy as np

COMPONENT_WISE_ENTROPY = False
DIR = 'embeddings'
DIM = 2048

def preprocessing(embeddings):

    embeddings = (embeddings - embeddings.min()) / (embeddings.max() - embeddings.min())
        
    n, d = embeddings.shape

    # Padding zeros if less than DIM features
    if DIM > d:
        padding = torch.zeros(n, DIM - d)
        embeddings = torch.cat([embeddings, padding], dim=1)
    elif DIM == d:
        pass
    else:
        raise ValueError("DIM must be greater than d")
    
    return embeddings


def get_ckpt_files(directory):
    ckpt_files = []
    methods = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".ckpt"):
                ckpt_files.append(os.path.join(root, file))
                methods.append(file.split(".")[0])
    return ckpt_files, methods

if __name__ == "__main__":

    ckpts, methods = get_ckpt_files(DIR)

    dicts = [torch.load(ckpt) for ckpt in ckpts]

    print('\n Found methods !\n')
    print(methods)

    #################################
    #################################
    #################################
    # Embedding analysis
    #################################
    #################################
    #################################
    print('\n Dimensional collapse')

    def plot_singular_values(embedding, name):
        batch_size, _ = embedding.shape
        embedding = preprocessing(embedding)
        cov = embedding.T @ embedding / batch_size
        U, S, Vh = torch.linalg.svd(cov)
        S = torch.log(S)
        S = S.sort(descending=True)[0]
        plt.plot(S, label=name)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylabel('Log of Singular Values', fontsize=20)
        plt.xlabel('Singular Value Rand Index', fontsize=20)
        plt.legend()

    for i in range(len(methods)):
        plot_singular_values(dicts[i]["embed"], methods[i])

    FONT = 20
    plt.xticks(fontsize=FONT-3)
    plt.yticks(fontsize=FONT-3)
    plt.locator_params(axis='y', nbins=6)
    plt.xlabel('Singular Value Rank Index', fontsize=FONT)
    plt.ylabel('Log of Singular Values', fontsize=FONT)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=3, fancybox=True, shadow=True, prop={'size': 13})
    plt.tight_layout()

    plt.savefig(DIR + '/Dimensional_collapse.pdf')
    plt.clf()
    plt.close()

    #################################
    #################################
    #################################
    # Entropy analysis
    #################################
    #################################
    #################################
    print('\n Intra-cluster collapse')
    def fit_gmm(data, num_components):
        model = GaussianMixture(n_components=num_components, covariance_type='diag')
        model.fit(data)
        return model

    def estimate_entropy_monte_carlo(model, num_samples=10000):
        """
        Entropy estimator of GMM through Monte Carlo sampling.
        """
        # Generate samples from the mixture model
        samples = model.sample(num_samples)[0]
        log_probs = model.score_samples(samples)
        
        # Estimate entropy using Monte Carlo sampling
        estimated_entropy = -np.mean(log_probs)
        
        return estimated_entropy

    def plot_entropy(test_feat, name, d):
        components = [10, 20, 50, 100, 200]
        mean = np.zeros(len(components))
        std = np.zeros(len(components))
        itr = 0
        feats = test_feat
        feats = preprocessing(feats).numpy()
        for num_components in components:

            try:                        
                # Fit a mixture of Gaussians
                model = fit_gmm(feats, num_components)


                # COMPONENTWISE ENTROPY
                if COMPONENT_WISE_ENTROPY == True:
                    entropies = []
                    for cov in model.covariances_:
                        # Ensure covariance matrix is 2D (handle diagonal case)
                        if cov.ndim == 1:
                            cov = np.diag(cov)
                        
                        entropy = 0.5 * d * np.log(2 * np.pi * np.e) + 0.5 * np.linalg.slogdet(cov)[1]
                        entropies.append(entropy)

                    # print(name, num_components, entropies)     
                    mean[itr] = np.mean(entropies)
                    std[itr] = np.std(entropies)

                else:
                    ent = []
                    for i in range(10):
                        # Compute the analytical lower bound of the entropy
                        entropy = estimate_entropy_monte_carlo(model)
                        ent.append(entropy)
                    
                    mean[itr] = np.mean(ent)
                    std[itr] = np.std(ent)

            except:
                mean[itr] = np.nan
                std[itr] = np.nan

            itr += 1
        plt.plot(components, mean, label='{}'.format(name))
        plt.fill_between(components, mean-std, mean+std, alpha=0.2)


    for i in range(len(methods)):
        _, d = dicts[i]["embed"].shape
        plot_entropy(dicts[i]["embed"], methods[i], d)

    FONT = 20
    plt.xscale("log")
    plt.xticks(fontsize=FONT)
    plt.yticks(fontsize=FONT)
    plt.locator_params(axis='y', nbins=6)
    plt.xlabel('Number of Mixture Components', fontsize=FONT)
    plt.ylabel('Entropy Estimate', fontsize=FONT)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=3, fancybox=True, shadow=True, prop={'size': 13})
    plt.tight_layout()
    plt.savefig(DIR + '/Intra_collapse.pdf')