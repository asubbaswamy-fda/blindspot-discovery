from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
from tqdm import tqdm


class PlaneSpot:
    # Default blindspot discovery method developed  in
    # https://arxiv.org/abs/2207.04104
    # Performs dimensionality reduction on representation
    # Then clusters along with model loss info

    def __init__(self, reduction_method='tsne', reduced_dim=2):
        self.reduction_method = reduction_method
        self.reduced_dim = reduced_dim
        self.low_dim_embeddings = None
        self.best_gmm = None
        self.cluster_assignments = None

    def _reduce_dimensionality(self, embeddings):
        if self.low_dim_embeddings is None:
            if self.reduction_method == 'tsne':
                # Note: TSNE cannot be used to produce embeddings for new data
                self.dim_reducer = TSNE(n_components=self.reduced_dim)
            elif self.reduction_method == 'pca':
                self.dim_reducer = PCA(n_components=self.reduced_dim)
            else:
                raise Exception("Dimensionality reduction method not implemented. Choose one of 'tsne' or 'pca'")

            self.low_dim_embeddings = self.dim_reducer.fit_transform(embeddings)
        
        return self.low_dim_embeddings
    
    # fit using embeddings alone
    def fit(self, embeddings, min_nc = 2, max_nc=20):
        if self.low_dim_embeddings is None:
            self.pse = self._reduce_dimensionality(embeddings)
        
        self._bic_selection(self.pse, min_nc, max_nc)
    
    # fit using embedding, label, and loss
    def fit_using_loss(self, embeddings, label, loss, min_nc = 2, max_nc=20):
        if self.low_dim_embeddings is None:
            self._reduce_dimensionality(embeddings)

        # weight losses to roughly be in range of embeddings
        def make_planespot_embedding(weight=0.025):
            X = np.copy(self.low_dim_embeddings)
            X -= np.min(self.low_dim_embeddings, axis=0)
            X /= np.max(self.low_dim_embeddings, axis=0)

            return np.concatenate((X, 0.1 * label.reshape(-1, 1), weight * loss.reshape(-1, 1)), axis=1)
                
        self.pse = make_planespot_embedding()
        self._bic_selection(self.pse, min_nc, max_nc)
    
    def _bic_selection(self, pse, min_n_components=2, max_n_components=20):
        # select the number of clusters using BIC
        lowest_bic = np.infty
        bic = []
        for nc in range(min_n_components, max_n_components):
            gmm = GaussianMixture(n_components = nc, 
                                                covariance_type = 'full', 
                                                random_state = 1234)
            gmm.fit(pse)
            bic.append(gmm.bic(pse))

            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm

        self.best_gmm = best_gmm
        self.cluster_assignments = best_gmm.predict(pse)

 