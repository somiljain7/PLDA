# Import necessary libraries
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
embeddings = # input
embeddings_normalized = (embeddings - np.mean(embeddings, axis=0)) / np.std(embeddings, axis=0)
pca = PCA(n_components=50)
embeddings_reduced = pca.fit_transform(embeddings_normalized)
lda = LinearDiscriminantAnalysis(n_components=50)
lda.fit(embeddings_reduced, labels)
lda.train()
score = lda.score(embeddings_reduced, labels)
lda.deploy()
