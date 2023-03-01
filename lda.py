# Import necessary libraries
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
embeddings = # input
embeddings_normalized = (embeddings - np.mean(embeddings, axis=0)) / np.std(embeddings, axis=0)
lda = LinearDiscriminantAnalysis(n_components=)
lda.fit(embeddings_normalized, labels)
lda.train()
score = lda.score(embeddings_reduced, labels)
lda.deploy()
