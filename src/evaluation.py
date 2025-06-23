import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from tensorflow.keras.models import Model


def visualize_tsne(model, X, y, label_map, layer_name='alpha_combination', title='t-SNE'):
    emb_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    emb = emb_model.predict(X)
    emb_2d = TSNE(n_components=2, random_state=42).fit_transform(emb)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=y, cmap='tab20', alpha=0.6)
    cbar = plt.colorbar(scatter, ticks=range(len(label_map)))
    cbar.ax.set_yticklabels(label_map.values())
    plt.title(title)
    plt.xlabel('TSNE-1')
    plt.ylabel('TSNE-2')
    plt.tight_layout()
    plt.show()