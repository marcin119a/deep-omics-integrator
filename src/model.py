import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Layer
from tensorflow.keras.models import Model


class TrainableAlphaLayer(Layer):
    def __init__(self, num_embeddings, **kwargs):
        super().__init__(**kwargs)
        self.alpha_vars = self.add_weight(shape=(num_embeddings,), initializer='ones', trainable=True)

    def call(self, embeddings):
        weights = tf.nn.softmax(self.alpha_vars)
        return tf.reduce_sum([weights[i] * embeddings[i] for i in range(len(embeddings))], axis=0)


def create_model(vocab_bin, vocab_gene, num_classes, gene_dim, sig_dim, rna_dim,
                 use_genomic_bin=True, use_gene=True, use_signature=True, use_rna=True):
    inputs, transformed = [], []
    if use_genomic_bin:
        x = Input(shape=(vocab_bin,), name='Genomic_Bin')
        inputs.append(x)
        transformed.append(Dense(128, activation='relu')(x))
    if use_gene:
        x = Input(shape=(vocab_gene,), name='Gene')
        inputs.append(x)
        transformed.append(Dense(128, activation='relu')(x))
    if use_signature:
        x = Input(shape=(sig_dim,), name='Signatures')
        inputs.append(x)
        transformed.append(Dense(128, activation='relu')(x))
    if use_rna:
        x = Input(shape=(rna_dim,), name='RNA')
        inputs.append(x)
        transformed.append(Dense(128, activation='relu')(x))
    combined = TrainableAlphaLayer(len(transformed))(transformed)
    x = Dense(256, activation='relu')(combined)
    out = Dense(num_classes, activation='softmax', name='output')(x)
    model = Model(inputs=inputs, outputs=[out])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model