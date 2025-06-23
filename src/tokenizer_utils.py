import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer


def create_tokenizer(texts):
    tokenizer = Tokenizer(filters='', lower=False, split=' ', oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    return tokenizer


def to_matrix(tokenizer, texts):
    sequences = tokenizer.texts_to_sequences(texts)
    mat = np.zeros((len(texts), len(tokenizer.word_index) + 1))
    for i, seq in enumerate(sequences):
        for token in seq:
            mat[i, token] += 1
    return mat


def prepare_data(df, tokenizer_bin, tokenizer_gene):
    X_bin = to_matrix(tokenizer_bin, df['Genomic_Bin'])
    X_gene = to_matrix(tokenizer_gene, df['Hugo_Symbol'])
    X_signatures = np.array(df['Signatures'].tolist())
    X_rna = np.array(df['tpm_unstranded'].tolist())
    return X_bin, X_gene, X_signatures, X_rna