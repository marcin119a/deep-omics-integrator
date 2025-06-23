import pandas as pd
import numpy as np
from Bio import SeqIO
import pyranges as pr
import argparse

from src import preprocess, annotation, tokenizer_utils, model, training, evaluation, config


def main(experiment_name="Full Model", folds=5):
    print(f"Running experiment: {experiment_name}")

    df = preprocess.load_mutation_data(config.data_paths['mutations'])
    expression_df = preprocess.load_expression_data(config.data_paths['expression'])
    signature_df = preprocess.load_signatures_data(config.data_paths['signatures'])

    gr = pr.PyRanges(df)
    fasta_sequences = SeqIO.to_dict(SeqIO.parse(config.data_paths['reference_fasta'], "fasta"))
    df['Mutational_Motif'] = df.apply(lambda row: annotation.get_cosmic_notation(row, fasta_sequences), axis=1)
    df['MutationType'] = df['Mutational_Motif'].apply(annotation.normalize_mutation)
    df = annotation.assign_genomic_bin(df)

    top_sites = df['primary_site'].value_counts().nlargest(24).index
    filtered_df = df[df['primary_site'].isin(top_sites)].copy()
    filtered_df['primary_site_cat'] = filtered_df['primary_site'].astype('category').cat.codes

    grouped_mutations_df = filtered_df.groupby('case_barcode').agg({
        'Mutational_Motif': list,
        'Genomic_Bin': list,
        'Hugo_Symbol': list,
        'primary_site': 'first',
        'primary_site_cat': 'first',
    }).reset_index()

    grouped_expression_df = expression_df.groupby('case_barcode').agg({
        'gene_name': list,
        'tpm_unstranded': list,
    }).reset_index()

    grouped_df = pd.merge(grouped_mutations_df, grouped_expression_df, on="case_barcode")
    grouped_df = pd.merge(grouped_df, signature_df, left_on="case_barcode", right_on="Samples")

    grouped_df['Signatures'] = grouped_df.apply(
        lambda row: list(np.log1p(row[config.signature_cols].astype(float).values)), axis=1
    )

    tokenizer_bin = tokenizer_utils.create_tokenizer(df['Genomic_Bin'].astype(str).to_list())
    tokenizer_gene = tokenizer_utils.create_tokenizer(df['Hugo_Symbol'].astype(str).to_list())
    X_bin, X_gene, X_signatures, X_rna = tokenizer_utils.prepare_data(grouped_df, tokenizer_bin, tokenizer_gene)

    y = np.array(grouped_df['primary_site_cat'].tolist())
    class_weights = training.get_class_weights(y)

    vocab_sizes = {
        'bin': len(tokenizer_bin.word_index) + 1,
        'gene': len(tokenizer_gene.word_index) + 1
    }

    experiments = {
        "Full Model": (True, True, True, True),
        "No RNA": (True, True, True, False),
        "No Signature": (True, True, False, True),
        "No Gene": (True, False, True, True),
        "No Genomic Bin": (False, True, True, True)
    }

    if experiment_name not in experiments:
        raise ValueError(f"Unknown experiment: {experiment_name}")

    use_bin, use_gene, use_sig, use_rna = experiments[experiment_name]

    def model_fn():
        return model.create_model(
            vocab_bin=vocab_sizes['bin'],
            vocab_gene=vocab_sizes['gene'],
            num_classes=len(np.unique(y)),
            gene_dim=1177,
            sig_dim=86,
            rna_dim=1177,
            use_genomic_bin=use_bin,
            use_gene=use_gene,
            use_signature=use_sig,
            use_rna=use_rna
        )

    inputs = []
    if use_bin:
        inputs.append(X_bin)
    if use_gene:
        inputs.append(X_gene)
    if use_sig:
        inputs.append(X_signatures)
    if use_rna:
        inputs.append(X_rna)

    histories, scores = training.train_model_kfold(model_fn, inputs, y, class_weights, folds=folds)
    print(f"{experiment_name} – Mean Accuracy: {np.mean(scores):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='Full Model', help='Nazwa eksperymentu')
    parser.add_argument('--folds', type=int, default=5, help='Liczba foldów')
    args = parser.parse_args()
    main(args.experiment, args.folds)
