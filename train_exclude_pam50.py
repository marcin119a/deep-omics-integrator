"""
Train model excluding PAM50 patients from the training set.
This allows for independent validation on PAM50 breast cancer subtypes.
"""
import pandas as pd
import numpy as np
from Bio import SeqIO
import pyranges as pr
import argparse
import os
from datetime import datetime
from src.config import set_seed

from src import preprocess, annotation, tokenizer_utils, model, training, config, data_cache


def load_pam50_barcodes(pam50_path="data/pam50.csv"):
    """Load PAM50 patient barcodes to exclude."""
    print(f"Loading PAM50 barcodes from: {pam50_path}")
    pam50_df = pd.read_csv(pam50_path)
    
    # Extract case_barcode from bcr_patient_barcode (first 12 characters)
    pam50_df['case_barcode'] = pam50_df['bcr_patient_barcode'].str[:12]
    
    pam50_barcodes = set(pam50_df['case_barcode'].unique())
    print(f"✓ Loaded {len(pam50_barcodes)} unique PAM50 patient barcodes to exclude")
    
    return pam50_barcodes


def train_and_save_model_no_pam50(model_name="model_no_pam50", pam50_path="data/pam50.csv", 
                                   use_cached=False, save_processed=True):
    """Train model excluding PAM50 patients."""
    
    # Load PAM50 barcodes to exclude
    pam50_barcodes = load_pam50_barcodes(pam50_path)
    
    print("\nProcessing data and excluding PAM50 patients...")
    print("Note: Not using cached data to ensure proper filtering")
    
    # Load data
    df = preprocess.load_mutation_data(config.data_paths['mutations'])
    expression_df = preprocess.load_expression_data(config.data_paths['expression'])
    signature_df = preprocess.load_signatures_data(config.data_paths['signatures'])
    
    # Filter out PAM50 patients from mutations
    print(f"Mutations before PAM50 filtering: {len(df)} rows")
    df = df[~df['case_barcode'].isin(pam50_barcodes)].copy()
    print(f"Mutations after PAM50 filtering: {len(df)} rows")
    
    # Filter out PAM50 patients from expression
    print(f"Expression before PAM50 filtering: {len(expression_df)} rows")
    expression_df = expression_df[~expression_df['case_barcode'].isin(pam50_barcodes)].copy()
    print(f"Expression after PAM50 filtering: {len(expression_df)} rows")
    
    # Filter out PAM50 patients from signatures
    print(f"Signatures before PAM50 filtering: {len(signature_df)} samples")
    signature_df = signature_df[~signature_df['Samples'].isin(pam50_barcodes)].copy()
    print(f"Signatures after PAM50 filtering: {len(signature_df)} samples")
    
    # Process mutations
    gr = pr.PyRanges(df)
    fasta_sequences = SeqIO.to_dict(SeqIO.parse(config.data_paths['reference_fasta'], "fasta"))
    df['Mutational_Motif'] = df.apply(lambda row: annotation.get_cosmic_notation(row, fasta_sequences), axis=1)
    df['MutationType'] = df['Mutational_Motif'].apply(annotation.normalize_mutation)
    df = annotation.assign_genomic_bin(df)
    
    # Filter top sites
    top_sites = df['primary_site'].value_counts().nlargest(24).index
    filtered_df = df[df['primary_site'].isin(top_sites)].copy()
    filtered_df['primary_site_cat'] = filtered_df['primary_site'].astype('category').cat.codes
    
    print(f"\nPrimary site distribution after PAM50 filtering:")
    print(filtered_df['primary_site'].value_counts().head(10))
    
    # Group mutations
    grouped_mutations_df = filtered_df.groupby('case_barcode').agg({
        'Mutational_Motif': list,
        'Genomic_Bin': list,
        'Hugo_Symbol': list,
        'primary_site': 'first',
        'primary_site_cat': 'first',
    }).reset_index()
    
    # Group expression
    grouped_expression_df = expression_df.groupby('case_barcode').agg({
        'gene_name': list,
        'tpm_unstranded': list,
    }).reset_index()
    
    # Merge all data
    grouped_df = pd.merge(grouped_mutations_df, grouped_expression_df, on="case_barcode")
    grouped_df = pd.merge(grouped_df, signature_df, left_on="case_barcode", right_on="Samples")
    
    print(f"\nFinal dataset size: {len(grouped_df)} samples")
    
    grouped_df['Signatures'] = grouped_df.apply(
        lambda row: list(np.log1p(row[config.signature_cols].astype(float).values)), axis=1
    )
    
    # Create tokenizers and prepare data
    print("Creating tokenizers and preparing data...")
    tokenizer_bin = tokenizer_utils.create_tokenizer(df['Genomic_Bin'].astype(str).to_list())
    tokenizer_gene = tokenizer_utils.create_tokenizer(df['Hugo_Symbol'].astype(str).to_list())
    X_bin, X_gene, X_signatures, X_rna = tokenizer_utils.prepare_data(grouped_df, tokenizer_bin, tokenizer_gene)
    
    y = np.array(grouped_df['primary_site_cat'].tolist())
    class_weights = training.get_class_weights(y)
    
    vocab_sizes = {
        'bin': len(tokenizer_bin.word_index) + 1,
        'gene': len(tokenizer_gene.word_index) + 1
    }
    num_classes = len(np.unique(y))
    
    print(f"\nDataset statistics:")
    print(f"  Total samples: {len(y)}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Genomic bin vocabulary size: {vocab_sizes['bin']}")
    print(f"  Gene vocabulary size: {vocab_sizes['gene']}")
    
    # Save processed data if requested
    if save_processed:
        print("\nSaving processed data for future use...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = "processed_data_no_pam50"
        os.makedirs(save_dir, exist_ok=True)
        
        data_path = os.path.join(save_dir, f"dataset_{timestamp}.npz")
        tokenizer_path = os.path.join(save_dir, f"tokenizers_{timestamp}.pkl")
        metadata_path = os.path.join(save_dir, f"metadata_{timestamp}.pkl")
        
        # Save arrays
        np.savez_compressed(
            data_path,
            X_bin=X_bin,
            X_gene=X_gene,
            X_signatures=X_signatures,
            X_rna=X_rna,
            y=y
        )
        
        # Save tokenizers and metadata
        import pickle
        with open(tokenizer_path, 'wb') as f:
            pickle.dump({'tokenizer_bin': tokenizer_bin, 'tokenizer_gene': tokenizer_gene}, f)
        
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'class_weights': class_weights,
                'vocab_sizes': vocab_sizes,
                'num_classes': num_classes,
                'excluded_pam50': True,
                'n_pam50_excluded': len(pam50_barcodes)
            }, f)
        
        print(f"✓ Saved processed data to {save_dir}")
    
    # Create full model
    print("\nCreating model architecture...")
    full_model = model.create_model(
        vocab_bin=vocab_sizes['bin'],
        vocab_gene=vocab_sizes['gene'],
        num_classes=len(np.unique(y)),
        gene_dim=1177,
        sig_dim=86,
        rna_dim=1177,
        use_genomic_bin=True,
        use_gene=True,
        use_signature=True,
        use_rna=True
    )
    
    # Prepare inputs
    inputs = [X_bin, X_gene, X_signatures, X_rna]
    
    # Train model
    print("\nTraining model (excluding PAM50 patients)...")
    print("="*80)
    history = full_model.fit(
        inputs, y, 
        epochs=20, 
        batch_size=128, 
        validation_split=0.2,
        class_weight=class_weights, 
        verbose=1
    )
    
    # Create models directory if it doesn't exist
    models_dir = "saved_models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(models_dir, f"{model_name}_{timestamp}.keras")
    
    # Save the model
    full_model.save(model_path)
    print(f"\n✓ Model successfully saved to: {model_path}")
    
    # Print training summary
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    print(f"\nTraining Summary:")
    print(f"  Final Training Accuracy: {final_train_acc:.4f}")
    print(f"  Final Validation Accuracy: {final_val_acc:.4f}")
    print(f"  PAM50 patients excluded: {len(pam50_barcodes)}")
    print(f"  Training samples: {len(y)}")
    
    return model_path, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model excluding PAM50 patients')
    parser.add_argument('--name', type=str, default='model_no_pam50', 
                        help='Name for the saved model file')
    parser.add_argument('--pam50', type=str, default='data/pam50.csv',
                        help='Path to PAM50 CSV file')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save preprocessed data')
    args = parser.parse_args()
    
    set_seed(42)  # Set seed for reproducibility
    train_and_save_model_no_pam50(
        model_name=args.name,
        pam50_path=args.pam50,
        save_processed=not args.no_save
    )

