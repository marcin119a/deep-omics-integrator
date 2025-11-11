import pandas as pd
import numpy as np
from Bio import SeqIO
import pyranges as pr
import argparse
import os
from datetime import datetime
from src.config import set_seed

from src import preprocess, annotation, tokenizer_utils, model, training, config, data_cache


def train_and_save_full_model(model_name="full_model", use_cached=True, save_processed=True):
    """Train the full model on entire dataset and save it."""
    
    # Try to load cached processed data
    data_loaded = False
    if use_cached:
        data_path, tokenizer_path, metadata_path = data_cache.find_latest_processed_data()
        if data_path:
            try:
                print("Found cached processed data, loading...")
                (X_bin, X_gene, X_signatures, X_rna, y, 
                 tokenizer_bin, tokenizer_gene, class_weights, 
                 vocab_sizes, num_classes) = data_cache.load_processed_data(data_path, tokenizer_path, metadata_path)
                data_loaded = True
            except Exception as e:
                print(f"Error loading cached data: {e}")
                print("Will process data from scratch...")
    
    # If data not loaded from cache, process it
    if not data_loaded:
        print("Loading and preprocessing data from scratch...")
        
        # Load data
        df = preprocess.load_mutation_data(config.data_paths['mutations'])
        expression_df = preprocess.load_expression_data(config.data_paths['expression'])
        signature_df = preprocess.load_signatures_data(config.data_paths['signatures'])
        
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
        
        # Save processed data if requested
        if save_processed:
            print("\nSaving processed data for future use...")
            data_cache.save_processed_data(X_bin, X_gene, X_signatures, X_rna, y, 
                                          tokenizer_bin, tokenizer_gene, class_weights, 
                                          vocab_sizes, num_classes)
    
    # Create full model
    print("Creating model architecture...")
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
    print("Training full model on entire dataset...")
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
    
    return model_path, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and save the full deep omics integration model')
    parser.add_argument('--name', type=str, default='full_model', 
                        help='Name for the saved model file')
    parser.add_argument('--no-cache', action='store_true',
                        help='Do not use cached preprocessed data')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save preprocessed data')
    args = parser.parse_args()
    
    set_seed(42)  # Set seed for reproducibility
    train_and_save_full_model(
        model_name=args.name,
        use_cached=not args.no_cache,
        save_processed=not args.no_save
    )

