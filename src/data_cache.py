"""
Module for caching and loading preprocessed datasets.
"""
import os
import pickle
import numpy as np
from datetime import datetime


def save_processed_data(X_bin, X_gene, X_signatures, X_rna, y, 
                        tokenizer_bin, tokenizer_gene, class_weights, 
                        vocab_sizes, num_classes, data_dir="processed_data"):
    """
    Save preprocessed data to disk.
    
    Args:
        X_bin: Genomic bin input data
        X_gene: Gene input data
        X_signatures: Signature input data
        X_rna: RNA expression input data
        y: Target labels
        tokenizer_bin: Tokenizer for genomic bins
        tokenizer_gene: Tokenizer for genes
        class_weights: Class weights dictionary
        vocab_sizes: Dictionary containing vocabulary sizes
        num_classes: Number of target classes
        data_dir: Directory to save data to
        
    Returns:
        Tuple of (data_path, tokenizer_path, metadata_path)
    """
    os.makedirs(data_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_path = os.path.join(data_dir, f"dataset_{timestamp}.npz")
    tokenizer_path = os.path.join(data_dir, f"tokenizers_{timestamp}.pkl")
    metadata_path = os.path.join(data_dir, f"metadata_{timestamp}.pkl")
    
    # Save arrays as compressed numpy file
    np.savez_compressed(
        data_path,
        X_bin=X_bin,
        X_gene=X_gene,
        X_signatures=X_signatures,
        X_rna=X_rna,
        y=y
    )
    
    # Save tokenizers
    with open(tokenizer_path, 'wb') as f:
        pickle.dump({
            'tokenizer_bin': tokenizer_bin,
            'tokenizer_gene': tokenizer_gene
        }, f)
    
    # Save metadata
    with open(metadata_path, 'wb') as f:
        pickle.dump({
            'class_weights': class_weights,
            'vocab_sizes': vocab_sizes,
            'num_classes': num_classes
        }, f)
    
    print(f"✓ Processed data saved to: {data_path}")
    print(f"✓ Tokenizers saved to: {tokenizer_path}")
    print(f"✓ Metadata saved to: {metadata_path}")
    
    return data_path, tokenizer_path, metadata_path


def load_processed_data(data_path, tokenizer_path, metadata_path):
    """
    Load preprocessed data from disk.
    
    Args:
        data_path: Path to numpy data file
        tokenizer_path: Path to tokenizers pickle file
        metadata_path: Path to metadata pickle file
        
    Returns:
        Tuple of (X_bin, X_gene, X_signatures, X_rna, y, 
                  tokenizer_bin, tokenizer_gene, class_weights, 
                  vocab_sizes, num_classes)
    """
    print(f"Loading preprocessed data from: {data_path}")
    
    # Load arrays
    data = np.load(data_path)
    X_bin = data['X_bin']
    X_gene = data['X_gene']
    X_signatures = data['X_signatures']
    X_rna = data['X_rna']
    y = data['y']
    
    # Load tokenizers
    with open(tokenizer_path, 'rb') as f:
        tokenizers = pickle.load(f)
        tokenizer_bin = tokenizers['tokenizer_bin']
        tokenizer_gene = tokenizers['tokenizer_gene']
    
    # Load metadata
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
        class_weights = metadata['class_weights']
        vocab_sizes = metadata['vocab_sizes']
        num_classes = metadata['num_classes']
    
    print("✓ Data loaded successfully")
    
    return (X_bin, X_gene, X_signatures, X_rna, y, 
            tokenizer_bin, tokenizer_gene, class_weights, 
            vocab_sizes, num_classes)


def find_latest_processed_data(data_dir="processed_data"):
    """
    Find the most recent processed data files.
    
    Args:
        data_dir: Directory to search for processed data
        
    Returns:
        Tuple of (data_path, tokenizer_path, metadata_path) or (None, None, None) if not found
    """
    if not os.path.exists(data_dir):
        return None, None, None
    
    data_files = [f for f in os.listdir(data_dir) if f.startswith("dataset_") and f.endswith(".npz")]
    if not data_files:
        return None, None, None
    
    # Sort by timestamp in filename
    data_files.sort(reverse=True)
    latest_file = data_files[0]
    timestamp = latest_file.replace("dataset_", "").replace(".npz", "")
    
    data_path = os.path.join(data_dir, f"dataset_{timestamp}.npz")
    tokenizer_path = os.path.join(data_dir, f"tokenizers_{timestamp}.pkl")
    metadata_path = os.path.join(data_dir, f"metadata_{timestamp}.pkl")
    
    if os.path.exists(data_path) and os.path.exists(tokenizer_path) and os.path.exists(metadata_path):
        return data_path, tokenizer_path, metadata_path
    
    return None, None, None

