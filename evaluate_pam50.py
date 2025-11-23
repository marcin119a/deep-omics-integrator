"""
Script to evaluate a trained model on PAM50 breast cancer subtypes.
This script filters the dataset to only include PAM50 patients and
generates t-SNE visualization with PAM50 subtype labels.
"""
import pandas as pd
import numpy as np
from Bio import SeqIO
import pyranges as pr
import argparse
import os
import json
import pickle
import glob
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from src.config import set_seed

from src import preprocess, annotation, tokenizer_utils, model, evaluation, config, data_cache


def find_latest_model(model_dir="saved_models", model_pattern="full_model"):
    """Find the most recent model file."""
    if not os.path.exists(model_dir):
        return None
    
    # Try with both underscore and hyphen
    patterns_to_try = [model_pattern, model_pattern.replace('_', '-'), model_pattern.replace('-', '_')]
    
    model_files = []
    for pattern in patterns_to_try:
        files = [f for f in os.listdir(model_dir) 
                 if f.startswith(pattern) and f.endswith(".keras")]
        model_files.extend(files)
    
    # Remove duplicates
    model_files = list(set(model_files))
    
    if not model_files:
        return None
    
    # Sort by modification time
    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
    return os.path.join(model_dir, model_files[0])


def find_latest_test_dataset(test_data_dir="processed_data_no_pam50"):
    """Find the most recent test dataset file."""
    if not os.path.exists(test_data_dir):
        return None, None
    
    test_files = [f for f in os.listdir(test_data_dir) 
                  if f.startswith("test_dataset_") and f.endswith(".npz")]
    if not test_files:
        return None, None
    
    # Sort by timestamp in filename
    test_files.sort(reverse=True)
    latest_file = test_files[0]
    timestamp = latest_file.replace("test_dataset_", "").replace(".npz", "")
    
    test_data_path = os.path.join(test_data_dir, f"test_dataset_{timestamp}.npz")
    test_metadata_path = os.path.join(test_data_dir, f"test_metadata_{timestamp}.pkl")
    
    if os.path.exists(test_data_path):
        return test_data_path, test_metadata_path if os.path.exists(test_metadata_path) else None
    
    return None, None


def load_test_dataset(test_data_path):
    """Load test dataset from npz file."""
    print(f"Loading test dataset from: {test_data_path}")
    data = np.load(test_data_path, allow_pickle=True)
    
    X_bin = data['X_bin']
    X_gene = data['X_gene']
    X_signatures = data['X_signatures']
    X_rna = data['X_rna']
    y = data['y']
    case_barcodes = data['case_barcodes'] if 'case_barcodes' in data else None
    
    print(f"✓ Loaded test dataset: {len(y)} samples")
    return X_bin, X_gene, X_signatures, X_rna, y, case_barcodes


def load_pam50_data(pam50_path="data/pam50.csv"):
    """Load PAM50 subtype data."""
    print(f"Loading PAM50 data from: {pam50_path}")
    pam50_df = pd.read_csv(pam50_path)
    
    # Extract case_barcode from bcr_patient_barcode (first 12 characters)
    pam50_df['case_barcode'] = pam50_df['bcr_patient_barcode'].str[:12]
    
    print(f"✓ Loaded {len(pam50_df)} PAM50 samples")
    print(f"  PAM50 subtype distribution:")
    print(pam50_df['PAM50'].value_counts().sort_index())
    
    return pam50_df


def plot_confusion_matrix(cm, labels, save_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - PAM50 Subtypes', fontsize=16, pad=20)
    plt.ylabel('True PAM50 Subtype', fontsize=12)
    plt.xlabel('Predicted Primary Site', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix saved to: {save_path}")


def plot_class_performance(report_dict, save_path):
    """Plot per-class performance metrics."""
    # Extract per-class metrics
    classes = [k for k in report_dict.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
    
    metrics = ['precision', 'recall', 'f1-score']
    data = {metric: [report_dict[cls][metric] for cls in classes] for metric in metrics}
    
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, metric in enumerate(metrics):
        offset = (i - 1) * width
        ax.bar(x + offset, data[metric], width, label=metric.capitalize())
    
    ax.set_xlabel('PAM50 Subtype', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Subtype Performance Metrics', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Class performance plot saved to: {save_path}")


def evaluate_pam50_model(model_path, pam50_path="data/pam50.csv", use_cached=True, 
                         output_dir="evaluation_results_pam50", 
                         visualize_tsne_flag=True, layer_name='alpha_combination'):
    """
    Evaluate a saved model on PAM50 breast cancer subtype data.
    
    Args:
        model_path: Path to the saved model file
        pam50_path: Path to PAM50 CSV file
        use_cached: Whether to use cached preprocessed data
        output_dir: Directory to save evaluation results
        visualize_tsne_flag: Whether to generate t-SNE visualization
        layer_name: Layer name for t-SNE embedding extraction
    """
    print(f"\nEvaluating model on PAM50 subtypes: {model_path}")
    print("="*80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load PAM50 data
    pam50_df = load_pam50_data(pam50_path)
    pam50_case_barcodes = set(pam50_df['case_barcode'].unique())
    
    # Load or process data
    data_loaded = False
    if use_cached:
        data_path, tokenizer_path, metadata_path = data_cache.find_latest_processed_data()
        if data_path:
            try:
                print("\nLoading cached preprocessed data...")
                (X_bin, X_gene, X_signatures, X_rna, y, 
                 tokenizer_bin, tokenizer_gene, class_weights, 
                 vocab_sizes, num_classes) = data_cache.load_processed_data(
                    data_path, tokenizer_path, metadata_path)
                data_loaded = True
            except Exception as e:
                print(f"Error loading cached data: {e}")
                print("Will process data from scratch...")
    
    # If data not loaded from cache, process it
    if not data_loaded:
        print("\nProcessing data from scratch...")
        
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
        
        # Store case_barcodes for filtering
        all_case_barcodes = grouped_df['case_barcode'].values
    else:
        # Need to load case_barcodes from original data
        print("\nLoading case_barcodes for filtering...")
        df = preprocess.load_mutation_data(config.data_paths['mutations'])
        expression_df = preprocess.load_expression_data(config.data_paths['expression'])
        signature_df = preprocess.load_signatures_data(config.data_paths['signatures'])
        
        top_sites = df['primary_site'].value_counts().nlargest(24).index
        filtered_df = df[df['primary_site'].isin(top_sites)].copy()
        filtered_df['primary_site_cat'] = filtered_df['primary_site'].astype('category').cat.codes
        
        grouped_mutations_df = filtered_df.groupby('case_barcode').agg({
            'primary_site': 'first',
            'primary_site_cat': 'first',
        }).reset_index()
        
        grouped_expression_df = expression_df.groupby('case_barcode').agg({
            'gene_name': list,
        }).reset_index()
        
        grouped_df = pd.merge(grouped_mutations_df, grouped_expression_df, on="case_barcode")
        grouped_df = pd.merge(grouped_df, signature_df, left_on="case_barcode", right_on="Samples")
        
        all_case_barcodes = grouped_df['case_barcode'].values
    
    # Filter to only PAM50 patients
    print(f"\nFiltering to PAM50 patients only...")
    print(f"Total samples before filtering: {len(all_case_barcodes)}")
    
    # Create mask for PAM50 samples
    pam50_mask = np.array([cb in pam50_case_barcodes for cb in all_case_barcodes])
    
    print(f"PAM50 samples found: {pam50_mask.sum()}")
    
    if pam50_mask.sum() == 0:
        print("ERROR: No PAM50 samples found in the dataset!")
        return None
    
    # Filter all data
    X_bin_pam50 = X_bin[pam50_mask]
    X_gene_pam50 = X_gene[pam50_mask]
    X_signatures_pam50 = X_signatures[pam50_mask]
    X_rna_pam50 = X_rna[pam50_mask]
    y_pam50 = y[pam50_mask]
    case_barcodes_pam50 = all_case_barcodes[pam50_mask]
    
    # Create mapping from case_barcode to PAM50 subtype
    pam50_label_map = dict(zip(pam50_df['case_barcode'], pam50_df['PAM50']))
    
    # Create PAM50 labels
    y_pam50_labels = np.array([pam50_label_map.get(cb, 'Unknown') for cb in case_barcodes_pam50])
    
    # Create PAM50 category codes
    pam50_categories = sorted(pam50_df['PAM50'].unique())
    pam50_cat_to_code = {cat: i for i, cat in enumerate(pam50_categories)}
    y_pam50_cat = np.array([pam50_cat_to_code[label] for label in y_pam50_labels])
    
    print(f"\nPAM50 subtype distribution in test set:")
    unique, counts = np.unique(y_pam50_labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  {u}: {c}")
    
    # Create label map for primary sites (for prediction interpretation)
    if not data_loaded:
        label_map = dict(enumerate(filtered_df.groupby('primary_site_cat')['primary_site'].first().sort_index()))
    else:
        df = preprocess.load_mutation_data(config.data_paths['mutations'])
        top_sites = df['primary_site'].value_counts().nlargest(24).index
        filtered_df = df[df['primary_site'].isin(top_sites)].copy()
        filtered_df['primary_site_cat'] = filtered_df['primary_site'].astype('category').cat.codes
        label_map = dict(enumerate(filtered_df.groupby('primary_site_cat')['primary_site'].first().sort_index()))
    
    # Load test dataset if available
    print("\nTrying to load test dataset...")
    test_data_path, test_metadata_path = find_latest_test_dataset()
    X_test_bin, X_test_gene, X_test_signatures, X_test_rna = None, None, None, None
    y_test = None
    case_barcodes_test = None
    
    if test_data_path:
        try:
            X_test_bin, X_test_gene, X_test_signatures, X_test_rna, y_test, case_barcodes_test = load_test_dataset(test_data_path)
            
            # Load test metadata if available
            if test_metadata_path and os.path.exists(test_metadata_path):
                with open(test_metadata_path, 'rb') as f:
                    test_metadata = pickle.load(f)
                    print(f"  Test dataset timestamp: {test_metadata.get('timestamp', 'unknown')}")
                    print(f"  Test samples: {test_metadata.get('n_test_samples', 'unknown')}")
        except Exception as e:
            print(f"Warning: Could not load test dataset: {e}")
            print("  Will use all available data as background")
    else:
        print("  No test dataset found. Will use all available data as background")
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    from tensorflow import keras
    from src.model import TrainableAlphaLayer
    
    try:
        loaded_model = keras.models.load_model(model_path)
    except (TypeError, ValueError) as e:
        print(f"Note: Loading with custom_objects (older model format)")
        loaded_model = keras.models.load_model(
            model_path, 
            custom_objects={'TrainableAlphaLayer': TrainableAlphaLayer}
        )
    print("✓ Model loaded successfully")
    
    # Prepare inputs
    inputs = [X_bin_pam50, X_gene_pam50, X_signatures_pam50, X_rna_pam50]
    
    # Make predictions
    print("\nGenerating predictions...")
    y_pred_proba = loaded_model.predict(inputs, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Map predictions to primary site names
    y_pred_names = np.array([label_map[p] for p in y_pred])
    
    # Calculate metrics
    print("\n" + "="*80)
    print("EVALUATION METRICS - PAM50 Subtypes")
    print("="*80)
    
    # Since we're predicting primary sites but have PAM50 labels, we need to check
    # how many breast cancer predictions we get
    breast_mask = y_pred_names == 'Breast'
    breast_accuracy = breast_mask.sum() / len(breast_mask)
    print(f"\nBreast cancer detection accuracy: {breast_accuracy:.4f} ({breast_accuracy*100:.2f}%)")
    print(f"  Correctly identified as Breast: {breast_mask.sum()}/{len(breast_mask)}")
    print(f"  Misclassified as other sites: {(~breast_mask).sum()}")
    
    if (~breast_mask).sum() > 0:
        print(f"\nMisclassification breakdown:")
        misclass_sites = y_pred_names[~breast_mask]
        unique_sites, counts = np.unique(misclass_sites, return_counts=True)
        for site, count in sorted(zip(unique_sites, counts), key=lambda x: x[1], reverse=True):
            print(f"  {site}: {count}")
    
    # Save detailed results
    results_df = pd.DataFrame({
        'case_barcode': case_barcodes_pam50,
        'PAM50_subtype': y_pam50_labels,
        'predicted_site': y_pred_names,
        'is_breast': breast_mask
    })
    results_path = os.path.join(output_dir, f"pam50_predictions_{timestamp}.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Detailed predictions saved to: {results_path}")
    
    # Save classification report grouped by PAM50 subtype
    print("\nBreast detection by PAM50 subtype:")
    print("-"*80)
    for subtype in pam50_categories:
        subtype_mask = y_pam50_labels == subtype
        if subtype_mask.sum() > 0:
            subtype_breast_acc = breast_mask[subtype_mask].sum() / subtype_mask.sum()
            print(f"{subtype:8s}: {breast_mask[subtype_mask].sum():3d}/{subtype_mask.sum():3d} = {subtype_breast_acc:.4f}")
    
    # Initialize variables for return
    emb_2d = None
    has_umap = False
    target_layer = None
    
    # t-SNE visualization with PAM50 labels
    if visualize_tsne_flag:
        print("\nGenerating t-SNE visualization with PAM50 labels...")
        print(f"Extracting embeddings from layer: {layer_name}")
        
        try:
            # Check if layer exists
            layer_names = [layer.name for layer in loaded_model.layers]
            
            # Try to find the layer
            target_layer = layer_name
            if layer_name not in layer_names:
                alternatives = ['trainable_alpha_layer', 'alpha_combination']
                for alt_name in alternatives:
                    if alt_name in layer_names:
                        target_layer = alt_name
                        print(f"Note: Using layer '{target_layer}' instead of '{layer_name}'")
                        break
                else:
                    print(f"Warning: Layer '{layer_name}' not found in model.")
                    print(f"Available layers: {layer_names}")
                    print("Skipping t-SNE visualization.")
                    target_layer = None
            
            if target_layer:
                # Generate t-SNE visualization
                from tensorflow.keras.models import Model
                
                emb_model = Model(
                    inputs=loaded_model.input,
                    outputs=loaded_model.get_layer(target_layer).output
                )

                # Embeddings for PAM50 subset (for t-SNE)
                emb_pam50 = emb_model.predict(inputs, verbose=1)
                
                # Prepare data for visualization: test dataset + PAM50
                emb_test = None
                y_test_labels = None
                test_inputs = None
                
                if X_test_bin is not None:
                    print("\nPreparing test dataset for visualization...")
                    test_inputs = [X_test_bin, X_test_gene, X_test_signatures, X_test_rna]
                    emb_test = emb_model.predict(test_inputs, verbose=1)
                    
                    # Map test labels to primary site names
                    y_test_labels = np.array([label_map.get(int(y_val), 'Unknown') for y_val in y_test])
                    print(f"  Test dataset embeddings: {emb_test.shape}")
                
                # Combine embeddings for joint visualization
                if emb_test is not None:
                    emb_combined = np.vstack([emb_test, emb_pam50])
                    print(f"  Combined embeddings: {emb_combined.shape} (test: {len(emb_test)}, PAM50: {len(emb_pam50)})")
                else:
                    emb_combined = emb_pam50
                    print(f"  Using PAM50 embeddings only: {emb_combined.shape}")
                
                print("\nComputing t-SNE projections...")
                from sklearn.manifold import TSNE
                
                # Compute t-SNE on combined data
                print("  Computing t-SNE on combined data...")
                tsne = TSNE(
                    n_components=2,
                    perplexity=min(30, (len(emb_combined)-1)//3),  # Adjust perplexity to data size
                    learning_rate=300,
                    early_exaggeration=15,
                    metric='cosine',
                    init='pca',
                    random_state=42,
                    verbose=1
                )
                emb_2d_combined = tsne.fit_transform(emb_combined)
                
                # Split back into test and PAM50
                if emb_test is not None:
                    n_test = len(emb_test)
                    emb_2d_test = emb_2d_combined[:n_test]
                    emb_2d = emb_2d_combined[n_test:]
                else:
                    emb_2d = emb_2d_combined
                    emb_2d_test = None
                
                # Also try UMAP for comparison
                has_umap = False
                emb_2d_umap = None
                try:
                    print("Computing UMAP projection for comparison...")
                    import umap
                    umap_reducer = umap.UMAP(
                        n_components=2,
                        n_neighbors=10,
                        min_dist=0.01,
                        metric='cosine',
                        random_state=42,
                        verbose=True
                    )

                    # Compute embeddings for test dataset + PAM50
                    print("Computing embeddings for test dataset + PAM50 for UMAP...")
                    if X_test_bin is not None:
                        # Combine test and PAM50 for UMAP
                        inputs_combined = [
                            np.vstack([X_test_bin, X_bin_pam50]),
                            np.vstack([X_test_gene, X_gene_pam50]),
                            np.vstack([X_test_signatures, X_signatures_pam50]),
                            np.vstack([X_test_rna, X_rna_pam50])
                        ]
                        emb_combined_umap = emb_model.predict(inputs_combined, verbose=1)
                        
                        # Fit UMAP on combined data
                        emb_2d_combined_umap = umap_reducer.fit_transform(emb_combined_umap)
                        
                        # Split back
                        n_test_umap = len(X_test_bin)
                        emb_2d_test_umap = emb_2d_combined_umap[:n_test_umap]
                        emb_2d_umap = emb_2d_combined_umap[n_test_umap:]
                        
                        # Get test labels for UMAP visualization
                        y_test_labels_umap = y_test_labels if y_test_labels is not None else None
                    else:
                        # Use all data as before
                        inputs_all = [X_bin, X_gene, X_signatures, X_rna]
                        emb_all = emb_model.predict(inputs_all, verbose=1)
                        emb_2d_all = umap_reducer.fit_transform(emb_all)
                        emb_2d_umap = emb_2d_all[pam50_mask]
                        emb_2d_test_umap = None
                        y_test_labels_umap = None
                    
                    has_umap = True
                except ImportError:
                    print("UMAP not available, skipping. Install with: pip install umap-learn")
                except Exception as e:
                    print(f"UMAP error: {e}")
                
                # Create color map for PAM50 subtypes
                pam50_colors = {
                    'Basal': '#E41A1C',
                    'Her2': '#377EB8',
                    'LumA': '#4DAF4A',
                    'LumB': '#984EA3',
                    'Normal': '#FF7F00'
                }
                
                # Plot t-SNE with test dataset as background and PAM50 on top
                plt.figure(figsize=(16, 12))
                
                # Plot test dataset as background (grouped by primary site)
                if emb_2d_test is not None and y_test_labels is not None:
                    unique_test_sites = np.unique(y_test_labels)
                    # Use lighter colors for test dataset
                    test_colors_map = cm.get_cmap('tab20')(np.linspace(0, 1, len(unique_test_sites)))
                    
                    for site in unique_test_sites:
                        site_mask = y_test_labels == site
                        if site_mask.sum() > 0:
                            plt.scatter(
                                emb_2d_test[site_mask, 0], 
                                emb_2d_test[site_mask, 1], 
                                c=[test_colors_map[np.where(unique_test_sites == site)[0][0]]],
                                label=f'Test: {site}' if site != 'Breast' else None,  # Skip Breast in legend
                                alpha=0.2,
                                s=20,
                                edgecolors='none',
                                marker='o'
                            )
                
                # Plot PAM50 subtypes on top with distinct colors
                for i, subtype in enumerate(pam50_categories):
                    mask = y_pam50_labels == subtype
                    if mask.sum() > 0:
                        plt.scatter(emb_2d[mask, 0], emb_2d[mask, 1], 
                                  c=pam50_colors.get(subtype, '#000000'),
                                  label=f'PAM50: {subtype} (n={mask.sum()})',
                                  alpha=0.8, s=80, edgecolors='black', linewidth=1.5, marker='^')
                
                plt.title(f't-SNE: PAM50 Breast Subtypes (colored triangles) vs Test Dataset (gray background)\n{target_layer} Layer Embeddings', 
                         fontsize=16, pad=20)
                plt.xlabel('t-SNE Component 1', fontsize=12)
                plt.ylabel('t-SNE Component 2', fontsize=12)
                plt.legend(loc='best', fontsize=9, framealpha=0.9, ncol=2)
                plt.grid(alpha=0.3)
                plt.tight_layout()
                
                tsne_path = os.path.join(output_dir, f"tsne_pam50_with_test_background_{timestamp}.png")
                plt.savefig(tsne_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✓ t-SNE visualization with test background saved to: {tsne_path}")
                
                # Also create a version showing only PAM50 (original plot)
                plt.figure(figsize=(14, 12))
                
                for i, subtype in enumerate(pam50_categories):
                    mask = y_pam50_labels == subtype
                    if mask.sum() > 0:
                        plt.scatter(emb_2d[mask, 0], emb_2d[mask, 1], 
                                  c=pam50_colors.get(subtype, '#000000'),
                                  label=f'{subtype} (n={mask.sum()})',
                                  alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
                
                plt.title(f't-SNE Visualization - PAM50 Breast Cancer Subtypes\n{target_layer} Layer Embeddings', 
                         fontsize=16, pad=20)
                plt.xlabel('t-SNE Component 1', fontsize=12)
                plt.ylabel('t-SNE Component 2', fontsize=12)
                plt.legend(loc='best', fontsize=10, framealpha=0.9)
                plt.grid(alpha=0.3)
                plt.tight_layout()
                
                tsne_pam50_only_path = os.path.join(output_dir, f"tsne_pam50_only_{timestamp}.png")
                plt.savefig(tsne_pam50_only_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✓ t-SNE visualization (PAM50 only) saved to: {tsne_pam50_only_path}")
                
                # Generate comparison plot showing test dataset grouped by primary site vs PAM50
                fig, axes = plt.subplots(1, 2, figsize=(20, 8))
                
                # Left plot: Test dataset grouped by primary site (excluding Breast)
                ax = axes[0]
                if emb_2d_test is not None and y_test_labels is not None:
                    unique_test_sites = np.unique(y_test_labels)
                    test_colors_map = cm.get_cmap('tab20')(np.linspace(0, 1, len(unique_test_sites)))
                    
                    for site in unique_test_sites:
                        if site == 'Breast':
                            continue  # Skip Breast as we'll show PAM50 separately
                        site_mask = y_test_labels == site
                        if site_mask.sum() > 0:
                            ax.scatter(
                                emb_2d_test[site_mask, 0], 
                                emb_2d_test[site_mask, 1], 
                                c=[test_colors_map[np.where(unique_test_sites == site)[0][0]]],
                                label=f'{site} (n={site_mask.sum()})',
                                alpha=0.5,
                                s=30,
                                edgecolors='none'
                            )
                
                ax.set_title('Test Dataset (Non-Breast Primary Sites)', fontsize=14)
                ax.set_xlabel('t-SNE Component 1', fontsize=10)
                ax.set_ylabel('t-SNE Component 2', fontsize=10)
                ax.grid(alpha=0.3)
                ax.legend(loc='best', fontsize=7, framealpha=0.9, ncol=2)
                
                # Right plot: PAM50 subtypes
                ax = axes[1]
                for i, subtype in enumerate(pam50_categories):
                    mask = y_pam50_labels == subtype
                    if mask.sum() > 0:
                        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1], 
                                  c=pam50_colors.get(subtype, '#000000'),
                                  label=f'{subtype} (n={mask.sum()})',
                                  alpha=0.8, s=60, edgecolors='black', linewidth=1)
                
                ax.set_title('PAM50 Breast Cancer Subtypes', fontsize=14)
                ax.set_xlabel('t-SNE Component 1', fontsize=10)
                ax.set_ylabel('t-SNE Component 2', fontsize=10)
                ax.grid(alpha=0.3)
                ax.legend(loc='best', fontsize=8, framealpha=0.9)
                
                fig.suptitle(f'Comparison: Test Dataset vs PAM50 Subtypes\n{target_layer} Layer Embeddings', 
                           fontsize=16, y=1.02)
                plt.tight_layout()
                
                tsne_comparison_path = os.path.join(output_dir, f"tsne_comparison_test_vs_pam50_{timestamp}.png")
                plt.savefig(tsne_comparison_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✓ t-SNE comparison plot saved to: {tsne_comparison_path}")
                
                # Generate UMAP visualization if available
                if has_umap:
                    plt.figure(figsize=(16, 12))

                    # Plot test dataset as background
                    if emb_2d_test_umap is not None and y_test_labels_umap is not None:
                        unique_test_sites = np.unique(y_test_labels_umap)
                        test_colors_map = cm.get_cmap('tab20')(np.linspace(0, 1, len(unique_test_sites)))
                        
                        for site in unique_test_sites:
                            if site == 'Breast':
                                continue  # Skip Breast as we'll show PAM50 separately
                            site_mask = y_test_labels_umap == site
                            if site_mask.sum() > 0:
                                plt.scatter(
                                    emb_2d_test_umap[site_mask, 0],
                                    emb_2d_test_umap[site_mask, 1],
                                    c=[test_colors_map[np.where(unique_test_sites == site)[0][0]]],
                                    label=f'Test: {site}',
                                    alpha=0.2,
                                    s=20,
                                    edgecolors='none'
                                )
                    elif emb_2d_test_umap is None:
                        # Fallback: use gray background if test data not available
                        non_pam_mask = ~pam50_mask
                        if non_pam_mask.sum() > 0:
                            plt.scatter(
                                emb_2d_all[non_pam_mask, 0],
                                emb_2d_all[non_pam_mask, 1],
                                c='#D3D3D3',
                                label=f'Other primary sites (n={non_pam_mask.sum()})',
                                alpha=0.3,
                                s=20
                            )

                    # Plot PAM50 subtypes on top
                    for i, subtype in enumerate(pam50_categories):
                        mask = y_pam50_labels == subtype
                        if mask.sum() > 0:
                            plt.scatter(
                                emb_2d_umap[mask, 0],
                                emb_2d_umap[mask, 1],
                                c=pam50_colors.get(subtype, '#000000'),
                                label=f'PAM50: {subtype} (n={mask.sum()})',
                                alpha=0.8,
                                s=80,
                                edgecolors='black',
                                linewidth=1.5,
                                marker='^'
                            )

                    plt.title(
                        'UMAP: PAM50 Breast Subtypes (colored triangles) vs Test Dataset (gray background)\n'
                        f'{target_layer} Layer Embeddings',
                        fontsize=16,
                        pad=20
                    )
                    plt.xlabel('UMAP Component 1', fontsize=12)
                    plt.ylabel('UMAP Component 2', fontsize=12)
                    plt.legend(loc='best', fontsize=9, framealpha=0.9, ncol=2)
                    plt.grid(alpha=0.3)
                    plt.tight_layout()

                    umap_path = os.path.join(output_dir, f"umap_pam50_with_test_background_{timestamp}.png")
                    plt.savefig(umap_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"✓ UMAP visualization saved to: {umap_path}")
                
                # Also create a version showing misclassifications
                plt.figure(figsize=(14, 12))
                
                # Plot correctly classified as breast in green, misclassified in red
                correct_mask = breast_mask
                incorrect_mask = ~breast_mask
                
                if correct_mask.sum() > 0:
                    plt.scatter(emb_2d[correct_mask, 0], emb_2d[correct_mask, 1], 
                              c='green', label=f'Correct (Breast) (n={correct_mask.sum()})',
                              alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
                
                if incorrect_mask.sum() > 0:
                    plt.scatter(emb_2d[incorrect_mask, 0], emb_2d[incorrect_mask, 1], 
                              c='red', label=f'Incorrect (Other) (n={incorrect_mask.sum()})',
                              alpha=0.6, s=50, edgecolors='black', linewidth=0.5, marker='x')
                
                plt.title(f't-SNE Visualization - Breast Cancer Detection\n{target_layer} Layer Embeddings', 
                         fontsize=16, pad=20)
                plt.xlabel('t-SNE Component 1', fontsize=12)
                plt.ylabel('t-SNE Component 2', fontsize=12)
                plt.legend(loc='best', fontsize=10, framealpha=0.9)
                plt.grid(alpha=0.3)
                plt.tight_layout()
                
                tsne_class_path = os.path.join(output_dir, f"tsne_classification_{timestamp}.png")
                plt.savefig(tsne_class_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✓ t-SNE classification visualization saved to: {tsne_class_path}")
                
        except Exception as e:
            print(f"Error generating t-SNE visualization: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Detailed predictions: {os.path.basename(results_path)}")
    if visualize_tsne_flag:
        print(f"  - t-SNE PAM50 visualization: tsne_pam50_{timestamp}.png")
        try:
            if has_umap:
                print(f"  - UMAP PAM50 visualization: umap_pam50_{timestamp}.png")
        except:
            pass
        print(f"  - t-SNE classification visualization: tsne_classification_{timestamp}.png")
    
    return {
        'breast_accuracy': breast_accuracy,
        'n_samples': len(case_barcodes_pam50),
        'n_correct': breast_mask.sum(),
        'predictions': y_pred_names,
        'pam50_labels': y_pam50_labels,
        'embeddings_2d': emb_2d if visualize_tsne_flag and target_layer else None
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model on PAM50 breast cancer subtypes')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to the saved model file (default: latest model)')
    parser.add_argument('--model-name', type=str, default='full_model',
                        help='Model name pattern to search for if --model not specified')
    parser.add_argument('--pam50', type=str, default='data/pam50.csv',
                        help='Path to PAM50 CSV file')
    parser.add_argument('--no-cache', action='store_true',
                        help='Do not use cached preprocessed data')
    parser.add_argument('--output-dir', type=str, default='evaluation_results_pam50',
                        help='Directory to save evaluation results')
    parser.add_argument('--no-tsne', action='store_true',
                        help='Skip t-SNE visualization')
    parser.add_argument('--layer', type=str, default='alpha_combination',
                        help='Layer name for t-SNE embedding extraction')
    args = parser.parse_args()
    
    set_seed(42)  # Set seed for reproducibility
    
    # Find model
    if args.model:
        model_path = args.model
        if not os.path.exists(model_path):
            print(f"Error: Model file not found: {model_path}")
            exit(1)
    else:
        print(f"Searching for latest model matching pattern: {args.model_name}")
        model_path = find_latest_model(model_pattern=args.model_name)
        if not model_path:
            print(f"Error: No model found matching pattern: {args.model_name}")
            print("Please train a model first using train_and_save_model.py")
            exit(1)
    
    # Run evaluation
    evaluate_pam50_model(
        model_path=model_path,
        pam50_path=args.pam50,
        use_cached=not args.no_cache,
        output_dir=args.output_dir,
        visualize_tsne_flag=not args.no_tsne,
        layer_name=args.layer
    )

