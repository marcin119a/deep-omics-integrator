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
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
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
                
                emb_model = Model(inputs=loaded_model.input, 
                                outputs=loaded_model.get_layer(target_layer).output)
                emb = emb_model.predict(inputs, verbose=1)
                
                print("Computing t-SNE projections with different perplexity values...")
                from sklearn.manifold import TSNE
                
                # Try multiple perplexity values
                # Lower perplexity = more local structure, Higher = more global structure
                perplexity_values = [30, 50, 100]
                tsne_results = {}
                
                for perp in perplexity_values:
                    print(f"  Computing t-SNE with perplexity={perp}...")
                    tsne = TSNE(n_components=2, 
                               perplexity=perp,
                               learning_rate=200,
                               max_iter=2000,  # Changed from n_iter to max_iter
                               random_state=42, 
                               verbose=0)
                    tsne_results[perp] = tsne.fit_transform(emb)
                
                # Use perplexity=50 as default for main visualization
                emb_2d = tsne_results[50]
                
                # Also try UMAP for comparison
                has_umap = False
                emb_2d_umap = None
                try:
                    print("Computing UMAP projection for comparison...")
                    import umap
                    umap_reducer = umap.UMAP(n_components=2, 
                                            n_neighbors=30,
                                            min_dist=0.1,
                                            metric='euclidean',
                                            random_state=42,
                                            verbose=True)
                    emb_2d_umap = umap_reducer.fit_transform(emb)
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
                
                # Plot t-SNE with PAM50 labels
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
                
                tsne_path = os.path.join(output_dir, f"tsne_pam50_perp50_{timestamp}.png")
                plt.savefig(tsne_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✓ t-SNE visualization (perplexity=50) saved to: {tsne_path}")
                
                # Generate comparison plot with all perplexity values
                fig, axes = plt.subplots(1, 3, figsize=(24, 7))
                
                for idx, perp in enumerate(perplexity_values):
                    ax = axes[idx]
                    emb_perp = tsne_results[perp]
                    
                    for i, subtype in enumerate(pam50_categories):
                        mask = y_pam50_labels == subtype
                        if mask.sum() > 0:
                            ax.scatter(emb_perp[mask, 0], emb_perp[mask, 1], 
                                      c=pam50_colors.get(subtype, '#000000'),
                                      label=f'{subtype} (n={mask.sum()})',
                                      alpha=0.6, s=30, edgecolors='black', linewidth=0.3)
                    
                    ax.set_title(f't-SNE (perplexity={perp})', fontsize=14)
                    ax.set_xlabel('Component 1', fontsize=10)
                    ax.set_ylabel('Component 2', fontsize=10)
                    ax.grid(alpha=0.3)
                    if idx == 2:  # Only show legend on last plot
                        ax.legend(loc='best', fontsize=8, framealpha=0.9)
                
                fig.suptitle(f'PAM50 Subtypes - t-SNE Comparison\n{target_layer} Layer Embeddings', 
                           fontsize=16, y=1.02)
                plt.tight_layout()
                
                tsne_comparison_path = os.path.join(output_dir, f"tsne_comparison_{timestamp}.png")
                plt.savefig(tsne_comparison_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✓ t-SNE comparison plot saved to: {tsne_comparison_path}")
                
                # Generate UMAP visualization if available
                if has_umap:
                    plt.figure(figsize=(14, 12))
                    
                    for i, subtype in enumerate(pam50_categories):
                        mask = y_pam50_labels == subtype
                        if mask.sum() > 0:
                            plt.scatter(emb_2d_umap[mask, 0], emb_2d_umap[mask, 1], 
                                      c=pam50_colors.get(subtype, '#000000'),
                                      label=f'{subtype} (n={mask.sum()})',
                                      alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
                    
                    plt.title(f'UMAP Visualization - PAM50 Breast Cancer Subtypes\n{target_layer} Layer Embeddings', 
                             fontsize=16, pad=20)
                    plt.xlabel('UMAP Component 1', fontsize=12)
                    plt.ylabel('UMAP Component 2', fontsize=12)
                    plt.legend(loc='best', fontsize=10, framealpha=0.9)
                    plt.grid(alpha=0.3)
                    plt.tight_layout()
                    
                    umap_path = os.path.join(output_dir, f"umap_pam50_{timestamp}.png")
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

