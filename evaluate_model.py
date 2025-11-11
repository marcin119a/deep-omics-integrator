"""
Script to evaluate a trained model and generate visualizations and metrics.
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


def plot_confusion_matrix(cm, labels, save_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
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
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    for i, metric in enumerate(metrics):
        offset = (i - 1) * width
        ax.bar(x + offset, data[metric], width, label=metric.capitalize())
    
    ax.set_xlabel('Cancer Type', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance Metrics', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Class performance plot saved to: {save_path}")


def evaluate_saved_model(model_path, use_cached=True, output_dir="evaluation_results", 
                        visualize_tsne_flag=True, layer_name='alpha_combination'):
    """
    Evaluate a saved model on the full dataset.
    
    Args:
        model_path: Path to the saved model file
        use_cached: Whether to use cached preprocessed data
        output_dir: Directory to save evaluation results
        visualize_tsne_flag: Whether to generate t-SNE visualization
        layer_name: Layer name for t-SNE embedding extraction
    """
    print(f"\nEvaluating model: {model_path}")
    print("="*80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
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
        
        # Create label map
        label_map = dict(enumerate(filtered_df.groupby('primary_site_cat')['primary_site'].first().sort_index()))
    else:
        # Create label map from cached data (need to reconstruct it)
        print("\nReconstructing label map...")
        df = preprocess.load_mutation_data(config.data_paths['mutations'])
        top_sites = df['primary_site'].value_counts().nlargest(24).index
        filtered_df = df[df['primary_site'].isin(top_sites)].copy()
        filtered_df['primary_site_cat'] = filtered_df['primary_site'].astype('category').cat.codes
        label_map = dict(enumerate(filtered_df.groupby('primary_site_cat')['primary_site'].first().sort_index()))
    
    # Load model (import custom layer first to register it)
    print(f"\nLoading model from: {model_path}")
    from tensorflow import keras
    from src.model import TrainableAlphaLayer  # Import custom layer before loading
    
    # Try loading with automatic registration first, then with custom_objects if needed
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
    inputs = [X_bin, X_gene, X_signatures, X_rna]
    
    # Make predictions
    print("\nGenerating predictions...")
    y_pred_proba = loaded_model.predict(inputs, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate metrics
    print("\n" + "="*80)
    print("EVALUATION METRICS")
    print("="*80)
    
    accuracy = accuracy_score(y, y_pred)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Classification report
    print("\nClassification Report:")
    print("-"*80)
    report = classification_report(y, y_pred, target_names=label_map.values(), digits=4)
    print(report)
    
    # Save classification report
    report_dict = classification_report(y, y_pred, target_names=label_map.values(), 
                                       output_dict=True)
    report_path = os.path.join(output_dir, f"classification_report_{timestamp}.json")
    with open(report_path, 'w') as f:
        json.dump(report_dict, f, indent=4)
    print(f"\n✓ Classification report saved to: {report_path}")
    
    # Save text report
    text_report_path = os.path.join(output_dir, f"classification_report_{timestamp}.txt")
    with open(text_report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"Model Evaluation Report\n")
        f.write(f"Model: {os.path.basename(model_path)}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        f.write(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
        f.write("Classification Report:\n")
        f.write("-"*80 + "\n")
        f.write(report)
    print(f"✓ Text report saved to: {text_report_path}")
    
    # Confusion matrix
    print("\nGenerating confusion matrix...")
    cm = confusion_matrix(y, y_pred)
    cm_path = os.path.join(output_dir, f"confusion_matrix_{timestamp}.png")
    plot_confusion_matrix(cm, list(label_map.values()), cm_path)
    
    # Per-class performance plot
    print("\nGenerating per-class performance plot...")
    perf_path = os.path.join(output_dir, f"class_performance_{timestamp}.png")
    plot_class_performance(report_dict, perf_path)
    
    # t-SNE visualization
    if visualize_tsne_flag:
        print("\nGenerating t-SNE visualization...")
        print(f"Extracting embeddings from layer: {layer_name}")
        
        try:
            # Check if layer exists
            layer_names = [layer.name for layer in loaded_model.layers]
            
            # Try to find the layer (with fallbacks for different naming conventions)
            target_layer = layer_name
            if layer_name not in layer_names:
                # Try alternative names
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
                
                print("Computing t-SNE projection...")
                from sklearn.manifold import TSNE
                tsne = TSNE(n_components=2, random_state=42, verbose=1)
                emb_2d = tsne.fit_transform(emb)
                
                # Plot t-SNE
                plt.figure(figsize=(14, 12))
                scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=y, 
                                    cmap='tab20', alpha=0.6, s=20)
                cbar = plt.colorbar(scatter, ticks=range(len(label_map)))
                cbar.ax.set_yticklabels(list(label_map.values()), fontsize=8)
                plt.title(f't-SNE Visualization of {target_layer} Layer Embeddings', 
                         fontsize=16, pad=20)
                plt.xlabel('t-SNE Component 1', fontsize=12)
                plt.ylabel('t-SNE Component 2', fontsize=12)
                plt.grid(alpha=0.3)
                plt.tight_layout()
                
                tsne_path = os.path.join(output_dir, f"tsne_visualization_{timestamp}.png")
                plt.savefig(tsne_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✓ t-SNE visualization saved to: {tsne_path}")
                
        except Exception as e:
            print(f"Error generating t-SNE visualization: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Classification report (JSON): {os.path.basename(report_path)}")
    print(f"  - Classification report (TXT): {os.path.basename(text_report_path)}")
    print(f"  - Confusion matrix: {os.path.basename(cm_path)}")
    print(f"  - Class performance: {os.path.basename(perf_path)}")
    if visualize_tsne_flag:
        print(f"  - t-SNE visualization: tsne_visualization_{timestamp}.png")
    
    return {
        'accuracy': accuracy,
        'report': report_dict,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'true_labels': y
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a trained deep omics integration model')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to the saved model file (default: latest model)')
    parser.add_argument('--model-name', type=str, default='full_model',
                        help='Model name pattern to search for if --model not specified')
    parser.add_argument('--no-cache', action='store_true',
                        help='Do not use cached preprocessed data')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
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
    evaluate_saved_model(
        model_path=model_path,
        use_cached=not args.no_cache,
        output_dir=args.output_dir,
        visualize_tsne_flag=not args.no_tsne,
        layer_name=args.layer
    )

