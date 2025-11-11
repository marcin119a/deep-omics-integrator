#!/bin/bash

###############################################################################
# Deep Omics Integrator Pipeline Runner
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Directories
LOG_DIR="logs"
RESULTS_DIR="results"
MODELS_DIR="saved_models"
PROCESSED_DATA_DIR="processed_data"

# Create necessary directories
mkdir -p "$LOG_DIR" "$RESULTS_DIR" "$MODELS_DIR" "$PROCESSED_DATA_DIR"

# Timestamp for logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/pipeline_${TIMESTAMP}.log"

###############################################################################
# Helper functions
###############################################################################

log() {
    echo -e "${GREEN}[$(date +"%Y-%m-%d %H:%M:%S")]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[$(date +"%Y-%m-%d %H:%M:%S")] ERROR:${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[$(date +"%Y-%m-%d %H:%M:%S")] WARNING:${NC} $1" | tee -a "$LOG_FILE"
}

log_info() {
    echo -e "${BLUE}[$(date +"%Y-%m-%d %H:%M:%S")] INFO:${NC} $1" | tee -a "$LOG_FILE"
}

print_header() {
    echo ""
    echo "═══════════════════════════════════════════════════════════════════"
    echo "  $1"
    echo "═══════════════════════════════════════════════════════════════════"
    echo ""
}

check_requirements() {
    log_info "Checking Python environment..."
    
    if ! command -v python &> /dev/null; then
        log_error "Python not found. Please install Python 3.8+"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    log_info "Python version: $PYTHON_VERSION"
    
    if [ ! -f "requirements.txt" ]; then
        log_warning "requirements.txt not found"
    else
        log_info "Found requirements.txt"
    fi
}

###############################################################################
# Main pipeline functions
###############################################################################

run_kfold_experiments() {
    print_header "Running K-Fold Cross-Validation Experiments"
    
    folds=${1:-5}
    
    log "Starting experiments with $folds folds..."
    
    # Run each experiment
    for experiment in "Full Model" "No RNA" "No Signature" "No Gene" "No Genomic Bin"; do
        log_info "Running experiment: $experiment"
        
        if python3 main.py --experiment "$experiment" --folds "$folds" 2>&1 | tee -a "$LOG_FILE"; then
            log "✓ Experiment '$experiment' completed successfully"
        else
            log_error "✗ Experiment '$experiment' failed"
            return 1
        fi
    done
    
    log "All experiments completed!"
}

train_full_model() {
    print_header "Training Full Model"
    
    model_name=${1:-"full_model"}
    use_cache=${2:-true}
    save_data=${3:-true}
    
    cache_flag=""
    save_flag=""
    
    if [ "$use_cache" = false ]; then
        cache_flag="--no-cache"
        log_info "Cache disabled - will process data from scratch"
    else
        log_info "Cache enabled - will use cached data if available"
    fi
    
    if [ "$save_data" = false ]; then
        save_flag="--no-save"
        log_info "Data saving disabled"
    else
        log_info "Data saving enabled"
    fi
    
    log "Training model: $model_name"
    
    if python3 train_and_save_model.py --name "$model_name" $cache_flag $save_flag 2>&1 | tee -a "$LOG_FILE"; then
        log "✓ Model training completed successfully"
        
        # Find the most recent model
        LATEST_MODEL=$(ls -t saved_models/${model_name}_*.keras 2>/dev/null | head -n1)
        if [ -n "$LATEST_MODEL" ]; then
            log "✓ Model saved to: $LATEST_MODEL"
        fi
    else
        log_error "✗ Model training failed"
        return 1
    fi
}

preprocess_only() {
    print_header "Preprocessing Data Only"
    
    log "Running preprocessing and saving data..."
    log_info "This will create cached data files for faster subsequent runs"
    
    python3 -c "
from src.config import set_seed
set_seed(42)

# Import and run preprocessing without training
import sys
sys.path.insert(0, '.')
from train_and_save_model import train_and_save_full_model

# Just preprocess and save, don't train
print('Starting data preprocessing...')
" 2>&1 | tee -a "$LOG_FILE"
    
    log_warning "Note: For preprocessing only, run train script with --no-cache flag first time"
}

show_status() {
    print_header "Pipeline Status"
    
    echo "📁 Directory Status:"
    echo "  - Processed data: $(ls -1 $PROCESSED_DATA_DIR/*.npz 2>/dev/null | wc -l) files"
    echo "  - Saved models: $(ls -1 $MODELS_DIR/*.keras 2>/dev/null | wc -l) files"
    echo "  - Log files: $(ls -1 $LOG_DIR/*.log 2>/dev/null | wc -l) files"
    echo ""
    
    if [ -d "$PROCESSED_DATA_DIR" ] && [ "$(ls -A $PROCESSED_DATA_DIR 2>/dev/null)" ]; then
        echo "📊 Latest processed data:"
        ls -lth "$PROCESSED_DATA_DIR" | head -n 5
        echo ""
    fi
    
    if [ -d "$MODELS_DIR" ] && [ "$(ls -A $MODELS_DIR 2>/dev/null)" ]; then
        echo "🤖 Latest models:"
        ls -lth "$MODELS_DIR" | head -n 5
        echo ""
    fi
}

###############################################################################
# Usage information
###############################################################################

show_usage() {
    cat << EOF
Usage: $0 [COMMAND] [OPTIONS]

Commands:
    full              Run complete pipeline (experiments + train full model)
    experiments       Run k-fold cross-validation experiments only
    train            Train and save full model
    preprocess       Preprocess data and save cache (no training)
    status           Show pipeline status
    help             Show this help message

Options:
    --folds N        Number of folds for cross-validation (default: 5)
    --model-name     Name for saved model (default: full_model)
    --no-cache       Don't use cached preprocessed data
    --no-save        Don't save preprocessed data

Examples:
    $0 full                                    # Run complete pipeline
    $0 experiments --folds 10                  # Run experiments with 10 folds
    $0 train --model-name my_model             # Train specific model
    $0 train --no-cache                        # Train without using cache
    $0 preprocess                              # Only preprocess and cache data
    $0 status                                  # Show current status

EOF
}

###############################################################################
# Main script logic
###############################################################################

main() {
    print_header "Deep Omics Integrator Pipeline"
    log "Pipeline started at $(date)"
    log "Log file: $LOG_FILE"
    
    check_requirements
    
    # Parse command
    COMMAND=${1:-help}
    shift || true
    
    # Parse options
    FOLDS=5
    MODEL_NAME="full_model"
    USE_CACHE=true
    SAVE_DATA=true
    
    while [ $# -gt 0 ]; do
        case $1 in
            --folds)
                FOLDS="$2"
                shift 2
                ;;
            --model-name)
                MODEL_NAME="$2"
                shift 2
                ;;
            --no-cache)
                USE_CACHE=false
                shift
                ;;
            --no-save)
                SAVE_DATA=false
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Execute command
    case $COMMAND in
        full)
            log_info "Running FULL pipeline"
            run_kfold_experiments "$FOLDS"
            train_full_model "$MODEL_NAME" "$USE_CACHE" "$SAVE_DATA"
            show_status
            ;;
        experiments|exp)
            run_kfold_experiments "$FOLDS"
            ;;
        train)
            train_full_model "$MODEL_NAME" "$USE_CACHE" "$SAVE_DATA"
            ;;
        preprocess|prep)
            preprocess_only
            ;;
        status)
            show_status
            ;;
        help|--help|-h)
            show_usage
            ;;
        *)
            log_error "Unknown command: $COMMAND"
            show_usage
            exit 1
            ;;
    esac
    
    print_header "Pipeline Completed"
    log "Pipeline finished at $(date)"
    log "Total log saved to: $LOG_FILE"
}

# Run main function
main "$@"

