#!/bin/bash

# ProviderVerify Docker Entrypoint Script

set -e

echo "Starting ProviderVerify..."

# Function to run pipeline
run_pipeline() {
    echo "Running ProviderVerify pipeline..."
    python -m src.pipeline.run_provider_verify "$@"
}

# Function to run audit UI
run_audit() {
    echo "Starting ProviderVerify Audit UI..."
    streamlit run src/audit/ui.py --server.port=8501 --server.address=0.0.0.0
}

# Function to run dashboard
run_dashboard() {
    echo "Starting ProviderVerify Dashboard..."
    streamlit run src/reporting/dashboard.py --server.port=8502 --server.address=0.0.0.0
}

# Function to train ML model
train_model() {
    echo "Training ML model..."
    python -m src.match.ml_model.train "$@"
}

# Function to start shell
start_shell() {
    echo "Starting interactive shell..."
    exec /bin/bash
}

# Main execution logic
case "$1" in
    "pipeline")
        shift
        run_pipeline "$@"
        ;;
    "audit")
        run_audit
        ;;
    "dashboard")
        run_dashboard
        ;;
    "train")
        shift
        train_model "$@"
        ;;
    "shell")
        start_shell
        ;;
    "help"|"-h"|"--help")
        echo "ProviderVerify Docker Commands:"
        echo "  pipeline [args]     - Run the entity resolution pipeline"
        echo "  audit               - Start the audit UI (port 8501)"
        echo "  dashboard          - Start the reporting dashboard (port 8502)"
        echo "  train [args]       - Train the ML model"
        echo "  shell              - Start interactive shell"
        echo "  help               - Show this help message"
        echo ""
        echo "Pipeline Usage:"
        echo "  docker run provider-verify pipeline --input data.csv --source EHR"
        echo ""
        echo "Training Usage:"
        echo "  docker run provider-verify train --data training_data.csv"
        ;;
    *)
        echo "Unknown command: $1"
        echo "Use 'help' for available commands"
        exit 1
        ;;
esac
