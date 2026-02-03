# ProviderVerify - Healthcare Provider Entity Resolution Engine

A production-grade Python implementation of a healthcare provider entity-resolution system that can ingest data from multiple sources, normalize fields, apply blocking to reduce candidate pairs, compute hybrid similarity scores, auto-merge high-confidence pairs, and route borderline pairs to human audit.

## 🎯 Features

- **Multi-Source Data Ingestion**: Support for CSV, Parquet, and JSON formats from S3-compatible storage
- **Advanced Normalization**: Name, affiliation, address, phone, and email standardization using spaCy, libpostal, and fuzzy matching
- **Intelligent Blocking**: Multi-layer deterministic blocking reducing billions of comparisons to millions
- **Hybrid Scoring**: Combines deterministic rules with XGBoost machine learning for accurate matching
- **Auto-Merge & Audit**: Automatic merging of high-confidence matches with human review for borderline cases
- **Feedback Loop**: Continuous model improvement through audit feedback
- **Performance Monitoring**: Real-time dashboards and comprehensive reporting
- **Production Ready**: Docker containerization, CI/CD pipeline, and comprehensive testing

## 🏗️ Architecture

```
ProviderVerify Pipeline
├── Data Ingestion (S3, validation)
├── Normalization (names, addresses, contacts)
├── Blocking (multi-layer keys)
├── Scoring (deterministic + ML)
├── Merging (canonical entities)
├── Audit (human review interface)
└── Reporting (performance metrics)
```

## 📋 Requirements

- Python 3.9+
- Docker (optional, for containerized deployment)
- 8GB+ RAM recommended for large datasets
- S3-compatible storage (optional)

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-org/provider-verify.git
cd provider-verify

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### 2. Configuration

Copy and customize the configuration:

```bash
cp config/provider_verify.yaml config/my_config.yaml
# Edit config/my_config.yaml with your settings
```

### 3. Run the Pipeline

```bash
# Basic usage
python -m src.pipeline.run_provider_verify \
    --input data/providers.csv \
    --source "EHR" \
    --config config/my_config.yaml \
    --output results/

# With Spark for large datasets
python -m src.pipeline.run_provider_verify \
    --input s3://my-bucket/providers/ \
    --source "Multiple" \
    --spark \
    --output results/
```

### 4. Launch Audit Interface

```bash
# Start audit UI (port 8501)
streamlit run src/audit/ui.py

# Start reporting dashboard (port 8502)
streamlit run src/reporting/dashboard.py
```

## 🐳 Docker Deployment

### Build and Run

```bash
# Build Docker image
docker build -t provider-verify .

# Run pipeline
docker run -v $(pwd)/data:/app/data provider-verify pipeline \
    --input /app/data/providers.csv --source EHR

# Start audit UI
docker run -p 8501:8501 provider-verify audit

# Start dashboard
docker run -p 8502:8502 provider-verify dashboard
```

### Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## 📊 Performance

Target performance metrics for production deployment:

- **Precision**: ≥92%
- **Recall**: ≥88%
- **Scalability**: 184K+ providers from 5+ sources
- **Reduction**: 34B → 1.63M candidate pairs (95% reduction)
- **Processing**: <5 minutes for 100K records

## 🔧 Configuration

Key configuration sections in `config/provider_verify.yaml`:

```yaml
# Data sources
data_sources:
  bucket_name: "provider-verify-data"
  supported_formats: ["csv", "parquet", "json"]

# Scoring weights
scoring:
  weights:
    name_exact: 0.30
    name_fuzzy: 0.20
    affiliation: 0.15
    location: 0.25
    contact: 0.10
  thresholds:
    auto_merge: 0.85
    audit_low: 0.65

# Blocking strategies
blocking:
  strategies:
    - name: "location_name"
      keys: ["state_norm", "zip_norm[:3]", "last_name_metaphone"]
```

## 🧪 Testing

Run the test suite:

```bash
# Unit tests
pytest tests/ -v

# Integration tests
pytest tests/test_integration.py -v

# Coverage report
pytest --cov=src --cov-report=html tests/
```

## 📈 Monitoring

### Performance Dashboard

Access the reporting dashboard at `http://localhost:8502`:

- Real-time pipeline metrics
- Model performance trends
- Audit completion rates
- System health monitoring

### Key Metrics

- **Pipeline Completion Rate**: % of records processed successfully
- **Merge Rate**: % of candidate pairs auto-merged
- **Audit Queue Length**: Number of pending human reviews
- **Model Performance**: Precision, recall, F1-score over time

## 🔒 Security & Compliance

- **HIPAA Compliance**: Built-in PHI protection and audit logging
- **Encryption**: Support for encryption at rest and in transit
- **Access Control**: Role-based permissions and audit trails
- **Data Masking**: Automatic PHI de-identification in logs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8 isort

# Run pre-commit checks
black src/
isort src/
flake8 src/
pytest tests/
```

## 📚 API Reference

### Pipeline Interface

```python
from src.pipeline.run_provider_verify import ProviderVerifyPipeline

# Initialize pipeline
pipeline = ProviderVerifyPipeline("config/provider_verify.yaml")

# Run pipeline
report = pipeline.run_pipeline(
    input_path="data/providers.csv",
    source_label="EHR",
    output_path="results/"
)
```

### Individual Components

```python
# Normalization
from src.normalize.name_normalizer import normalize_provider_names
normalized_df = normalize_provider_names(df, config)

# Blocking
from src.blocking.block_key_builder import create_blocking_keys
providers_df, candidates_df = create_blocking_keys(df, config)

# Scoring
from src.match.scorer import apply_hybrid_scoring
hybrid_df = apply_hybrid_scoring(scored_pairs, candidates, providers)
```

## 🐛 Troubleshooting

### Common Issues

**Memory Issues with Large Datasets**
```bash
# Use Spark processing
python -m src.pipeline.run_provider_verify --spark --input large_dataset.csv
```

**spaCy Model Not Found**
```bash
python -m spacy download en_core_web_sm
```

**S3 Connection Issues**
```bash
# Set AWS credentials
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
```

### Debug Mode

Enable debug logging:

```bash
python -m src.pipeline.run_provider_verify \
    --log-level DEBUG \
    --input data.csv --source EHR
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

- **Documentation**: [Full Documentation](docs/)
- **Issues**: [GitHub Issues](https://github.com/your-org/provider-verify/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/provider-verify/discussions)

## 🗺️ Roadmap

- [ ] Support for additional data sources (FHIR, HL7)
- [ ] Advanced ML models (transformers, graph neural networks)
- [ ] Real-time processing capabilities
- [ ] Multi-tenant architecture
- [ ] Advanced analytics and insights
- [ ] Cloud-native deployment options

---

**ProviderVerify** - Empowering healthcare data integration with confidence and compliance.
