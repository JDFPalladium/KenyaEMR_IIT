[![Run Makefile Targets](https://github.com/JDFPalladium/KenyaEMR_IIT/actions/workflows/main.yml/badge.svg)](https://github.com/JDFPalladium/KenyaEMR_IIT/actions/workflows/main.yml)

# KenyaEMR IIT Prediction Models

Clinical decision support tools integrated into KenyaEMR (Kenya's Electronic Medical Record system) to predict patient treatment interruption and return to care patterns for HIV/AIDS care continuity.

## Overview

This repository contains machine learning models that provide real-time predictions for HIV/AIDS patient care continuity:

### IIT Model (Interruption in Treatment)
Predicts whether a patient will be **30+ days late** to their next scheduled appointment, enabling proactive outreach and intervention.

### RTC Model (Return to Care)
For patients already experiencing treatment interruption, predicts the likelihood of **returning to care within 90 days**.

## Architecture

- **API Framework**: FastAPI for high-performance inference endpoints
- **ML Models**: XGBoost classifiers with feature engineering pipelines
- **Deployment**: Docker containers deployed on-premise at clinics
- **Data Storage**: AWS S3 for training data and model artifacts
- **CI/CD**: GitHub Actions for automated testing and deployment

## Project Structure

```
├── src/
│   ├── common/          # Shared data processing utilities
│   │   ├── clean_data.py
│   │   ├── create_target.py
│   │   ├── visit_features.py
│   │   └── target_features.py
│   ├── inference/       # Real-time prediction API
│   │   ├── api.py
│   │   └── generate_inference.py
│   └── training/        # Model retraining pipelines
│       └── refresh_model.py
├── pipelines/
│   ├── inference_pipeline.py
│   ├── rtc_inference_pipeline.py
│   └── retrain_pipeline.py
├── data/               # Model artifacts and encoders
├── tests/              # Unit tests
└── Dockerfile          # Production container
```

## Installation

### Local Development

```bash
# Clone repository
git clone https://github.com/JDFPalladium/KenyaEMR_IIT.git
cd KenyaEMR_IIT

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
make install

# Run tests
make test

# Run linting
make lint
```

## Usage

### Running the API Server

#### Using Docker (Recommended for Production)

```bash
# Build container
docker build -t kenyaemr-inference .

# Run container
docker run -p 8000:8000 kenyaemr-inference
```

#### Local Development

```bash
uvicorn src.inference.api:app --host 0.0.0.0 --port 8000
```

### API Endpoints

#### IIT Prediction
Predict if a patient will miss their appointment by 30+ days:

```bash
curl -X POST "http://localhost:8000/inference" \
  -H "Content-Type: application/json" \
  -d '{
    "ppk": "7E14A8034F39478149EE6A4CA37A247C631D17907C746BE0336D3D7CEC68F66F",
    "sc": "13074",
    "start_date": "2021-01-01",
    "end_date": "2025-01-01"
  }'
```

#### RTC Prediction
Predict if an IIT patient will return to care within 90 days:

```bash
curl -X POST "http://localhost:8000/rtc_inference" \
  -H "Content-Type: application/json" \
  -d '{
    "ppk": "7E14A8034F39478149EE6A4CA37A247C631D17907C746BE0336D3D7CEC68F66F",
    "sc": "13074",
    "start_date": "2021-01-01",
    "end_date": "2025-01-01"
  }'
```

**Parameters:**
- `ppk`: Patient primary key (hashed identifier)
- `sc`: Site code (clinic identifier)
- `start_date`: Data collection start date
- `end_date`: Data collection end date

## Model Retraining

Retrain models with updated data:

```bash
make retrain-and-build
```

This command:
1. Runs the retraining pipeline (`pipelines/retrain_pipeline.py`)
2. Saves updated model artifacts to `data/` and `models/`
3. Rebuilds the Docker inference container

## Development Workflow

### Available Make Commands

```bash
make install    # Install dependencies
make lint       # Run pylint checks
make test       # Run pytest with coverage
make format     # Format code with black
make retrain-and-build  # Retrain models and rebuild Docker image
```

### Running Tests

```bash
# Run all tests with coverage
make test

# Run specific test file
pytest tests/test_cleandata.py -v
```

### Code Quality

The project uses:
- **pylint** for linting (configured to disable R,C categories)
- **black** for code formatting
- **pytest** with coverage reporting

## Docker Management

```bash
# Build image
docker build -t kenyaemr-inference .

# Run container
docker run -p 8000:8000 kenyaemr-inference

# Clean up resources
docker container prune
docker image prune
docker builder prune

# Remove specific image
docker rmi <IMAGE_ID>
```

## CI/CD Pipeline

GitHub Actions automatically runs on every push:
1. Install dependencies
2. Format code with black
3. Lint with pylint
4. Run test suite with coverage

See [.github/workflows/main.yml](.github/workflows/main.yml) for details.

## Dependencies

### Production
- `fastapi` - API framework
- `uvicorn` - ASGI server
- `xgboost` - ML model framework
- `pandas`, `numpy` - Data processing
- `scikit-learn` - Preprocessing utilities
- `treelite_runtime` - Optimized model serving

### Development
- `pytest`, `pytest-cov` - Testing
- `pylint` - Linting
- `black` - Code formatting

See [requirements.txt](requirements.txt) for full dependencies.

## Data Privacy

All patient identifiers (ppk) are cryptographically hashed to protect privacy. No personally identifiable information is stored or transmitted through these systems.

## License

See [LICENSE](LICENSE) for details.

## Contributing

1. Create a feature branch
2. Make changes with tests
3. Ensure all tests pass: `make test`
4. Ensure linting passes: `make lint`
5. Format code: `make format`
6. Submit pull request

## Support

For issues or questions related to:
- **Model integration**: Contact the KenyaEMR development team
- **Technical issues**: Open an issue on GitHub
- **Clinical questions**: Consult with facility medical staff
