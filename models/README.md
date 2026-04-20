# Models Directory

Store trained models, serialized artifacts, and predictions here.

## What Goes Here

- Trained/serialized models (`.pkl`, `.joblib`, `.h5`, `.pth`, etc.)
- Model predictions and outputs
- Model metadata, summaries, and evaluation reports

## Best Practices

- **Never commit** large model files to Git
- Use DVC or cloud storage (S3, GCS) to version and manage models
- Document model architecture, hyperparameters, and performance metrics
- Include training dates and dataset versions in model naming
