# Data Directory

Store all data files for the Multimodal Anti Money Laundering project here.

## Structure

- **`raw/`** — Original, immutable data as received. Never modify files here.
- **`processed/`** — Cleaned, transformed, and feature-engineered data ready for modeling.

## Best Practices

- Use **DVC** to version large data files instead of Git
- Track `.dvc` files in Git; store actual data remotely
- **Never commit** large data files directly to Git
- Document data sources and transformations in notebooks or scripts
