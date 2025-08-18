# HistoPLUS

## Towards Comprehensive Cellular Characterisation of H&E slides

[![License](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/deed.en)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-orange.svg)](https://pytorch.org/)
[![Documentation](https://img.shields.io/badge/docs-github.com/owkin/histoplus-green.svg)](https://github.com/owkin/histoplus)
---

Corresponding pre-print paper can be found [here](https://arxiv.org/abs/2508.09926).


## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack--model-details)
- [Installation](#installation)
- [Quick Start](#quick-start--example-usage)
- [Evaluation & Metrics](#evaluation--metrics)
- [Model Card](#model-card--responsible-use)
- [Repository Structure](#repository-structure)
- [Documentation & Tutorials](#documentation--tutorials)
- [Citing This Work](#citing-this-work)
- [Contributing](#contributing)
- [License & Authorship](#license--authorship)
- [Acknowledgements](#acknowledgements)

## Overview

HistoPLUS addresses critical challenges in analyzing tumor microenvironments (TME) on hematoxylin and eosin (H&E) stained histopathology slides. Existing methods suffer from poor performance on understudied cell types and limited cross-domain generalization, hindering comprehensive TME analysis.

**Why it matters:**
- Cell detection, segmentation, and classification are fundamental for understanding tumor biology
- Current methods fail on rare cell types not well-represented in public datasets
- Cross-institutional and cross-indication performance remains limited

**Our approach:**
HistoPLUS introduces a CellViT architecture trained on a carefully curated pan-cancer dataset of 108,722 nuclei spanning 13 distinct cell types. The model achieves state-of-the-art performance while using significantly fewer parameters, enabling robust analysis of both common and understudied cell populations.


## Key Features

- 🎯 **Unprecedented performances**: Demonstrates 5.2% improvement in detection quality and 23.7% improvement in overall F1 classification score
- 📊 **13 cell types**: Improves cell type coverage by enabling the study of 7 previously understudied populations
- 🧠 **State-of-the-art Foundation Models**: Integrates the latest specialized Pathology Foundation Models
- 🌐 **Cross-domain robustness**: Validated across 6 independent cohorts, including 2 unseen oncology indications
- ⚡ **Efficient inference**: 5x fewer parameters than competing methods
- 🔧 **Easy deployment**: Simple CLI and Python API
- 📁 **Multiple formats**: Supports various whole slide image formats
- 🔬 **Adaptable to multiple magnification**: Supports 20x and 40x magnification

## Technology Stack & Model Details

**Core Dependencies:**
- **Python**: ≥3.10
- **PyTorch**: ≥2.4.1 with GPU support
- **OpenSlide**: ≥1.3.1 for WSI processing
- **TIMM**: 1.0.8 for transformer backbones
- **XFormers**: ≥0.0.29 for efficient attention

**Input/Output Formats:**
- **Input**: H&E whole slide images (.svs, .tiff, .ndpi, etc.)
- **Output**: JSON annotations with cell coordinates, types, and confidence scores

**Hardware Requirements:**
- **GPU**: NVIDIA GPU with ≥8GB VRAM recommended
- **RAM**: ≥16GB system memory
- **Storage**: Variable based on slide size (typically 100MB-2GB per slide)

## Installation

### Prerequisites

```bash
# Ensure Python 3.10+ is installed
python --version

# Install OpenSlide system dependencies (Ubuntu/Debian)
sudo apt-get install openslide-tools

# For macOS with Homebrew
brew install openslide
```

### Install from PyPI (Recommended)

```bash
pip install histoplus
```

### Install from Source (Alternative)

```bash
# Clone the repository
git clone https://github.com/owkin/histoplus.git
cd histoplus

# Install in development mode
pip install -e .

# Optional: Install development dependencies
pip install -e ".[dev,testing,linting]"
```

### Verify Installation

```bash
histoplus --help
```

## Quick Start / Example Usage

### Command Line Interface

```bash
# Basic nuclei detection and classification
histoplus extract \
  --slides /path/to/slide.svs \
  --export_dir /path/to/results/

# With custom parameters
histoplus extract \
  --slides /path/to/slide.svs \
  --export_dir /path/to/results/ \
  --batch-size 16 \
  --n_workers 8
```

### Python API

```python
from histoplus.extract import extract
from histoplus.helpers.segmentor import CellViTSegmentor

# Instantiate segmentor
segmentor = CellViTSegmentor.default()

# Process a whole slide image
results = extract(
    slide=slide,
    segmentor=segmentor,
    tile_size=224,
    n_workers=8,
    batch_size=16,
    verbose=1,
)

# Save results
results.save("output/results.json")
```

### Expected Output

```json
{
  "model_name": "histoplus_v1",
  "inference_mpp": 0.25,
  "cell_masks": [
    {
      "x": 1,
      "y": 1,
      "level": 16,
      "width": 224,
      "height": 224,
      "masks": [
        {
            "cell_id": 1,
            "cell_type": "Plasmocyte",
            "confidence": 0.94,
            "coordinates": [[1000, 2024], [1048, 2024], ...],
            "centroid": [1024, 2048]
        }
      ]
    }
  ]
}
```

## Evaluation & Metrics

### Performance in External Validation
![Comparison Figure](docs/scatter_plot_fig.png)
*Comparison of CellViT models with different backbones based on detection quality, classification performance
and model size confirm the superiority of pathology-specific encoders on external set.  The CellViT model with the H0-mini encoder emerges as the best trade-off, combining strong
detection and classification performance with a compact architecture.*

### Per-Cell-Type Performance
![Comparison Figure](docs/comparison_metrics.png)
*HistoTRAIN dataset enables the analysis of 13 cell types and HistoPLUS proposes a SOTA model for detecting and classifying them.*

## Model Card / Responsible Use

### Primary Use Cases

- Research applications in tumor microenvironment analysis
- Cell population quantification in H&E stained slides
- Biomarker discovery and validation studies

### Intended Users
- Computational pathologists and researchers
- Bioinformatics professionals
- Clinical researchers (with appropriate validation)

### Limitations

- **H&E specific**: Trained exclusively on hematoxylin and eosin stained slides
- **Human tissue**: Validated on human tissue samples only
- **Research tool**: Not validated for clinical diagnosis

### Bias and Fairness

- **Dataset composition**: Training data includes samples from diverse institutions and demographics
- **Cell type representation**: Some rare cell types have limited representation
- **Technical bias**: Performance may vary with different scanners and staining protocols

### Ethical Considerations

- **Human oversight required**: Results should be validated by domain experts
- **Privacy**: Ensure compliance with data protection regulations
- **Transparency**: Model predictions include confidence scores for interpretation

## Repository Structure

```
histoplus/
├── histoplus/                     # Main package
│   ├── cli/                       # Command-line interface
│   │   ├── app.py                 # Main CLI application
│   │   ├── extract.py             # Extraction commands
│   │   └── utils.py               # CLI utilities
│   ├── extract/                   # Core extraction functionality
│   │   ├── core.py                # Main extraction logic
│   │   ├── segmentation/          # Segmentation modules
│   │   │   ├── core.py            # Segmentation algorithms
│   │   │   ├── predict/           # Inference pipeline
│   │   │   ├── dataloader.py      # Data loading utilities
│   │   │   └── postprocess.py     # Post-processing
│   │   └── utils.py               # Extraction utilities
│   └── helpers/                   # Utility modules
│       ├── nn/                    # Neural network components
│       │   ├── maskdino/          # MaskDINO implementation
│       │   ├── extractor/         # Feature extractors
│       │   └── vit.py             # Vision transformer components
│       ├── segmentor/             # Segmentation models
│       ├── tiling/                # Slide tiling utilities
│       ├── border_effects/        # Edge case handling
│       ├── data/                  # Data structures
│       └── serializers/           # Output formatting
├── tests/                         # Test suite
├── docs/                          # Documentation
├── pyproject.toml                 # Project configuration
├── README.md                      # This file
└── LICENSE                        # License information
```

## Citing This Work

If you use HistoPLUS in your research, please cite our work:

```bibtex
@misc{histoplus2025,
  title        = {Towards Comprehensive Cellular Characterisation of H\&E Slides},
  author       = {B. Adjadj, P.-A. Bannier, G. Horent, S. Mandela, A. Lyon, K. Schutte, U. Marteau, V. Gaury, L. Dumont, T. Mathieu, R. Belbahri, B. Schmauch, E. Durand, K. Von Loga, L. Gillet},
  year         = {2025},
  eprint       = {2508.09926},
  archivePrefix= {arXiv},
  primaryClass = {cs.CV},
  doi          = {10.48550/arXiv.2508.09926},
  url          = {https://arxiv.org/abs/2508.09926}
}
```

## Contributing

We welcome contributions to HistoPLUS! Please see our contributing guidelines for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/owkin/histoplus.git
cd histoplus

# Install development dependencies
pip install -e ".[dev,testing,linting]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Run linting
ruff check histoplus/
mypy histoplus/
```

### Guidelines

- **Code Style**: We use Ruff for linting and formatting
- **Testing**: Maintain >90% test coverage
- **Documentation**: Update docs for all new features
- **Type Hints**: All code should be fully typed

### Reporting Issues

- **Bug Reports**: Use the GitHub issue tracker
- **Feature Requests**: Discuss in GitHub Discussions first
- **Security Issues**: Email security@owkin.com

## License & Authorship

### License

This project is licensed under the CC BY-NC-ND 4.0 License. See the [LICENSE](https://github.com/owkin/histoplus/docs/LICENSE.md) file for details.

---
<br>

**🔬 Advancing computational pathology through robust, accessible AI tools.**  
For questions, support, or collaboration opportunities, please reach out via [GitHub Issues](https://github.com/owkin/histoplus/issues).

## Acknowledgements

The present study was funded by Owkin.

This study makes use of data generated by the
MOSAIC consortium (Owkin; Charité – Universitätsmedizin Berlin (DE); Lausanne University
Hospital - CHUV (CH); Universitätsklinikum Erlangen (DE); Institut Gustave Roussy (FR);
University of Pittsburgh (USA)).

This authors thank Dr Kathrina Alexander, Dr Audrey Caudron, Dr Richard Doughty, 
Dr Romain Dubois, Dr Thibaut Gioanni, Dr Camelia Radulescu, Dr Thomas Rialland, 
Dr Pierre Romero and Dr Yannis Roxanis for their contributions to HistoTRAIN and HistoVAL.

