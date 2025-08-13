# drift-datasets Documentation

Generate standardized concept drift datasets with ground truth metadata for machine learning research through simple TOML configuration.

[![Version](https://img.shields.io/pypi/v/drift-datasets?color=blue)](https://pypi.org/project/drift-datasets/)
[![Python](https://img.shields.io/pypi/pyversions/drift-datasets)](https://pypi.org/project/drift-datasets/)
[![License](https://img.shields.io/github/license/BorjaEst/drift-datasets)](../LICENSE)
[![Tests](https://img.shields.io/github/actions/workflow-status/BorjaEst/drift-datasets/tests.yml?branch=main&label=tests)](https://github.com/BorjaEst/drift-datasets/actions)

## Quick Navigation

### Getting Started

- [Installation Guide](installation.md) - System setup and requirements
- [Quickstart Tutorial](quickstart.md) - Generate your first dataset in 2 minutes
- [Configuration Guide](configuration.md) - Comprehensive TOML configuration reference

### Core Features

- [Drift Patterns](drift-patterns.md) - Complete guide to drift types and patterns
- [Configuration Guide](configuration.md) - Comprehensive TOML configuration reference

### Research Applications

- [ExpertSystems Compatibility](expertsystems.md) - Reproduce research paper configurations

### Advanced Topics

- [API Reference](api-reference.md) - Complete method and class documentation
- [Requirements Specification](requirements.md) - Detailed functional requirements

### Development

- [Troubleshooting](troubleshooting.md) - Common issues and solutions

## Overview

**drift-datasets** is a Python library for generating standardized concept drift datasets with comprehensive ground truth metadata. It serves machine learning researchers, data scientists, and algorithm developers who need reproducible datasets for evaluating drift detection methods, adaptive learning algorithms, and streaming data analysis.

### Key Problems Solved

1. **Inconsistent drift dataset generation** across research studies
2. **Lack of ground truth metadata** for drift detection evaluation
3. **Complex parameter translation** between research concepts and implementation
4. **Reproducibility challenges** in comparative studies

### Core Value Propositions

- **Unified dataset objects** containing feature matrices, targets, and detailed drift annotations
- **Configuration-driven reproducible generation** through TOML files
- **Support for synthetic, real-world, and mixed datasets** with optional drift injection
- **Research-friendly parameter translation** for common drift simulation frameworks

## Target Audiences

### ML Researchers

- Evaluating drift detection algorithms
- Conducting comparative studies
- Reproducing published research results

### Data Scientists

- Prototyping adaptive learning systems
- Validating streaming data approaches
- Testing concept drift robustness

### Algorithm Developers

- Benchmarking streaming methods
- Developing drift-aware models
- Creating drift detection tools

### Educators

- Teaching concept drift concepts
- Demonstrating streaming learning
- Providing standardized datasets for coursework

## Library Philosophy

### Research-First Design

- Ground truth metadata is a first-class citizen
- Research parameter names take precedence over implementation details
- Reproducibility is built into the core architecture

### Configuration-Driven

- TOML files enable version-controlled experimental setups
- Declarative configuration reduces programming errors
- Shareable configurations improve research collaboration

### Standards-Based

- Compatible with established research frameworks (CapyMOA, UCI)
- Follows pandas/sklearn conventions for familiar APIs
- Supports common data science workflows

## What's Next?

- **New users**: Start with the [Installation Guide](installation.md)
- **Quick demo**: Follow the [Quickstart Tutorial](quickstart.md)
- **Research focus**: Jump to [ExpertSystems Compatibility](expertsystems.md)
- **API exploration**: Browse the [API Reference](api-reference.md)

## Support and Community

- **GitHub Issues**: [Report bugs and request features](https://github.com/BorjaEst/drift-datasets/issues)
- **GitHub Discussions**: [Ask questions and share ideas](https://github.com/BorjaEst/drift-datasets/discussions)
- **Security Contact**: [boressan@outlook.com](mailto:boressan@outlook.com)

---

Documentation last updated: August 13, 2025
