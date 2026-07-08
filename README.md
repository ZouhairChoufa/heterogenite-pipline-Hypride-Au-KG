# Heterogeneous Tabular Data Pipeline for Knowledge Graph Generation

## Overview

This repository contains the implementation of my Master's thesis:

> **Managing Heterogeneous Tabular Data for Knowledge Graph Generation (Big Data + Knowledge Graphs)**

The project proposes a modular and reproducible pipeline that automatically transforms heterogeneous tabular datasets into RDF Knowledge Graphs using Semantic Web technologies and Large Language Models (LLMs).

---

## Architecture

The pipeline consists of four main modules:

1. Multi-source data ingestion and profiling
2. Hybrid schema matching (Sentence-BERT + LLaMA 3.1)
3. Automatic YARRRML mapping generation and RDF construction using Morph-KGC
4. Entity Resolution with Splink and SHACL validation

---

## Technologies

- Python
- Sentence-BERT
- LLaMA 3.1 (Groq)
- LangChain
- YARRRML
- Morph-KGC
- RDF
- SHACL
- Splink
- Docker Compose

---

## Repository Structure

```
lib/                Source code
reports/            Generated reports
utils/              Utility functions
*.py                Pipeline modules
GUIDE_EXECUTION.md  Execution guide
```

---

## Research Objectives

- Integrate heterogeneous tabular datasets
- Automatically detect schema correspondences
- Generate RDF Knowledge Graphs
- Perform entity resolution
- Validate the generated Knowledge Graph

---

## Results

The proposed hybrid approach achieved:

- Schema Matching F1-score: **0.870**
- Entity Resolution F1-score: **0.990**
- RDF triples generated: **82,420**
- RDF generation time: **7 seconds**

---

## Author

**Zouhair Choufa**

Master's Thesis (PFE)

Faculty of Sciences and Techniques – Settat

2026
