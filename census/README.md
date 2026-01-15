Census Data Module

This module contains code and documentation for collecting, processing, and analyzing U.S. Census data for use in quantitative and geospatial analysis pipelines.

The primary goal of this module is to support **reproducible, configurable, and statistically sound** use of Census indicators in downstream analytical workflows.

---

## Overview

The Census data workflows implemented here are designed to:

- Programmatically retrieve demographic, economic, and housing indicators from U.S. Census sources
- Support flexible selection of geographic levels (e.g., county-level)
- Clean and standardize raw Census variables for analysis
- Enable integration with external datasets (e.g., geospatial or environmental data)
- Provide inputs suitable for statistical modeling and validation

All code emphasizes **modularity, transparency, and reproducibility**.

---

## Data Sources

- **U.S. Census Bureau**
  - American Community Survey (ACS)
  - Decennial Census
- Data access via publicly available APIs and published tables

No proprietary or restricted datasets are included in this repository.

---

## Example Variables

This module supports analysis of Census indicators such as:

- Population size and density
- Transportation and commuting patterns
- Household income distribution
- Housing stock characteristics
- Energy and heating source proxies
- Age, vulnerability, and demographic structure

Variables are referenced using official Census table and field identifiers.

---

## Folder Structure
census/
├── fetch_census_data.py     # Functions for pulling Census data via API
├── clean_census_data.py     # Data cleaning and standardization utilities
├── census_variables.py      # Centralized variable definitions and mappings
└── README.md
