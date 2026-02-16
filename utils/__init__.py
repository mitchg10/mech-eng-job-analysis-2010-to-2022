"""
Utilities Package

This package provides utilities for analyzing mechanical engineering job postings
from Burning Glass Technologies data. It supports the three-phase analysis pipeline:

Phase 1: Data preparation and filtering
Phase 2: NLP classification of mechanical engineering jobs
Phase 3: Skills extraction and temporal trend analysis

Modules
-------
onet_utils
    O*NET code conversion and discipline assignment
classifier_utils
    Job classification with confidence scoring and adaptive thresholding
skills_utils
    Skills parsing from BGT clusters and prevalence calculations
statistics_utils
    Statistical modeling, diagnostics, and trajectory classification
visualization_utils
    Plotting functions for trajectories and diagnostics
publication_utils
    IEEE-format table generation and LaTeX export
"""

# Import key functions for convenient access
from .onet_utils import (
    convert_onet_numeric_to_standard,
    convert_onet_standard_to_numeric,
    load_discipline_codes,
    assign_discipline,
    create_discipline_lookup_expression
)

from .classifier_utils import (
    load_classifier_config,
    compile_patterns,
    calculate_confidence_score,
    classify_with_adaptive_threshold,
    classify_batch
)

from .skills_utils import (
    parse_skill_cluster,
    transform_to_long_format,
    calculate_skill_prevalence,
    filter_common_skills,
    get_top_skills
)

from .duckdb_manager import DuckDBManager

__all__ = [
    # O*NET utilities
    'convert_onet_numeric_to_standard',
    'convert_onet_standard_to_numeric',
    'load_discipline_codes',
    'assign_discipline',
    'create_discipline_lookup_expression',

    # Classifier utilities
    'load_classifier_config',
    'compile_patterns',
    'calculate_confidence_score',
    'classify_with_adaptive_threshold',
    'classify_batch',

    # Skills utilities
    'parse_skill_cluster',
    'transform_to_long_format',
    'calculate_skill_prevalence',
    'filter_common_skills',
    'get_top_skills',

    # DuckDB Manager
    'DuckDBManager'
]
