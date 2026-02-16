"""
Classification Utilities for Mechanical Engineering Jobs

This module provides utilities for classifying job postings as mechanical
engineering positions using NLP techniques, confidence scoring, and adaptive
thresholding. Parameters are Bayesian-optimized for F1-score maximization.

The classification system uses multiple signals:
- Degree requirements (BS/MS in Mechanical Engineering)
- Job title patterns (core, design, manufacturing, etc.)
- Software mentions (CAD, ANSYS, MATLAB, etc.)
- Industry context (automotive, aerospace, HVAC, etc.)
- Mechanical concepts (thermodynamics, fluid mechanics, etc.)
"""

from typing import Dict, List, Optional, Tuple
import json
import re
from pathlib import Path
import polars as pl
import pandas as pd


def load_classifier_config(config_dir: Optional[Path] = None) -> Dict:
    """
    Load classifier configuration from JSON files.

    Loads both mechanical_terms.json and classifier_parameters.json
    and combines them into a single configuration dictionary.

    Parameters
    ----------
    config_dir : Path or None, optional
        Directory containing config files. If None, uses default path.

    Returns
    -------
    dict
        Combined configuration with keys:
        - terms: All term lists (core, design, software, etc.)
        - weights: Confidence scoring weights
        - thresholds: Adaptive classification thresholds
        - patterns: Compiled regex patterns for detection

    Examples
    --------
    >>> config = load_classifier_config()
    >>> config['terms']['core_terms']
    ['Mechanical Engineer', 'Mech Engineer', ...]
    """
    if config_dir is None:
        config_dir = Path(__file__).parent.parent / 'config'

    # Load terms
    with open(config_dir / 'mechanical_terms.json', 'r') as f:
        terms = json.load(f)

    # Load parameters
    with open(config_dir / 'classifier_parameters.json', 'r') as f:
        params = json.load(f)

    # Combine into single config
    config = {
        'terms': terms,
        'weights': params['confidence_weights'],
        'thresholds': params['adaptive_thresholds'],
        'chunk_size': params.get('chunk_size', 50000)
    }

    return config


def build_flexible_regex(terms: List[str]) -> re.Pattern:
    """
    Create regex pattern with flexible matching for word order and separators.

    Handles:
    - Plural forms (adds optional 's')
    - Multiple word orders for 2-word terms
    - Flexible separators (spaces, hyphens, underscores, slashes)

    Parameters
    ----------
    terms : list of str
        List of terms to match

    Returns
    -------
    re.Pattern
        Compiled regex pattern with case-insensitive, multiline flags

    Examples
    --------
    >>> pattern = build_flexible_regex(['Mechanical Engineer', 'CAD'])
    >>> bool(pattern.search('mechanical-engineer'))
    True
    >>> bool(pattern.search('engineer mechanical'))
    True
    >>> bool(pattern.search('CAD software'))
    True
    """
    patterns = []
    for term in terms:
        if ' ' in term:
            parts = term.split()
            if len(parts) == 2:
                # Both orders with flexible separators
                pattern1 = r'\b' + r'[\s\-_/]*'.join(parts) + r's?\b'
                pattern2 = r'\b' + r'[\s\-_/]*'.join(reversed(parts)) + r's?\b'
                patterns.extend([pattern1, pattern2])
            else:
                # Multi-word with flexible separators
                pattern = r'\b' + r'[\s\-_/]*'.join(parts) + r's?\b'
                patterns.append(pattern)
        else:
            # Single word with plural/abbreviation handling
            pattern = r'\b' + term + r's?\b'
            patterns.append(pattern)

    combined_pattern = '|'.join(f'({pattern})' for pattern in patterns)
    return re.compile(combined_pattern, re.IGNORECASE | re.MULTILINE)


def build_degree_pattern(degree_config: Dict) -> re.Pattern:
    """
    Build comprehensive degree requirement detection pattern.

    Parameters
    ----------
    degree_config : dict
        Degree patterns from config with keys: bachelors, masters, doctoral

    Returns
    -------
    re.Pattern
        Compiled pattern matching degree requirements in text

    Examples
    --------
    >>> config = load_classifier_config()
    >>> pattern = build_degree_pattern(config['terms']['degree_patterns'])
    >>> bool(pattern.search('BS in Mechanical Engineering required'))
    True
    """
    bachelors = degree_config['bachelors']
    masters = degree_config['masters']
    doctoral = degree_config['doctoral']

    patterns = [
        # Degree-specific patterns
        r'\b(?:BS|B\.S\.|Bachelor(?:s?)|MS|M\.S\.|Master(?:s?))\s*(?:degree\s*)?(?:in\s*)?(?:of\s*)?Mechanical\s*Engineering\b',
        r'\bMechanical\s*Engineering\s*(?:degree|BS|B\.S\.|MS|M\.S\.|Bachelor|Master)\b',
        r'\bBSME\b|\bMSME\b',
        r'\bB\.S\.M\.E\b|\bM\.S\.M\.E\b',
        r'\bMech\s*Eng\s*degree\b',
        r'\bDegree.*Mechanical\s*Engineering\b',
        r'\bMechanical\s*Engineering.*(?:required|preferred|desired)\b',

        # Bachelor's level patterns
        rf'\b(?:{"|".join(bachelors)})\s*(?:degree\s*)?(?:in\s*)?(?:of\s*)?Mechanical\s*Engineering\b',
        rf'\bMechanical\s*Engineering\s*(?:{"|".join(bachelors)})\b',

        # Master's level patterns
        rf'\b(?:{"|".join(masters)})\s*(?:degree\s*)?(?:in\s*)?(?:of\s*)?Mechanical\s*Engineering\b',
        rf'\bMechanical\s*Engineering\s*(?:{"|".join(masters)})\b',

        # Doctoral level patterns
        rf'\b(?:{"|".join(doctoral)})\s*(?:degree\s*)?(?:in\s*)?(?:of\s*)?Mechanical\s*Engineering\b',
        rf'\bMechanical\s*Engineering\s*(?:{"|".join(doctoral)})\b',

        # Additional abbreviations
        r'\bBSME\b|\bMSME\b|\bPhDME\b',
        r'\bB\.S\.M\.E\b|\bM\.S\.M\.E\b|\bPh\.D\.M\.E\b',
        r'\bMech\.?\s*Eng\.?\s*(?:degree|BS|MS|PhD|Bachelor|Master|Doctor)\b'
    ]

    combined = '|'.join(f'({pattern})' for pattern in patterns)
    return re.compile(combined, re.IGNORECASE)


def detect_degree_requirement(text: str, degree_pattern: re.Pattern) -> bool:
    """
    Detect explicit mechanical engineering degree requirements.

    Parameters
    ----------
    text : str
        Job text to search
    degree_pattern : re.Pattern
        Compiled degree detection pattern

    Returns
    -------
    bool
        True if degree requirement detected, False otherwise
    """
    if not text:
        return False
    return bool(degree_pattern.search(str(text)))


def count_software_mentions(text: str, software_pattern: re.Pattern) -> int:
    """
    Count mechanical engineering software mentions.

    Parameters
    ----------
    text : str
        Text to search
    software_pattern : re.Pattern
        Compiled software detection pattern

    Returns
    -------
    int
        Number of software mentions found
    """
    if not text:
        return 0
    matches = software_pattern.findall(str(text))
    return len(matches)


def count_industry_mentions(text: str, industry_pattern: re.Pattern) -> int:
    """
    Count mechanical engineering industry mentions.

    Parameters
    ----------
    text : str
        Text to search
    industry_pattern : re.Pattern
        Compiled industry detection pattern

    Returns
    -------
    int
        Number of industry mentions found
    """
    if not text:
        return 0
    matches = industry_pattern.findall(str(text))
    return len(matches)


def calculate_confidence_score(
    job_text: str,
    job_title: str,
    config: Dict,
    patterns: Dict[str, re.Pattern]
) -> float:
    """
    Calculate confidence score for mechanical engineering classification.

    Uses Bayesian-optimized weights to combine multiple signals:
    - Degree requirements (15.48)
    - Core title match (8.55)
    - Design/manufacturing/specialized context (2.35-5.88)
    - Software mentions (1.10-1.73 per mention, capped)
    - Industry mentions (1.43-1.97 per mention, capped)
    - Mechanical concepts (1.75 per concept, capped)
    - Negative indicators (-4.77 for other engineering, -1.42 for non-engineering)

    Parameters
    ----------
    job_text : str
        Full job description text
    job_title : str
        Job title
    config : dict
        Configuration with weights and terms
    patterns : dict
        Pre-compiled regex patterns

    Returns
    -------
    float
        Confidence score (higher = more likely mechanical engineering)

    Examples
    --------
    >>> config = load_classifier_config()
    >>> patterns = compile_patterns(config)
    >>> score = calculate_confidence_score(
    ...     "Design CAD models using SolidWorks...",
    ...     "Mechanical Design Engineer",
    ...     config,
    ...     patterns
    ... )
    >>> score > 10  # High confidence
    True
    """
    score = 0
    weights = config['weights']

    title_lower = str(job_title).lower() if job_title else ""
    text_lower = str(job_text).lower() if job_text else ""
    combined = f"{title_lower} {text_lower}"

    # VERY HIGH CONFIDENCE INDICATORS
    if detect_degree_requirement(combined, patterns['degree']):
        score += weights['degree_requirement']

    if patterns['core'].search(title_lower):
        score += weights['core_title']

    # HIGH CONFIDENCE INDICATORS
    if patterns['design'].search(combined):
        score += weights['design_context']

    if patterns['manufacturing'].search(combined):
        score += weights['manufacturing_context']

    if patterns['specialized'].search(combined):
        score += weights['specialized_context']

    if patterns['hierarchy'].search(combined):
        score += weights['hierarchy_context']

    # MEDIUM CONFIDENCE - Project roles
    if patterns['project'].search(combined):
        project_score = weights['project_base']

        industry_mentions = count_industry_mentions(combined, patterns['industry'])
        if industry_mentions > 0:
            project_score += min(
                industry_mentions * weights['industry_mention_multiplier'],
                weights['industry_mention_max']
            )

        software_mentions = count_software_mentions(combined, patterns['software'])
        if software_mentions > 0:
            project_score += min(
                software_mentions * weights['software_mention_multiplier'],
                weights['software_mention_max']
            )

        score += project_score

    # SUPPORTING INDICATORS
    software_count = count_software_mentions(combined, patterns['software'])
    if software_count > 0:
        score += min(
            software_count * weights['software_standalone_multiplier'],
            weights['software_standalone_max']
        )

    industry_count = count_industry_mentions(combined, patterns['industry'])
    if industry_count > 0:
        score += min(
            industry_count * weights['industry_standalone_multiplier'],
            weights['industry_standalone_max']
        )

    # Mechanical concepts
    concepts = config['terms']['mechanical_concepts']
    concept_count = sum(1 for concept in concepts if concept in combined)
    if concept_count > 0:
        score += min(
            concept_count * weights['mechanical_concept_multiplier'],
            weights['mechanical_concept_max']
        )

    # NEGATIVE INDICATORS
    exclusion_patterns = config['terms']['exclusion_patterns']
    for pattern_str in exclusion_patterns:
        if re.search(pattern_str, combined, re.IGNORECASE):
            score += weights['exclusion_penalty']

    non_engineering_patterns = config['terms']['non_engineering_patterns']
    for pattern_str in non_engineering_patterns:
        if re.search(pattern_str, combined, re.IGNORECASE):
            score += weights['non_engineering_penalty']

    return score


def classify_with_adaptive_threshold(
    confidence: float,
    job_title: str,
    thresholds: Dict
) -> str:
    """
    Classify job using adaptive thresholds based on job category.

    Different job categories have different thresholds to optimize
    for precision/recall tradeoffs. Lower thresholds for categories
    with historically high false negative rates.

    Parameters
    ----------
    confidence : float
        Confidence score from calculate_confidence_score()
    job_title : str
        Job title for category detection
    thresholds : dict
        Threshold configuration from classifier_parameters.json

    Returns
    -------
    str
        Classification: 'mechanical-major' or 'non-mechanical-major'

    Examples
    --------
    >>> thresholds = load_classifier_config()['thresholds']
    >>> classify_with_adaptive_threshold(8.0, "Mechanical Engineer", thresholds)
    'mechanical-major'
    >>> classify_with_adaptive_threshold(3.0, "Project Engineer", thresholds)
    'mechanical-major'  # Lower threshold for project engineers
    >>> classify_with_adaptive_threshold(2.0, "Software Engineer", thresholds)
    'non-mechanical-major'
    """
    title_lower = str(job_title).lower()

    # DEFINITIVE POSITIVE
    if confidence >= thresholds['definitive_positive']:
        return 'mechanical-major'

    # CATEGORY-SPECIFIC THRESHOLDS
    if 'project engineer' in title_lower:
        threshold = thresholds['project_engineer']
        if confidence >= threshold:
            return 'mechanical-major'

    elif re.search(r'\bengineer\b', title_lower) and not re.search(
        r'\b(?:software|electrical|civil|chemical|computer)\b', title_lower
    ):
        threshold = thresholds['generic_engineer']
        if confidence >= threshold:
            return 'mechanical-major'

    elif 'design' in title_lower:
        threshold = thresholds['design_role']
        if confidence >= threshold:
            return 'mechanical-major'

    elif any(term in title_lower for term in ['manufacturing', 'production', 'process']):
        threshold = thresholds['manufacturing_role']
        if confidence >= threshold:
            return 'mechanical-major'

    else:
        threshold = thresholds['standard']
        if confidence >= threshold:
            return 'mechanical-major'

    return 'non-mechanical-major'


def compile_patterns(config: Dict) -> Dict[str, re.Pattern]:
    """
    Compile all regex patterns from configuration.

    Parameters
    ----------
    config : dict
        Configuration from load_classifier_config()

    Returns
    -------
    dict
        Dictionary of compiled patterns with keys:
        core, design, manufacturing, project, specialized, hierarchy,
        software, industry, degree
    """
    terms = config['terms']

    patterns = {
        'core': build_flexible_regex(terms['core_terms']),
        'design': build_flexible_regex(terms['design_terms']),
        'manufacturing': build_flexible_regex(terms['manufacturing_terms']),
        'project': build_flexible_regex(terms['project_terms']),
        'specialized': build_flexible_regex(terms['specialized_terms']),
        'hierarchy': build_flexible_regex(terms['hierarchy_terms']),
        'software': re.compile(
            '|'.join(f'\\b{re.escape(sw)}\\b' for sw in terms['software']),
            re.IGNORECASE
        ),
        'industry': re.compile(
            '|'.join(f'\\b{re.escape(ind)}\\b' for ind in terms['industries']),
            re.IGNORECASE
        ),
        'degree': build_degree_pattern(terms['degree_patterns'])
    }

    return patterns


def classify_batch(
    df: pl.DataFrame,
    config: Optional[Dict] = None,
    text_col: str = 'JobText',
    title_col: str = 'CleanJobTitle',
    chunk_size: Optional[int] = None,
    verbose: bool = True
) -> pl.DataFrame:
    """
    Classify a batch of jobs with progress tracking.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with job data
    config : dict or None, optional
        Configuration from load_classifier_config(). If None, loads default.
    text_col : str, default 'JobText'
        Name of job text column
    title_col : str, default 'CleanJobTitle'
        Name of job title column
    chunk_size : int or None, optional
        Records per chunk. If None, uses config value.
    verbose : bool, default True
        Print progress updates

    Returns
    -------
    pl.DataFrame
        Input DataFrame with added columns:
        - NLP_Classification_Improvement_2: Classification result
        - Confidence_Score: Confidence score

    Examples
    --------
    >>> classified = classify_batch(study_data, verbose=True)
    Processing Mechanical engineering jobs...
        Processed 50,000 records...
        ...
    Classification complete for 982,254 records
    """
    if config is None:
        config = load_classifier_config()

    if chunk_size is None:
        chunk_size = config.get('chunk_size', 50000)

    # Compile patterns once
    patterns = compile_patterns(config)
    thresholds = config['thresholds']

    if verbose:
        print("Processing classification...")

    # Convert to pandas for row-wise processing
    pandas_df = df.to_pandas()

    results = []
    confidence_scores = []

    for i in range(0, len(pandas_df), chunk_size):
        chunk = pandas_df.iloc[i:i + chunk_size]
        chunk_results = []
        chunk_scores = []

        for _, row in chunk.iterrows():
            job_text = str(row.get(text_col, '')) if row.get(text_col) is not None else ''
            job_title = str(row.get(title_col, '')) if row.get(title_col) is not None else ''

            # Calculate confidence
            confidence = calculate_confidence_score(job_text, job_title, config, patterns)

            # Classify with adaptive thresholds
            classification = classify_with_adaptive_threshold(
                confidence, job_title, thresholds
            )

            chunk_results.append(classification)
            chunk_scores.append(confidence)

        results.extend(chunk_results)
        confidence_scores.extend(chunk_scores)

        if verbose and i + chunk_size < len(pandas_df):
            print(f"    Processed {i + chunk_size:,} records...")

    # Add results to DataFrame
    classified_df = df.with_columns([
        pl.Series("NLP_Classification_Improvement_2", results),
        pl.Series("Confidence_Score", confidence_scores, dtype=pl.Float64)
    ])

    if verbose:
        print(f"\nClassification complete for {len(classified_df):,} records")

        # Summary statistics
        summary = classified_df.group_by("NLP_Classification_Improvement_2").agg(
            pl.count().alias("count")
        ).sort("count", descending=True)

        print("\n=== CLASSIFICATION SUMMARY ===")
        for row in summary.iter_rows(named=True):
            classification = row['NLP_Classification_Improvement_2']
            count = row['count']
            percentage = (count / len(classified_df)) * 100
            print(f"{classification}: {count:,} ({percentage:.1f}%)")

    return classified_df


# Example usage
if __name__ == "__main__":
    print("Testing classifier_utils module")
    print("=" * 60)

    # Load config
    print("\n1. Loading configuration...")
    config = load_classifier_config()
    print(f"   Loaded {len(config['terms']['core_terms'])} core terms")
    print(f"   Loaded {len(config['terms']['software'])} software tools")
    print(f"   Loaded {len(config['weights'])} weight parameters")

    # Compile patterns
    print("\n2. Compiling patterns...")
    patterns = compile_patterns(config)
    print(f"   Compiled {len(patterns)} pattern types")

    # Test classification
    print("\n3. Testing classification...")
    test_cases = [
        ("Mechanical Engineer with SolidWorks experience", "Mechanical Engineer"),
        ("Design complex mechanical systems using CAD", "Design Engineer"),
        ("Software Engineer - Python developer", "Software Engineer"),
        ("Project Engineer in automotive manufacturing", "Project Engineer")
    ]

    for text, title in test_cases:
        score = calculate_confidence_score(text, title, config, patterns)
        classification = classify_with_adaptive_threshold(
            score, title, config['thresholds']
        )
        print(f"\n   Title: {title}")
        print(f"   Score: {score:.2f}")
        print(f"   Classification: {classification}")

    print("\n" + "=" * 60)
    print("Module tests complete!")
