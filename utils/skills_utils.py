"""
Skills Parsing and Analysis Utilities

This module provides utilities for parsing and analyzing skill data from
Burning Glass Technologies (BGT) job postings. It handles the transformation
of skill cluster strings into structured data and calculates skill prevalence
metrics.

BGT Skill Cluster Format:
"Skill Family: Skill; Skill Type | Skill Family: Skill; Skill Type | ..."

Example:
"Engineering: Mechanical Engineering;Specialized Skills|Software: CAD;Technical Skills"
"""

from typing import List, Dict, Tuple, Optional
import polars as pl
import pandas as pd
from pathlib import Path


def parse_skill_cluster(skill_cluster_str: str) -> Tuple[List[Tuple[str, str, str]], Dict]:
    """
    Parse BGT skill cluster string with flexible handling of malformed data.

    Expected format: "Skill Family: Skill; Skill Type|Skill Family: Skill; Skill Type|..."

    Handles common malformations:
    - "Skill Type" only (no family/skill) - skipped, counted as type_only
    - "Skill Family: Skill" (no type) - uses default "Specialized Skills"
    - "Skill Family" only (no skill/type) - creates "Unknown Skill" entry
    - Empty entries - skipped, counted separately

    Parameters
    ----------
    skill_cluster_str : str
        Raw skill cluster string from BGT data

    Returns
    -------
    tuple
        - List of (family, skill, type) tuples for successfully parsed skills
        - Dictionary of parsing statistics with keys:
          * total_entries: Total skill entries in string
          * complete_entries: Entries with all three components
          * family_skill_only: Entries missing skill type
          * family_only: Entries with only family name
          * type_only: Entries with only skill type (not extracted)
          * empty_entries: Empty or whitespace-only entries
          * unparseable: Entries that couldn't be parsed
          * empty_input: 1 if input was None/empty, else not present

    Examples
    --------
    >>> parse_skill_cluster("Engineering: CAD;Technical Skills")
    ([('Engineering', 'CAD', 'Technical Skills')], {'total_entries': 1, ...})

    >>> parse_skill_cluster("Engineering: CAD;Technical|Specialized Skills")
    ([('Engineering', 'CAD', 'Technical')], {'total_entries': 2, ...})

    >>> # Malformed - missing type
    >>> parse_skill_cluster("Engineering: CAD")
    ([('Engineering', 'CAD', 'Specialized Skills')], {'total_entries': 1, ...})

    Notes
    -----
    The function is optimized for robustness over strict parsing. It attempts
    to extract as much useful information as possible from malformed data.
    Approximately 72.5% of entries are complete, 27.5% are type-only.
    """
    if not skill_cluster_str or skill_cluster_str == "" or pd.isna(skill_cluster_str):
        return [], {'empty_input': 1}

    skills = []
    stats = {
        'total_entries': 0,
        'complete_entries': 0,      # Family: Skill; Type
        'family_skill_only': 0,     # Family: Skill (no type)
        'family_only': 0,           # Family only
        'type_only': 0,             # Type only (like "Specialized Skills")
        'empty_entries': 0,         # Empty or whitespace only
        'unparseable': 0            # Couldn't parse at all
    }

    # Split by | to get individual skill entries
    skill_entries = str(skill_cluster_str).strip().split('|')
    stats['total_entries'] = len(skill_entries)

    for entry in skill_entries:
        entry = entry.strip()

        if not entry:
            stats['empty_entries'] += 1
            continue

        # Case 1: Complete format "Skill Family: Skill; Skill Type"
        if ':' in entry and ';' in entry:
            try:
                # Split by semicolon to separate main content from type
                parts = entry.split(';', 1)
                main_part = parts[0].strip()
                skill_type = parts[1].strip() if len(parts) > 1 else "Specialized Skills"

                # Split main part by colon to get family and skill
                if ':' in main_part:
                    family_skill = main_part.split(':', 1)
                    family = family_skill[0].strip()
                    skill = family_skill[1].strip()

                    # Only include if we have meaningful content
                    if family and skill:
                        skills.append((family, skill, skill_type))
                        stats['complete_entries'] += 1
                        continue
            except Exception:
                pass

        # Case 2: Family and skill only "Skill Family: Skill" (no type)
        elif ':' in entry:
            try:
                family_skill = entry.split(':', 1)
                family = family_skill[0].strip()
                skill = family_skill[1].strip()
                if family and skill:
                    skills.append((family, skill, "Specialized Skills"))  # Default type
                    stats['family_skill_only'] += 1
                    continue
            except Exception:
                pass

        # Case 3: Check if it's likely a skill type only (common patterns)
        skill_type_patterns = [
            'specialized skills', 'common skills', 'software and programming',
            'basic skills', 'core skills', 'technical skills'
        ]

        entry_lower = entry.lower()
        if any(pattern in entry_lower for pattern in skill_type_patterns):
            # This looks like a skill type only - we can't extract meaningful family/skill
            # so we'll skip it but count it
            stats['type_only'] += 1
            continue

        # Case 4: Might be a skill family only - less common but possible
        # If it doesn't match common skill type patterns, treat as family only
        if len(entry.split()) <= 3 and entry.replace(' ', '').isalpha():
            # Looks like it could be a family name
            skills.append((entry, "Unknown Skill", "Specialized Skills"))
            stats['family_only'] += 1
            continue

        # Case 5: Couldn't parse - count it but skip
        stats['unparseable'] += 1

    return skills, stats


def transform_to_long_format(
    df: pl.DataFrame,
    id_col: str = 'JobID',
    skill_col: str = 'CanonSkillClusters',
    keep_cols: Optional[List[str]] = None,
    chunk_size: int = 10000,
    verbose: bool = True
) -> pl.DataFrame:
    """
    Transform wide-format job data to long-format skill-job pairs.

    Takes a DataFrame where each row represents one job with a skill cluster
    string, and returns a DataFrame where each row represents one skill
    associated with one job.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with wide format (1 row per job)
    id_col : str, default 'JobID'
        Name of the unique job identifier column
    skill_col : str, default 'CanonSkillClusters'
        Name of column containing skill cluster strings
    keep_cols : list of str or None, optional
        Additional columns to keep from input DataFrame.
        If None, keeps: 'Discipline', 'JobDate'
    chunk_size : int, default 10000
        Number of records to process per chunk (for memory efficiency)
    verbose : bool, default True
        If True, prints progress updates and parsing statistics

    Returns
    -------
    pl.DataFrame
        Long-format DataFrame with columns:
        - JobID (or specified id_col)
        - SkillFamily
        - Skill
        - SkillType
        - Plus any columns specified in keep_cols

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({
    ...     'JobID': [1, 2],
    ...     'Discipline': ['Mechanical', 'Mechanical'],
    ...     'JobDate': ['2020-01-01', '2020-01-02'],
    ...     'CanonSkillClusters': [
    ...         'Engineering: CAD;Technical',
    ...         'Software: Python;Programming'
    ...     ]
    ... })
    >>> skills_df = transform_to_long_format(df)
    >>> len(skills_df)
    2

    Notes
    -----
    Processing statistics are printed if verbose=True:
    - Total entries processed
    - Parsing success rates by category
    - Average skills per job
    - Total skill-job pairs created

    Typical results:
    - Complete entries: ~72.5% of total
    - Type-only entries: ~27.5% (skipped)
    - Average skills per job: ~9.36
    """
    if verbose:
        print("Transforming to long format with flexible skill parsing...")

    # Default columns to keep
    if keep_cols is None:
        keep_cols = ['Discipline', 'JobDate']

    # Ensure required columns exist
    required_cols = [id_col, skill_col] + keep_cols
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    long_data = []

    # Convert to pandas for row-wise processing
    pandas_df = df.select(required_cols).to_pandas()

    total_processed = 0

    # Aggregated parsing statistics
    total_stats = {
        'total_entries': 0,
        'complete_entries': 0,
        'family_skill_only': 0,
        'family_only': 0,
        'type_only': 0,
        'empty_entries': 0,
        'unparseable': 0,
        'empty_input': 0,
        'total_jobs_with_skills': 0,
        'total_successfully_parsed_skills': 0
    }

    for i in range(0, len(pandas_df), chunk_size):
        chunk = pandas_df.iloc[i:i + chunk_size]

        for _, row in chunk.iterrows():
            # Extract base job information
            job_data = {id_col: row[id_col]}
            for col in keep_cols:
                job_data[col] = row[col]

            skill_clusters = row[skill_col]

            # Parse skills using flexible method
            parsed_skills, parsing_stats = parse_skill_cluster(skill_clusters)

            # Aggregate statistics
            for key, value in parsing_stats.items():
                if key in total_stats:
                    total_stats[key] += value

            # Track jobs that have successfully parsed skills
            if len(parsed_skills) > 0:
                total_stats['total_jobs_with_skills'] += 1
                total_stats['total_successfully_parsed_skills'] += len(parsed_skills)

            for family, skill, skill_type in parsed_skills:
                skill_data = job_data.copy()
                skill_data.update({
                    'SkillFamily': family,
                    'Skill': skill,
                    'SkillType': skill_type
                })
                long_data.append(skill_data)

        total_processed += len(chunk)
        if verbose and total_processed % 50000 == 0:
            print(f"  Processed {total_processed:,} records...")

    # Print final parsing statistics
    if verbose:
        print(f"\n=== PARSING STATISTICS ===")
        print(f"Total skill entries processed: {total_stats['total_entries']:,}")
        total_entries_safe = max(total_stats['total_entries'], 1)
        print(f"Complete entries (Family: Skill; Type): {total_stats['complete_entries']:,} "
              f"({total_stats['complete_entries']/total_entries_safe*100:.1f}%)")
        print(f"Family & Skill only (no type): {total_stats['family_skill_only']:,} "
              f"({total_stats['family_skill_only']/total_entries_safe*100:.1f}%)")
        print(f"Family only: {total_stats['family_only']:,} "
              f"({total_stats['family_only']/total_entries_safe*100:.1f}%)")
        print(f"Skill type only: {total_stats['type_only']:,} "
              f"({total_stats['type_only']/total_entries_safe*100:.1f}%)")
        print(f"Empty entries: {total_stats['empty_entries']:,} "
              f"({total_stats['empty_entries']/total_entries_safe*100:.1f}%)")
        print(f"Unparseable entries: {total_stats['unparseable']:,} "
              f"({total_stats['unparseable']/total_entries_safe*100:.1f}%)")
        print(f"Records with empty input: {total_stats['empty_input']:,}")
        print(f"Total job postings processed: {len(pandas_df):,}")

        successfully_parsed = (total_stats['complete_entries'] +
                             total_stats['family_skill_only'] +
                             total_stats['family_only'])
        print(f"\nSuccessfully parsed entries: {successfully_parsed:,} "
              f"({successfully_parsed/total_entries_safe*100:.1f}%)")

        # Add average skills per job calculations
        if total_stats['total_jobs_with_skills'] > 0:
            avg_skills_per_job = (total_stats['total_successfully_parsed_skills'] /
                                total_stats['total_jobs_with_skills'])
            print(f"\n=== SKILLS PER JOB ANALYSIS ===")
            print(f"Jobs with successfully parsed skills: {total_stats['total_jobs_with_skills']:,}")
            print(f"Total successfully parsed skills: {total_stats['total_successfully_parsed_skills']:,}")
            print(f"Average skills per job (with skills): {avg_skills_per_job:.2f}")

            # Calculate overall average including jobs without skills
            overall_avg_skills = total_stats['total_successfully_parsed_skills'] / len(pandas_df)
            print(f"Average skills per job (all jobs): {overall_avg_skills:.2f}")
        else:
            print(f"\nNo jobs with successfully parsed skills found.")

    if long_data:
        result_df = pl.DataFrame(long_data)
        if verbose:
            print(f"Final skill-job pairs created: {len(result_df):,}")
        return result_df
    else:
        # Return empty DataFrame with correct schema
        schema = {id_col: pl.Int64}
        for col in keep_cols:
            schema[col] = pl.Utf8  # Default to string type
        schema.update({
            'SkillFamily': pl.Utf8,
            'Skill': pl.Utf8,
            'SkillType': pl.Utf8
        })
        return pl.DataFrame(schema=schema)


def calculate_skill_prevalence(
    skills_df: pl.DataFrame,
    id_col: str = 'JobID',
    skill_col: str = 'Skill',
    year_col: str = 'Year',
    verbose: bool = True
) -> pl.DataFrame:
    """
    Calculate skill prevalence (proportion of jobs) by year.

    For each skill in each year, calculates:
    - Number of jobs mentioning the skill
    - Total jobs in that year
    - Proportion (jobs with skill / total jobs)

    Parameters
    ----------
    skills_df : pl.DataFrame
        Long-format DataFrame with one row per skill-job pair
    id_col : str, default 'JobID'
        Name of the unique job identifier column
    skill_col : str, default 'Skill'
        Name of the skill column
    year_col : str, default 'Year'
        Name of the year column
    verbose : bool, default True
        If True, prints summary statistics

    Returns
    -------
    pl.DataFrame
        DataFrame with columns:
        - Skill
        - Year
        - jobs_with_skill: Number of unique jobs mentioning this skill
        - total_jobs: Total unique jobs in this year
        - prevalence: Proportion (0-1) of jobs with this skill

    Examples
    --------
    >>> prevalence = calculate_skill_prevalence(skills_long_df)
    >>> # Filter to one skill
    >>> cad_prev = prevalence.filter(pl.col('Skill') == 'CAD')
    >>> cad_prev.select(['Year', 'prevalence'])

    Notes
    -----
    The prevalence is calculated as:
    prevalence = (unique jobs with skill) / (total unique jobs in year)

    This accounts for the fact that a skill can appear multiple times
    per job (different families/types), but each job should only be
    counted once per skill per year.
    """
    if verbose:
        print("Calculating skill prevalence by year...")

    # Step 1: Count unique jobs per skill per year
    skills_per_year = skills_df.group_by([skill_col, year_col]).agg([
        pl.n_unique(id_col).alias('jobs_with_skill')
    ])

    # Step 2: Count total unique jobs per year
    total_jobs_per_year = skills_df.group_by(year_col).agg([
        pl.n_unique(id_col).alias('total_jobs')
    ])

    # Step 3: Join and calculate prevalence
    prevalence = skills_per_year.join(
        total_jobs_per_year,
        on=year_col,
        how='left'
    ).with_columns([
        (pl.col('jobs_with_skill') / pl.col('total_jobs')).alias('prevalence')
    ]).sort([skill_col, year_col])

    if verbose:
        total_skills = prevalence.select(pl.n_unique(skill_col)).item()
        year_range = prevalence.select([
            pl.col(year_col).min().alias('min_year'),
            pl.col(year_col).max().alias('max_year')
        ]).row(0)
        print(f"Calculated prevalence for {total_skills:,} unique skills")
        print(f"Year range: {year_range[0]} - {year_range[1]}")
        print(f"Total records: {len(prevalence):,}")

    return prevalence


def filter_common_skills(
    prevalence_df: pl.DataFrame,
    min_avg_prevalence: float = 0.02,
    skill_col: str = 'Skill',
    verbose: bool = True
) -> List[str]:
    """
    Filter to skills with minimum average prevalence across all years.

    Identifies "common" skills that appear in at least min_avg_prevalence
    proportion of jobs on average. This removes rare/niche skills that
    lack sufficient data for temporal trend analysis.

    Parameters
    ----------
    prevalence_df : pl.DataFrame
        DataFrame from calculate_skill_prevalence()
    min_avg_prevalence : float, default 0.02
        Minimum average prevalence threshold (0-1).
        Default 0.02 means skill must appear in ≥2% of jobs on average.
    skill_col : str, default 'Skill'
        Name of the skill column
    verbose : bool, default True
        If True, prints filtering statistics

    Returns
    -------
    list of str
        List of skill names meeting the prevalence threshold

    Examples
    --------
    >>> prevalence = calculate_skill_prevalence(skills_long_df)
    >>> common_skills = filter_common_skills(prevalence, min_avg_prevalence=0.02)
    >>> print(f"Found {len(common_skills)} common skills")

    >>> # Filter original data to common skills only
    >>> filtered = skills_long_df.filter(pl.col('Skill').is_in(common_skills))

    Notes
    -----
    The threshold of 2% (0.02) is commonly used in workforce research to:
    - Ensure sufficient sample size for statistical analysis
    - Remove one-off or highly specialized skills
    - Focus on skills with meaningful temporal coverage

    Typical results with min_avg_prevalence=0.02:
    - Approximately 200-250 skills pass threshold
    - Represents core skill set for the occupation
    """
    if verbose:
        print(f"Filtering to skills with ≥{min_avg_prevalence*100:.1f}% average prevalence...")

    # Calculate average prevalence across all years for each skill
    avg_prevalence = prevalence_df.group_by(skill_col).agg([
        pl.col('prevalence').mean().alias('avg_prevalence')
    ]).filter(
        pl.col('avg_prevalence') >= min_avg_prevalence
    ).sort('avg_prevalence', descending=True)

    common_skills = avg_prevalence.select(skill_col).to_series().to_list()

    if verbose:
        total_skills = prevalence_df.select(pl.n_unique(skill_col)).item()
        print(f"Skills meeting threshold: {len(common_skills):,} / {total_skills:,} "
              f"({len(common_skills)/total_skills*100:.1f}%)")

        if len(avg_prevalence) > 0:
            top_skill = avg_prevalence.row(0, named=True)
            print(f"Highest average prevalence: '{top_skill[skill_col]}' ({top_skill['avg_prevalence']*100:.1f}%)")

    return common_skills


def get_top_skills(
    skills_df: pl.DataFrame,
    n: int = 30,
    id_col: str = 'JobID',
    skill_col: str = 'Skill',
    exclude_skills: Optional[List[str]] = None
) -> pl.DataFrame:
    """
    Get top N skills by job proportion.

    Parameters
    ----------
    skills_df : pl.DataFrame
        Long-format skills DataFrame
    n : int, default 30
        Number of top skills to return
    id_col : str, default 'JobID'
        Name of job identifier column
    skill_col : str, default 'Skill'
        Name of skill column
    exclude_skills : list of str or None, optional
        Skills to exclude from ranking (e.g., overly generic skills)

    Returns
    -------
    pl.DataFrame
        Top N skills with columns:
        - Rank
        - Skill
        - unique_jobs: Number of jobs mentioning skill
        - job_proportion_pct: Percentage of jobs

    Examples
    --------
    >>> top_30 = get_top_skills(skills_long_df, n=30)
    >>> # Exclude generic "Mechanical Engineering" skill
    >>> top_30_clean = get_top_skills(
    ...     skills_long_df,
    ...     n=30,
    ...     exclude_skills=['Mechanical Engineering']
    ... )
    """
    # Calculate skill counts
    skill_counts = skills_df.group_by(skill_col).agg([
        pl.n_unique(id_col).alias('unique_jobs')
    ]).sort('unique_jobs', descending=True)

    total_jobs = skills_df.select(pl.n_unique(id_col)).item()

    skill_proportions = skill_counts.with_columns([
        (pl.col('unique_jobs') / total_jobs * 100).round(2).alias('job_proportion_pct')
    ])

    # Apply exclusions if provided
    if exclude_skills:
        skill_proportions = skill_proportions.filter(
            ~pl.col(skill_col).is_in(exclude_skills)
        )

    # Get top N
    top_skills = skill_proportions.head(n)

    # Add rank
    top_skills = top_skills.with_columns([
        pl.lit(range(1, len(top_skills) + 1)).alias('Rank')
    ]).select(['Rank', skill_col, 'unique_jobs', 'job_proportion_pct'])

    return top_skills


def compute_skills_per_job_distribution(
    skills_df: pl.DataFrame,
    id_col: str = 'JobID',
    skill_col: str = 'Skill'
) -> Dict:
    """
    Compute descriptive statistics and range breakdown for skills per job.

    Groups the long-format skills DataFrame by job, counts unique skills
    per job, and returns summary statistics plus a binned distribution.

    Parameters
    ----------
    skills_df : pl.DataFrame
        Long-format DataFrame with one row per skill-job pair
    id_col : str, default 'JobID'
        Name of the unique job identifier column
    skill_col : str, default 'Skill'
        Name of the skill column

    Returns
    -------
    dict
        Dictionary with two keys:
        - 'statistics': dict with 'min', 'max', 'mean', 'std' (float values)
        - 'ranges': list of dicts, each with 'label' (str), 'count' (int),
          'percent' (float) representing binned skill count distribution

    Examples
    --------
    >>> dist = compute_skills_per_job_distribution(skills_long_df)
    >>> print(f"Mean skills per job: {dist['statistics']['mean']:.2f}")
    >>> for r in dist['ranges']:
    ...     print(f"{r['label']}: {r['count']:,} ({r['percent']:.2f}%)")
    """
    skills_per_job = skills_df.group_by(id_col).agg(
        pl.n_unique(skill_col).alias('skill_count')
    )

    counts = skills_per_job['skill_count']
    total_jobs = len(skills_per_job)

    statistics = {
        'min': float(counts.min()),
        'max': float(counts.max()),
        'mean': float(counts.mean()),
        'std': float(counts.std()),
    }

    range_bins = [
        ('1-5 skills', 1, 5),
        ('6-10 skills', 6, 10),
        ('11-20 skills', 11, 20),
        ('21-50 skills', 21, 50),
        ('51+ skills', 51, None),
    ]

    ranges = []
    for label, low, high in range_bins:
        if high is not None:
            count = skills_per_job.filter(
                (pl.col('skill_count') >= low) & (pl.col('skill_count') <= high)
            ).height
        else:
            count = skills_per_job.filter(
                pl.col('skill_count') >= low
            ).height
        ranges.append({
            'label': label,
            'count': count,
            'percent': count / total_jobs * 100 if total_jobs > 0 else 0.0,
        })

    return {'statistics': statistics, 'ranges': ranges}


def display_skills_per_job_table(distribution: Dict) -> None:
    """
    Display an HTML-formatted summary table for skills per job distribution.

    Renders two sub-tables using IPython HTML display:
    1. Descriptive statistics (min, max, std, mean)
    2. Binned distribution by skill count range

    Parameters
    ----------
    distribution : dict
        Output from ``compute_skills_per_job_distribution()``.
        Must contain 'statistics' and 'ranges' keys.

    Returns
    -------
    None
        Displays HTML inline in a Jupyter notebook.

    Examples
    --------
    >>> dist = compute_skills_per_job_distribution(skills_long_df)
    >>> display_skills_per_job_table(dist)
    """
    from IPython.display import display, HTML

    stats = distribution['statistics']
    ranges = distribution['ranges']

    style = (
        "style='border-collapse: collapse; margin: 10px 0; font-size: 14px;'"
    )
    th_style = (
        "style='border: 1px solid #ddd; padding: 8px 16px; "
        "background-color: #f2f2f2; text-align: left;'"
    )
    td_style = (
        "style='border: 1px solid #ddd; padding: 8px 16px; text-align: left;'"
    )

    stats_rows = [
        ('Minimum', f"{stats['min']:.1f} skill{'s' if stats['min'] != 1 else ''}"),
        ('Maximum', f"{stats['max']:.1f} skills"),
        ('Standard Deviation', f"{stats['std']:.2f} skills"),
        ('Mean', f"{stats['mean']:.2f} skills"),
    ]

    html = f"<h4>Distribution of Skills per Job</h4>"
    html += f"<table {style}>"
    html += f"<tr><th {th_style}>Statistic</th><th {th_style}>Value</th></tr>"
    for label, value in stats_rows:
        html += f"<tr><td {td_style}>{label}</td><td {td_style}>{value}</td></tr>"
    html += "</table>"

    html += f"<h4>Distribution by Skill Count Range</h4>"
    html += f"<table {style}>"
    html += (
        f"<tr><th {th_style}>Range</th>"
        f"<th {th_style}>Jobs (% of total)</th></tr>"
    )
    for r in ranges:
        html += (
            f"<tr><td {td_style}>{r['label']}</td>"
            f"<td {td_style}>{r['count']:,} ({r['percent']:.2f})</td></tr>"
        )
    html += "</table>"

    display(HTML(html))


# Example usage and testing
if __name__ == "__main__":
    print("Testing skills_utils module")
    print("=" * 60)

    # Test 1: parse_skill_cluster
    print("\n1. Testing parse_skill_cluster():")
    print("-" * 60)

    test_cases = [
        "Engineering: CAD;Technical Skills",
        "Engineering: CAD",  # Missing type
        "Specialized Skills",  # Type only
        "Engineering: CAD;Technical|Specialized Skills",  # Mixed
        None,  # Empty input
        ""  # Empty string
    ]

    for i, test in enumerate(test_cases, 1):
        skills, stats = parse_skill_cluster(test)
        print(f"\nTest {i}: {test!r}")
        print(f"  Parsed {len(skills)} skills")
        print(f"  Stats: {stats}")
        if skills:
            for skill in skills:
                print(f"    - {skill}")

    print("\n" + "=" * 60)
    print("Module tests complete!")
