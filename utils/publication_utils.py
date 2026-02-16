"""
Publication Utilities for IEEE-Format Table Generation

Provides functions to generate publication-ready tables in CSV and LaTeX formats
following IEEE style guidelines.
"""

from typing import Dict, Optional
from pathlib import Path
import polars as pl


def extract_model_coefficient(
    model_coefficients: Dict[str, float],
    model_type: str,
    coefficient_type: str = 'slope'
) -> Optional[float]:
    """
    Extract the appropriate coefficient from model_coefficients based on model type.

    Parameters
    ----------
    model_coefficients : dict
        Dictionary of coefficient names to values from ModelSelection
    model_type : str
        Type of model ('linear', 'log_year', 'exponential', 'quadratic', 'null')
    coefficient_type : str
        Type of coefficient to extract:
        - 'slope': Primary trend coefficient
        - 'quadratic': Quadratic term for quadratic models
        - 'intercept': Model intercept

    Returns
    -------
    float or None
        The coefficient value, or None if not applicable

    Notes
    -----
    Coefficient mappings by model type:
    - linear: slope = year_std
    - log_year: slope = log_year
    - exponential: slope = year_std (in log space)
    - quadratic: slope = year_std, quadratic = year_std_sq
    - null: no slope (returns None)
    """
    if model_coefficients is None:
        return None

    if coefficient_type == 'intercept':
        return model_coefficients.get('Intercept')

    if coefficient_type == 'quadratic':
        return model_coefficients.get('year_std_sq')

    # Primary slope/growth rate coefficient
    if model_type == 'linear':
        return model_coefficients.get('year_std')
    elif model_type == 'log_year':
        return model_coefficients.get('log_year')
    elif model_type == 'exponential':
        return model_coefficients.get('year_std')
    elif model_type == 'quadratic':
        return model_coefficients.get('year_std')
    elif model_type == 'null':
        return None

    return None


def add_coefficient_columns(summary_df: pl.DataFrame) -> pl.DataFrame:
    """
    Add Slope, Growth_Rate, and Quadratic_Coef columns derived from Model_Coefficients.

    Parameters
    ----------
    summary_df : pl.DataFrame
        Summary DataFrame with 'Model_Coefficients' and 'Selected_Model' columns

    Returns
    -------
    pl.DataFrame
        DataFrame with added coefficient columns

    Notes
    -----
    Expects 'Model_Coefficients' column to contain dicts with coefficient values.
    If column doesn't exist, returns DataFrame unchanged.
    """
    if 'Model_Coefficients' not in summary_df.columns:
        return summary_df

    # Convert to Python for processing
    rows = summary_df.to_dicts()

    for row in rows:
        model_type = row.get('Selected_Model', '')
        coefficients = row.get('Model_Coefficients', {})

        # Extract slope/growth rate
        slope = extract_model_coefficient(coefficients, model_type, 'slope')
        row['Slope'] = slope
        row['Growth_Rate'] = slope  # Same as slope for most models

        # Extract quadratic coefficient
        row['Quadratic_Coef'] = extract_model_coefficient(coefficients, model_type, 'quadratic')

        # Calculate inflection point for quadratic models
        quad_coef = row.get('Quadratic_Coef')
        if model_type == 'quadratic' and quad_coef is not None and quad_coef != 0:
            linear_coef = slope if slope is not None else 0.0
            row['Inflection_Point'] = -linear_coef / (2 * quad_coef)
        else:
            row['Inflection_Point'] = None

    return pl.DataFrame(rows)


def generate_ieee_table(
    summary_df: pl.DataFrame,
    table_spec: Dict
) -> pl.DataFrame:
    """
    Generate IEEE-format table from summary data.

    Parameters
    ----------
    summary_df : pl.DataFrame
        Skill trajectory summary DataFrame
    table_spec : dict
        Table specification with keys:
        - filters: List of filter expressions
        - sort_by: Column to sort by
        - columns: Columns to include
        - limit: Maximum rows (optional)
        - ascending: Sort order (default False)

    Returns
    -------
    pl.DataFrame
        Formatted table ready for export

    Examples
    --------
    >>> table_spec = {
    ...     'filters': [pl.col('Percent_Change') > 0],
    ...     'sort_by': 'Percent_Change',
    ...     'columns': ['Skill', 'Start_Value_2010', 'End_Value_2022', 'Percent_Change'],
    ...     'limit': 20
    ... }
    >>> table = generate_ieee_table(summary_df, table_spec)
    """
    result = summary_df

    # Apply filters
    for filter_expr in table_spec.get('filters', []):
        result = result.filter(filter_expr)

    # Sort
    sort_col = table_spec.get('sort_by')
    ascending = table_spec.get('ascending', False)
    if sort_col:
        result = result.sort(sort_col, descending=not ascending)

    # Select columns
    columns = table_spec.get('columns')
    if columns:
        result = result.select(columns)

    # Limit rows
    limit = table_spec.get('limit')
    if limit:
        result = result.head(limit)

    return result


def export_to_latex(
    table_df: pl.DataFrame,
    caption: str,
    label: str
) -> str:
    """
    Export table to LaTeX format.

    Parameters
    ----------
    table_df : pl.DataFrame
        Table data
    caption : str
        Table caption
    label : str
        LaTeX label for referencing

    Returns
    -------
    str
        LaTeX table code

    Examples
    --------
    >>> latex = export_to_latex(table, "Top Growing Skills", "tab:growing")
    >>> print(latex)
    \\begin{table}...
    """
    # Convert to pandas for easier LaTeX generation
    pdf = table_df.to_pandas()

    # Generate LaTeX
    latex = pdf.to_latex(
        index=False,
        escape=False,
        column_format='l' + 'r' * (len(pdf.columns) - 1),
        caption=caption,
        label=label
    )

    return latex


def export_all_ieee_tables(
    summary_df: pl.DataFrame,
    output_dir: str
):
    """
    Generate and export all 8 IEEE tables.

    Tables:
    1. Top Growing Skills
    2. Top Declining Skills
    3. Stable High Prevalence Skills
    4. High Demand Skills
    5. Linear Trend Skills
    6. Logarithmic Trend Skills
    7. Quadratic Trend Skills
    8. Exponential Trend Skills

    Parameters
    ----------
    summary_df : pl.DataFrame
        Complete skill trajectory summary. If 'Model_Coefficients' column exists
        (containing dicts from ModelSelection.model_coefficients), will derive
        Slope, Growth_Rate, Quadratic_Coef, and Inflection_Point columns.
    output_dir : str
        Directory to save tables

    Notes
    -----
    Creates both CSV and combined LaTeX file.

    Coefficient derivation from Model_Coefficients:
    - Slope: year_std coefficient (linear, quadratic) or log_year coefficient (log_year)
    - Growth_Rate: Same as Slope for trend interpretation
    - Quadratic_Coef: year_std_sq coefficient (quadratic models only)
    - Inflection_Point: -linear_coef / (2 * quad_coef) for quadratic models
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Derive coefficient columns from Model_Coefficients if present
    df = add_coefficient_columns(summary_df)

    tables = {}
    latex_sections = []

    # Table 1: Top Growing Skills
    table1 = generate_ieee_table(df, {
        'filters': [pl.col('Percent_Change') > 0],
        'sort_by': 'Percent_Change',
        'columns': ['Skill', 'Start_Value_2010', 'End_Value_2022', 'Percent_Change', 'Selected_Model'],
        'limit': 20,
        'ascending': False
    })
    tables['Table_IEEE_1_Top_Growing_Skills'] = table1
    latex_sections.append(export_to_latex(table1, "Top 20 Growing Skills", "tab:growing"))

    # Table 2: Top Declining Skills
    table2 = generate_ieee_table(df, {
        'filters': [pl.col('Percent_Change') < 0],
        'sort_by': 'Percent_Change',
        'columns': ['Skill', 'Start_Value_2010', 'End_Value_2022', 'Percent_Change', 'Selected_Model'],
        'limit': 20,
        'ascending': True
    })
    tables['Table_IEEE_2_Top_Declining_Skills'] = table2
    latex_sections.append(export_to_latex(table2, "Top 20 Declining Skills", "tab:declining"))

    # Table 3: Stable High Prevalence
    table3 = generate_ieee_table(df, {
        'filters': [
            pl.col('Mean_Prevalence') > 0.10,
            pl.col('Percent_Change').abs() < 5
        ],
        'sort_by': 'Mean_Prevalence',
        'columns': ['Skill', 'Mean_Prevalence', 'Percent_Change', 'R_squared'],
        'limit': 20,
        'ascending': False
    })
    tables['Table_IEEE_3_Stable_High_Prevalence_Skills'] = table3
    latex_sections.append(export_to_latex(table3, "Stable High Prevalence Skills", "tab:stable"))

    # Table 4: High Demand Skills
    table4 = generate_ieee_table(df, {
        'filters': [
            (pl.col('Start_Value_2010') > 0.15) | (pl.col('End_Value_2022') > 0.15)
        ],
        'sort_by': 'End_Value_2022',
        'columns': ['Skill', 'Start_Value_2010', 'End_Value_2022', 'Trajectory_Class'],
        'limit': 20,
        'ascending': False
    })
    tables['Table_IEEE_4_High_Demand_Skills'] = table4
    latex_sections.append(export_to_latex(table4, "High Demand Skills", "tab:demand"))

    # Table 5: Linear Trends
    table5 = generate_ieee_table(df, {
        'filters': [pl.col('Selected_Model') == 'linear'],
        'sort_by': 'Slope',
        'columns': ['Skill', 'Slope', 'R_squared', 'Confidence_Tier', 'Trajectory_Class'],
        'limit': 20,
        'ascending': False
    })
    tables['Table_IEEE_5_Linear_Trend_Skills'] = table5
    latex_sections.append(export_to_latex(table5, "Skills with Linear Trends", "tab:linear"))

    # Table 6: Logarithmic Trends
    table6 = generate_ieee_table(df, {
        'filters': [pl.col('Selected_Model') == 'log_year'],
        'sort_by': 'R_squared',
        'columns': ['Skill', 'Growth_Rate', 'R_squared', 'Confidence_Tier'],
        'limit': 20,
        'ascending': False
    })
    tables['Table_IEEE_6_Logarithmic_Trend_Skills'] = table6
    latex_sections.append(export_to_latex(table6, "Skills with Logarithmic Trends", "tab:log"))

    # Table 7: Quadratic Trends
    table7 = generate_ieee_table(df, {
        'filters': [pl.col('Selected_Model') == 'quadratic'],
        'sort_by': 'R_squared',
        'columns': ['Skill', 'Quadratic_Coef', 'Inflection_Point', 'Trajectory_Class'],
        'limit': 20,
        'ascending': False
    })
    tables['Table_IEEE_7_Quadratic_Trend_Skills'] = table7
    latex_sections.append(export_to_latex(table7, "Skills with Quadratic Trends", "tab:quadratic"))

    # Table 8: Exponential Trends
    table8 = generate_ieee_table(df, {
        'filters': [pl.col('Selected_Model') == 'exponential'],
        'sort_by': 'Growth_Rate',
        'columns': ['Skill', 'Growth_Rate', 'R_squared', 'Trajectory_Class'],
        'limit': 20,
        'ascending': False
    })
    tables['Table_IEEE_8_Exponential_Trend_Skills'] = table8
    latex_sections.append(export_to_latex(table8, "Skills with Exponential Trends", "tab:exponential"))

    # Export all tables to CSV
    for name, table in tables.items():
        csv_path = output_path / f"{name}.csv"
        table.write_csv(csv_path)
        print(f"Exported {name} to {csv_path}")

    # Export combined LaTeX
    latex_combined = "\n\n".join(latex_sections)
    latex_path = output_path / "IEEE_Tables_Combined.tex"
    with open(latex_path, 'w') as f:
        f.write(latex_combined)
    print(f"Exported combined LaTeX to {latex_path}")

    print(f"\nGenerated {len(tables)} IEEE tables in {output_dir}")


# Example usage
if __name__ == "__main__":
    print("publication_utils module loaded")
    print("Use export_all_ieee_tables() to generate publication tables")
