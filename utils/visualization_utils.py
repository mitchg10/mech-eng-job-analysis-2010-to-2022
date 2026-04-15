"""
Visualization Utilities for Skill Trajectory Analysis

Provides plotting functions for skill trajectories, diagnostic plots,
and publication-quality figures.
"""

from typing import Optional, List, Dict
from pathlib import Path
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from .statistics_utils import (
    ModelResult, ModelSelection,
    fit_candidate_models, select_best_model
)


# Model display names for legend
MODEL_DISPLAY_NAMES = {
    'null': 'Constant',
    'linear': 'Linear',
    'log_year': 'Log-Year',
    'exponential': 'Exponential',
    'quadratic': 'Quadratic'
}


def _generate_smooth_curve(
    model_name: str,
    model_coefficients: dict,
    prevalence_df: pl.DataFrame,
    skill: str,
    n_points: int = 100
) -> tuple:
    """
    Generate smooth model curve using stored coefficients.

    This function recreates the model predictions at many dense points to produce
    a mathematically accurate smooth curve, rather than connecting the 13 discrete
    fitted values with straight line segments.

    Parameters
    ----------
    model_name : str
        Model type ('null', 'linear', 'log_year', 'exponential', 'quadratic')
    model_coefficients : dict
        Coefficient dict from summary_df['Model_Coefficients']
        Keys depend on model type (e.g., 'Intercept', 'year_centered', 'log_year')
    prevalence_df : pl.DataFrame
        Prevalence data (needed for centering parameters)
    skill : str
        Skill name
    n_points : int, default 100
        Number of points for smooth curve

    Returns
    -------
    years_dense : np.ndarray
        Dense year array (n_points values)
    y_dense : np.ndarray
        Predicted prevalence values (not percentages, in [0, 1] scale)

    Notes
    -----
    The centering parameter (year_mean, min_year) are computed from the
    prevalence_df to match the centering used during model fitting in
    statistics_utils.py (B&A 2002: center only, do not scale).

    Model equations:
    - null: y = intercept
    - linear: y = intercept + coef_year * year_centered + coef_text * text_std
    - log_year: y = intercept + coef_log * log(year - min_year + 1) + coef_text * text_std
    - exponential: y = exp(intercept + coef_year * year_centered + coef_text * text_std)
    - quadratic: y = intercept + coef_year * year_centered + coef_sq * year_centered² + coef_text * text_std
    """
    # Filter to skill data
    skill_prev = prevalence_df.filter(pl.col('Skill') == skill)

    if len(skill_prev) == 0:
        raise ValueError(f"No data found for skill: {skill}")

    years_orig = skill_prev['Year'].to_numpy()

    # Compute centering parameter (matching statistics_utils.py: center only, do not scale)
    year_mean = years_orig.mean()
    min_year = years_orig.min()
    max_year = years_orig.max()
    job_text_mean = skill_prev['job_text_length_std'].mean()

    # Create dense year array
    years_dense = np.linspace(min_year, max_year, n_points)

    # Get coefficients
    coef = model_coefficients
    intercept = coef.get('Intercept', 0.0)
    job_text_coef = coef.get('job_text_length_std', 0.0)

    # Compute predictions based on model type
    if model_name == 'null':
        # Null model: constant prevalence
        y_dense = intercept * np.ones_like(years_dense)

    elif model_name == 'linear':
        # Linear model: prevalence ~ year_centered + job_text_length_std
        year_centered_dense = years_dense - year_mean
        y_dense = intercept + coef['year_centered'] * year_centered_dense + job_text_coef * job_text_mean

    elif model_name == 'log_year':
        # Log-year model: prevalence ~ log(year - min_year + 1) + job_text_length_std
        log_year_dense = np.log(years_dense - min_year + 1)
        y_dense = intercept + coef['log_year'] * log_year_dense + job_text_coef * job_text_mean

    elif model_name == 'exponential':
        # Exponential model: log(prevalence) ~ year_centered + job_text_length_std
        # Need to back-transform from log space
        year_centered_dense = years_dense - year_mean
        log_y = intercept + coef['year_centered'] * year_centered_dense + job_text_coef * job_text_mean
        y_dense = np.exp(log_y)

    elif model_name == 'quadratic':
        # Quadratic model: prevalence ~ year_centered + year_centered_sq + job_text_length_std
        year_centered_dense = years_dense - year_mean
        year_centered_sq_dense = year_centered_dense ** 2
        y_dense = (intercept +
                   coef['year_centered'] * year_centered_dense +
                   coef['year_centered_sq'] * year_centered_sq_dense +
                   job_text_coef * job_text_mean)

    else:
        raise ValueError(f"Unknown model type: {model_name}")

    return years_dense, y_dense


def plot_trajectory_grid(
    summary_df: pl.DataFrame,
    prevalence_df: pl.DataFrame,
    discriminability: str,
    output_path: str,
    max_cols: int = 5,
    figsize_per_subplot: tuple = (4, 3),
    show_model_fit: bool = True,
    show_statistics: bool = True,
    show_smooth_curve: bool = True
):
    """
    Plot skill trajectories in a grid layout with optional model fits.

    Parameters
    ----------
    summary_df : pl.DataFrame
        Summary with columns: Skill, Selected_Model, Discriminability,
        Assumption_Quality, R_squared, AICc, Weight, Fitted_Values,
        Model_Coefficients, etc.
    prevalence_df : pl.DataFrame
        Prevalence data with columns: Skill, Year, prevalence, job_text_length_std
    discriminability : str
        Discriminability level to plot ('Strong', 'Moderate', or 'Weak')
    output_path : str
        Path to save figure
    max_cols : int, default 5
        Maximum columns in grid
    figsize_per_subplot : tuple, default (4, 3)
        Size of each subplot in inches
    show_model_fit : bool, default True
        If True, overlay best-fitting model curve
    show_statistics : bool, default True
        If True, display R² and AICc on each subplot
    show_smooth_curve : bool, default True
        If True, generates smooth model curves using dense predictions (100 points).
        If False, uses the 13 stored fitted values from summary_df (original behavior).
        Smooth curves are more visually clear and mathematically accurate.

    Notes
    -----
    When show_smooth_curve=True, the function generates predictions at 100 evenly-spaced
    years using the model coefficients stored in summary_df. This produces mathematically
    accurate smooth curves rather than connecting 13 discrete predictions.

    The smooth curve uses the same centering as the original model fitting:
    - year_centered = Year - Year.mean()
    - log_year = log(Year - min_year + 1)
    - Coefficients are retrieved from summary_df['Model_Coefficients']
    """
    # Filter to discriminability level
    tier_skills = summary_df.filter(pl.col('Discriminability') == discriminability)
    skills = tier_skills['Skill'].to_list()

    if not skills:
        print(f"No skills with {discriminability} discriminability")
        return

    # Calculate grid dimensions
    n_skills = len(skills)
    n_cols = min(n_skills, max_cols)
    n_rows = (n_skills + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_subplot[0] * n_cols, figsize_per_subplot[1] * n_rows)
    )

    # Flatten axes for easier iteration
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten() if n_rows > 1 or n_cols > 1 else axes

    for idx, skill in enumerate(skills):
        ax = axes[idx]

        # Get data for this skill
        skill_data = prevalence_df.filter(pl.col('Skill') == skill).sort('Year')

        if len(skill_data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(skill[:30])
            continue

        years = skill_data['Year'].to_numpy()
        prev = skill_data['prevalence'].to_numpy() * 100  # Convert to percentage

        # Plot line connecting datapoints (lighter hue, behind everything)
        ax.plot(years, prev, '-', color='steelblue', alpha=0.4,
                linewidth=1.5, zorder=1, label='_nolegend_')

        # Plot observed data points on top of line
        ax.plot(years, prev, 'o', color='steelblue', markersize=4,
                label='Observed', zorder=2)

        # Get model info from summary_df
        skill_summary = tier_skills.filter(pl.col('Skill') == skill).row(0, named=True)
        model_name = skill_summary['Selected_Model']
        r_squared = skill_summary.get('R_squared', 0)
        weight = skill_summary.get('Weight', 0)
        disc = skill_summary.get('Discriminability', 'N/A')
        aq = skill_summary.get('Assumption_Quality', 'N/A')

        # Overlay model fit if requested
        if show_model_fit:
            # Get fitted values from summary_df
            fitted_values_list = skill_summary.get('Fitted_Values', None)
            model_coefficients = skill_summary.get('Model_Coefficients', None)
            if fitted_values_list and model_coefficients is not None:
                model_display = MODEL_DISPLAY_NAMES.get(model_name, model_name)
                if show_smooth_curve and model_coefficients is not None:
                    # Generate smooth curve from model equation
                    try:
                        years_dense, y_dense = _generate_smooth_curve(
                            model_name, model_coefficients, prevalence_df, skill, n_points=100
                        )
                        ax.plot(years_dense, y_dense * 100, '-', color='coral',
                                linewidth=2.5, label=f'{model_display}', zorder=3)
                    except Exception as e:
                        # Fallback to jagged line if smooth curve fails
                        print(f"Warning: Smooth curve failed for {skill}, using fitted values: {e}")
                        fitted_values = np.array(fitted_values_list) * 100
                        ax.plot(years, fitted_values, '-', color='coral',
                                linewidth=2.5, label=f'{model_display}', zorder=3)
                else:
                    # Original behavior: plot stored fitted values
                    fitted_values = np.array(fitted_values_list) * 100
                    ax.plot(years, fitted_values, '-', color='coral',
                            linewidth=2.5, label=f'{model_display}', zorder=3)
        # Add statistics annotation
        if show_statistics:
            stats_text = f"R²={r_squared:.3f}\nDisc: {disc}\nAQ: {aq}"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                    fontsize=6, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Add legend
        ax.legend(loc='lower right', fontsize=6, framealpha=0.9)

        ax.set_title(skill[:30], fontsize=9)
        ax.set_xlabel('Year', fontsize=8)
        ax.set_ylabel('Prevalence (%)', fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)

    # Hide unused subplots
    for idx in range(n_skills, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f'{discriminability} Discriminability Skill Trajectories',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved {discriminability} discriminability trajectories to {output_path}")


def plot_single_skill_trajectory(
    skill: str,
    prevalence_df: pl.DataFrame,
    summary_df: Optional[pl.DataFrame] = None,
    output_path: Optional[str] = None,
    figsize: tuple = (10, 6),
    show_all_models: bool = False,
    show_smooth_curve: bool = True
):
    """
    Plot detailed trajectory for a single skill with model fit and statistics.

    Creates a publication-quality figure showing observed data, best-fitting
    model curve, and comprehensive fit statistics in a legend box.

    Parameters
    ----------
    skill : str
        Skill name to plot
    prevalence_df : pl.DataFrame
        Prevalence data with columns: Skill, Year, prevalence, job_text_length_std
    summary_df : pl.DataFrame, optional
        Summary DataFrame with pre-computed statistics. If None, will fit models.
    output_path : str, optional
        Path to save figure. If None, displays instead.
    figsize : tuple, default (10, 6)
        Figure size in inches
    show_all_models : bool, default False
        If True, show all candidate model fits (not just best)
    show_smooth_curve : bool, default True
        If True, generates smooth model curves using dense predictions (100 points).
        If False, uses fitted values from model (13 discrete points).
        Smooth curves are more visually clear and mathematically accurate.
    Returns
    -------
    dict
        Dictionary with model selection results and statistics
    """
    # Get skill data
    skill_data = prevalence_df.filter(pl.col('Skill') == skill).sort('Year')

    if len(skill_data) == 0:
        raise ValueError(f"No data found for skill: {skill}")

    years = skill_data['Year'].to_numpy()
    prev = skill_data['prevalence'].to_numpy() * 100  # Convert to percentage

    # Fit models
    models = fit_candidate_models(prevalence_df, skill)
    selection = select_best_model(models)

    # Get pre-computed stats if available
    if summary_df is not None:
        skill_row = summary_df.filter(pl.col('Skill') == skill)
        if len(skill_row) > 0:
            skill_stats = skill_row.row(0, named=True)
            discriminability = skill_stats.get('Discriminability', 'N/A')
            assumption_quality = skill_stats.get('Assumption_Quality', 'N/A')
            trajectory = skill_stats.get('Trajectory_Class', 'N/A')
        else:
            discriminability = 'N/A'
            assumption_quality = 'N/A'
            trajectory = 'N/A'
    else:
        discriminability = 'N/A'
        assumption_quality = 'N/A'
        trajectory = 'N/A'

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot line connecting datapoints (lighter hue, behind everything)
    ax.plot(years, prev, '-', color='steelblue', alpha=0.4,
            linewidth=1.5, zorder=1, label='_nolegend_')

    # Plot observed data points on top of line
    ax.plot(years, prev, 'o', color='steelblue', markersize=8,
            label='Observed Data', zorder=2, markeredgecolor='white',
            markeredgewidth=1)

    # Color palette for models
    model_colors = {
        'null': '#808080',      # Gray
        'linear': '#E74C3C',    # Red
        'log_year': '#9B59B6',  # Purple
        'exponential': '#27AE60',  # Green
        'quadratic': '#F39C12'  # Orange
    }

    if show_all_models:
        # Plot all model fits
        for model_name_iter, model_result in models.items():
            try:
                display_name = MODEL_DISPLAY_NAMES.get(model_name_iter, model_name_iter)
                is_best = model_name_iter == selection.best_model
                linewidth = 2.5 if is_best else 1.5
                linestyle = '-' if is_best else '--'
                alpha = 1.0 if is_best else 0.6

                label = f"{display_name}"
                if is_best:
                    label += " (Best)"

                # Best model on top (zorder=3), other models behind (zorder=2.5)
                zorder_val = 3 if is_best else 2.5
                if show_smooth_curve:
                    # Generate smooth curve from model coefficients
                    try:
                        # Extract coefficients from fitted model
                        fitted_model = model_result.fitted_model
                        model_coefs = {
                            'Intercept': fitted_model.params.get('Intercept', 0.0),
                            'year_centered': fitted_model.params.get('year_centered', 0.0),
                            'log_year': fitted_model.params.get('log_year', 0.0),
                            'year_centered_sq': fitted_model.params.get('year_centered_sq', 0.0),
                            'job_text_length_std': fitted_model.params.get('job_text_length_std', 0.0)
                        }

                        years_dense, y_dense = _generate_smooth_curve(
                            model_name_iter, model_coefs, prevalence_df, skill, n_points=100
                        )
                        ax.plot(years_dense, y_dense * 100, linestyle,
                                color=model_colors.get(model_name_iter, 'gray'),
                                linewidth=linewidth, alpha=alpha, label=label, zorder=zorder_val)
                    except Exception as e:
                        # Fallback to fitted values
                        fitted = model_result.fitted_model
                        if model_name_iter == 'exponential':
                            fitted_values = np.exp(fitted.fittedvalues) * 100
                        else:
                            fitted_values = fitted.fittedvalues * 100
                        ax.plot(years, fitted_values, linestyle,
                                color=model_colors.get(model_name_iter, 'gray'),
                                linewidth=linewidth, alpha=alpha, label=label, zorder=zorder_val)
                else:
                    # Original behavior: use fitted values
                    fitted = model_result.fitted_model
                    if model_name_iter == 'exponential':
                        fitted_values = np.exp(fitted.fittedvalues) * 100
                    else:
                        fitted_values = fitted.fittedvalues * 100
                    ax.plot(years, fitted_values, linestyle,
                            color=model_colors.get(model_name_iter, 'gray'),
                            linewidth=linewidth, alpha=alpha, label=label, zorder=zorder_val)
            except Exception:
                pass
    else:
        # Plot only best model
        if show_smooth_curve and summary_df is not None:
            # Try to use pre-computed coefficients from summary_df
            try:
                skill_row = summary_df.filter(pl.col('Skill') == skill)
                if len(skill_row) > 0:
                    skill_stats = skill_row.row(0, named=True)
                    model_coefficients = skill_stats.get('Model_Coefficients', None)

                    if model_coefficients is not None:
                        years_dense, y_dense = _generate_smooth_curve(
                            best_model_name, model_coefficients, prevalence_df, skill, n_points=100
                        )
                        display_name = MODEL_DISPLAY_NAMES.get(best_model_name, best_model_name)
                        ax.plot(years_dense, y_dense * 100, '-',
                                color=model_colors.get(best_model_name, 'coral'),
                                linewidth=2.5, label=f'{display_name} Model', zorder=3)
                    else:
                        raise ValueError("Model_Coefficients not found in summary_df")
                else:
                    raise ValueError("Skill not found in summary_df")
            except Exception as e:
                # Fallback to refitting if needed
                if best_model_name in models:
                    fitted = models[best_model_name].fitted_model
                    if best_model_name == 'exponential':
                        fitted_values = np.exp(fitted.fittedvalues) * 100
                    else:
                        fitted_values = fitted.fittedvalues * 100

                    display_name = MODEL_DISPLAY_NAMES.get(best_model_name, best_model_name)
                    ax.plot(years, fitted_values, '-',
                            color=model_colors.get(best_model_name, 'coral'),
                            linewidth=2.5, label=f'{display_name} Model', zorder=3)
        else:
            # Original behavior: use fitted values from models
            if best_model_name in models:
                fitted = models[best_model_name].fitted_model
                if best_model_name == 'exponential':
                    fitted_values = np.exp(fitted.fittedvalues) * 100
                else:
                    fitted_values = fitted.fittedvalues * 100

                display_name = MODEL_DISPLAY_NAMES.get(best_model_name, best_model_name)
                ax.plot(years, fitted_values, '-',
                        color=model_colors.get(best_model_name, 'coral'),
                        linewidth=2.5, label=f'{display_name} Model', zorder=3)

    # Build statistics text box
    stats_lines = [
        f"Best Model: {MODEL_DISPLAY_NAMES.get(selection.best_model, selection.best_model)}",
        f"R² = {selection.r_squared:.4f}",
        f"Adj. R² = {selection.adj_r_squared:.4f}",
        f"AICc = {selection.aicc:.2f}",
        f"Akaike Weight = {selection.weight:.3f}",
        f"Δᵢ = {selection.delta_i:.2f}",
        "",
        f"Trajectory: {trajectory}",
        f"Discriminability: {discriminability}",
        f"Assumption Quality: {assumption_quality}",
    ]

    # Add competitive models if any
    if len(selection.competitive_models) > 1:
        competitors = [m for m in selection.competitive_models if m != selection.best_model]
        if competitors:
            comp_names = [MODEL_DISPLAY_NAMES.get(m, m) for m in competitors]
            stats_lines.insert(6, f"Competitive: {', '.join(comp_names)}")

    stats_text = '\n'.join(stats_lines)

    # Add statistics box
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace', bbox=props)

    # Formatting
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Prevalence (%)', fontsize=12)
    ax.set_title(f'Skill Trajectory: {skill}', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)

    # Set x-axis to show all years
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=45, ha='right')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved trajectory plot to {output_path}")
    else:
        plt.show()

    return {
        'skill': skill,
        'best_model': selection.best_model,
        'r_squared': selection.r_squared,
        'adj_r_squared': selection.adj_r_squared,
        'aicc': selection.aicc,
        'weight': selection.weight,
        'delta_i': selection.delta_i,
        'competitive_models': selection.competitive_models,
        'trajectory': trajectory,
        'discriminability': discriminability,
        'assumption_quality': assumption_quality,
    }


def plot_residual_diagnostics(
    model_result: ModelResult,
    skill_name: str,
    output_path: str
):
    """
    Create 4-panel diagnostic plot for model residuals.

    Panels:
    1. Residuals vs Fitted
    2. Q-Q plot
    3. Scale-Location
    4. Residuals vs Leverage

    Parameters
    ----------
    model_result : ModelResult
        Fitted model
    skill_name : str
        Skill name for title
    output_path : str
        Path to save figure
    """
    fitted = model_result.fitted_model
    residuals = model_result.residuals
    fitted_values = fitted.fittedvalues

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1: Residuals vs Fitted
    axes[0, 0].scatter(fitted_values, residuals, alpha=0.6)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_xlabel('Fitted values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Fitted')
    axes[0, 0].grid(alpha=0.3)

    # Panel 2: Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Normal Q-Q')
    axes[0, 1].grid(alpha=0.3)

    # Panel 3: Scale-Location
    standardized_resid = residuals / np.std(residuals)
    axes[1, 0].scatter(fitted_values, np.sqrt(np.abs(standardized_resid)), alpha=0.6)
    axes[1, 0].set_xlabel('Fitted values')
    axes[1, 0].set_ylabel('√|Standardized residuals|')
    axes[1, 0].set_title('Scale-Location')
    axes[1, 0].grid(alpha=0.3)

    # Panel 4: Residuals vs Leverage
    try:
        leverage = fitted.get_influence().hat_matrix_diag
        axes[1, 1].scatter(leverage, standardized_resid, alpha=0.6)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Leverage')
        axes[1, 1].set_ylabel('Standardized residuals')
        axes[1, 1].set_title('Residuals vs Leverage')
        axes[1, 1].grid(alpha=0.3)
    except Exception:
        axes[1, 1].text(0.5, 0.5, 'Leverage unavailable', ha='center', va='center')

    plt.suptitle(f'Diagnostic Plots: {skill_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confidence_distribution(
    df: pl.DataFrame,
    score_col: str = 'Confidence_Score',
    output_path: Optional[str] = None
):
    """
    Plot distribution of confidence scores.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with confidence scores
    score_col : str, default 'Confidence_Score'
        Name of confidence score column
    output_path : str or None, optional
        Path to save figure. If None, displays instead.
    """
    scores = df[score_col].to_numpy()

    plt.figure(figsize=(12, 6))

    # Histogram
    plt.hist(scores, bins=30, color='blue', alpha=0.7, density=True, edgecolor='black')

    # Overlay normal distribution
    from scipy.stats import norm
    mean = np.mean(scores)
    std = np.std(scores)
    x = np.linspace(scores.min(), scores.max(), 200)
    plt.plot(x, norm.pdf(x, mean, std), 'r-', lw=2, label='Normal curve')

    plt.title('Distribution of Confidence Scores', fontsize=14, fontweight='bold')
    plt.xlabel('Confidence Score', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved confidence distribution to {output_path}")
    else:
        plt.show()


# Example usage
if __name__ == "__main__":
    print("visualization_utils module loaded")
    print("Use in notebooks for plotting")
