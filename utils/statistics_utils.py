"""
Statistical Analysis Utilities

This module provides utilities for statistical modeling and trajectory analysis
of skill trends over time. It implements model fitting, information criterion
calculation, diagnostic testing, and trajectory classification.

Models fitted:
- Null: No temporal trend (baseline)
- Linear: Constant rate of change
- Log-year: Logarithmic time trend (decelerating)
- Exponential: Exponential growth/decline
- Quadratic: Accelerating/decelerating with inflection points
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import polars as pl
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from scipy.stats import shapiro
import warnings

warnings.filterwarnings('ignore')


@dataclass
class ModelResult:
    """
    Results from fitting a single model.

    Attributes
    ----------
    model_name : str
        Name of model ('null', 'linear', 'log_year', 'exponential', 'quadratic')
    fitted_model : object
        Fitted statsmodels result object
    formula : str
        Model formula used
    n_params : int
        Number of parameters in model
    loglik : float
        Log-likelihood of fitted model
    n_obs : int
        Number of observations
    residuals : np.ndarray
        Model residuals
    """
    model_name: str
    fitted_model: object
    formula: str
    n_params: int
    loglik: float
    n_obs: int
    residuals: np.ndarray


@dataclass
class DiagnosticResult:
    """
    Results from diagnostic tests.

    Attributes
    ----------
    shapiro_w : float
        Shapiro-Wilk test statistic
    shapiro_p : float
        Shapiro-Wilk p-value
    shapiro_fail : bool
        True if fails normality test (p < alpha)
    bp_stat : float
        Breusch-Pagan test statistic
    bp_p : float
        Breusch-Pagan p-value
    bp_fail : bool
        True if fails homoscedasticity test (p < alpha)
    dw_stat : float
        Durbin-Watson statistic
    dw_fail : bool
        True if autocorrelation detected (DW < 1.5 or > 2.5)
    any_fail : bool
        True if any diagnostic test fails
    """
    shapiro_w: float
    shapiro_p: float
    shapiro_fail: bool
    bp_stat: float
    bp_p: float
    bp_fail: bool
    dw_stat: float
    dw_fail: bool
    any_fail: bool


@dataclass
class ModelSelection:
    """
    Results from model selection.

    Attributes
    ----------
    best_model : str
        Name of selected model
    aicc : float
        AICc value of best model
    delta_i : float
        Delta_i (difference from best AICc)
    weight : float
        Akaike weight
    r_squared : float
        R-squared value
    adj_r_squared : float
        Adjusted R-squared
    competitive_models : list of str
        Models within delta_i <= 2
    model_coefficients : dict
        Coefficients of the best model
    """
    best_model: str
    aicc: float
    delta_i: float
    weight: float
    r_squared: float
    adj_r_squared: float
    competitive_models: List[str]
    model_coefficients: Dict[str, float]


def fit_candidate_models(
    prevalence_df: pl.DataFrame,
    skill: str,
    alpha: float = 0.10
) -> Dict[str, ModelResult]:
    """
    Fit all candidate models for a single skill.

    Fits five models to skill prevalence data:
    1. Null: prevalence ~ 1
    2. Linear: prevalence ~ year + job_text_length_std
    3. Log-year: prevalence ~ log(year - min_year + 1) + job_text_length_std
    4. Exponential: log(prevalence) ~ year + job_text_length_std
    5. Quadratic: prevalence ~ year + year^2 + job_text_length_std

    Parameters
    ----------
    prevalence_df : pl.DataFrame
        Skill prevalence data with columns: Year, prevalence, job_text_length_std
    skill : str
        Skill name to fit models for
    alpha : float, default 0.10
        Significance level for tests

    Returns
    -------
    dict
        Dictionary mapping model_name -> ModelResult

    Notes
    -----
    All models include job_text_length_std as a covariate to control for
    the correlation between job text length and number of skills.
    """
    # Filter to this skill
    skill_data = prevalence_df.filter(pl.col('Skill') == skill)

    if len(skill_data) < 5:
        raise ValueError(f"Insufficient data for {skill}: {len(skill_data)} observations")

    # Convert to pandas for statsmodels
    df = skill_data.to_pandas()

    # Standardize year for numerical stability
    df['year_std'] = (df['Year'] - df['Year'].mean()) / df['Year'].std()
    df['year_std_sq'] = df['year_std'] ** 2

    # Log-year transformation
    min_year = df['Year'].min()
    df['log_year'] = np.log(df['Year'] - min_year + 1)

    models = {}

    # Model 1: Null (intercept only)
    try:
        null_model = smf.ols('prevalence ~ 1', data=df).fit()
        models['null'] = ModelResult(
            model_name='null',
            fitted_model=null_model,
            formula='prevalence ~ 1',
            n_params=1,
            loglik=null_model.llf,
            n_obs=null_model.nobs,
            residuals=null_model.resid.values
        )
    except Exception as e:
        print(f"  Warning: Null model failed for {skill}: {e}")

    # Model 2: Linear
    try:
        linear_model = smf.ols('prevalence ~ year_std + job_text_length_std', data=df).fit()
        models['linear'] = ModelResult(
            model_name='linear',
            fitted_model=linear_model,
            formula='prevalence ~ year_std + job_text_length_std',
            n_params=3,
            loglik=linear_model.llf,
            n_obs=linear_model.nobs,
            residuals=linear_model.resid.values
        )
    except Exception as e:
        print(f"  Warning: Linear model failed for {skill}: {e}")

    # Model 3: Log-year
    try:
        log_model = smf.ols('prevalence ~ log_year + job_text_length_std', data=df).fit()
        models['log_year'] = ModelResult(
            model_name='log_year',
            fitted_model=log_model,
            formula='prevalence ~ log_year + job_text_length_std',
            n_params=3,
            loglik=log_model.llf,
            n_obs=log_model.nobs,
            residuals=log_model.resid.values
        )
    except Exception as e:
        print(f"  Warning: Log-year model failed for {skill}: {e}")

    # Model 4: Exponential (log-transformed)
    try:
        # Add small constant to avoid log(0)
        df['log_prevalence'] = np.log(df['prevalence'] + 1e-10)
        exp_model = smf.ols('log_prevalence ~ year_std + job_text_length_std', data=df).fit()
        models['exponential'] = ModelResult(
            model_name='exponential',
            fitted_model=exp_model,
            formula='log(prevalence) ~ year_std + job_text_length_std',
            n_params=3,
            loglik=exp_model.llf,
            n_obs=exp_model.nobs,
            residuals=exp_model.resid.values
        )
    except Exception as e:
        print(f"  Warning: Exponential model failed for {skill}: {e}")

    # Model 5: Quadratic
    try:
        quad_model = smf.ols('prevalence ~ year_std + year_std_sq + job_text_length_std', data=df).fit()
        models['quadratic'] = ModelResult(
            model_name='quadratic',
            fitted_model=quad_model,
            formula='prevalence ~ year_std + year_std_sq + job_text_length_std',
            n_params=4,
            loglik=quad_model.llf,
            n_obs=quad_model.nobs,
            residuals=quad_model.resid.values
        )
    except Exception as e:
        print(f"  Warning: Quadratic model failed for {skill}: {e}")

    return models


def calculate_aicc(model_result: ModelResult, c_hat: float = 1.0) -> float:
    """
    Calculate AICc (corrected AIC for small samples).

    For small samples (n/K < 40), AICc provides better model selection
    than standard AIC. Also supports QAICc for overdispersed data.

    Parameters
    ----------
    model_result : ModelResult
        Fitted model results
    c_hat : float, default 1.0
        Overdispersion parameter. Use c_hat > 1.0 for QAICc.

    Returns
    -------
    float
        AICc or QAICc value

    Notes
    -----
    AICc = -2 * log-likelihood + 2K + (2K(K+1))/(n-K-1)
    QAICc = AICc / c_hat for overdispersed data

    For exponential models, add Jacobian correction to log-likelihood.
    """
    n = model_result.n_obs
    k = model_result.n_params
    ll = model_result.loglik

    # Jacobian correction for log-transformed response
    if model_result.model_name == 'exponential':
        # Add sum(log(y)) to log-likelihood
        # This is already handled in statsmodels for log link functions
        pass

    # AICc formula
    aic = -2 * ll + 2 * k
    correction = (2 * k * (k + 1)) / (n - k - 1) if n - k - 1 > 0 else 0
    aicc = aic + correction

    # Apply overdispersion correction if needed
    if c_hat > 1.0:
        aicc = aicc / c_hat

    return aicc


def calculate_overdispersion(model_result: ModelResult) -> float:
    """
    Calculate overdispersion parameter (c-hat).

    c-hat = Pearson chi-square / degrees of freedom

    Parameters
    ----------
    model_result : ModelResult
        Fitted model results

    Returns
    -------
    float
        c-hat value (1.0 = no overdispersion, >1.0 = overdispersion)
    """
    residuals = model_result.residuals
    fitted = model_result.fitted_model.fittedvalues

    # Pearson residuals
    pearson_resid = residuals / np.sqrt(np.abs(fitted))
    chi_square = np.sum(pearson_resid ** 2)

    # Degrees of freedom
    df = model_result.n_obs - model_result.n_params

    c_hat = chi_square / df if df > 0 else 1.0

    return c_hat


def select_best_model(
    models: Dict[str, ModelResult],
    c_hat_threshold: float = 1.0,
    parsimony: bool = True,
    delta_i_threshold: float = 2.0
) -> ModelSelection:
    """
    Select best model using AICc and parsimony principle.

    Uses information-theoretic approach:
    1. Calculate AICc for all models
    2. Calculate delta_i (difference from best AICc)
    3. Calculate Akaike weights
    4. Apply parsimony: if delta_i <= 2, prefer simpler model

    Parameters
    ----------
    models : dict
        Dictionary of ModelResult objects
    c_hat_threshold : float, default 1.0
        If c_hat > threshold, use QAICc instead of AICc
    parsimony : bool, default True
        Apply parsimony principle for tied models
    delta_i_threshold : float, default 2.0
        Threshold for considering models competitive

    Returns
    -------
    ModelSelection
        Selected model and statistics

    Notes
    -----
    Parsimony principle: When delta_i < 2 between models, prefer the
    simpler model (fewer parameters). This avoids overfitting.
    """
    if not models:
        raise ValueError("No models to select from")

    # Calculate c-hat from best model (lowest AIC)
    temp_aics = {name: calculate_aicc(model, 1.0) for name, model in models.items()}
    best_temp = min(temp_aics, key=temp_aics.get)
    c_hat = calculate_overdispersion(models[best_temp])

    # Calculate AICc/QAICc for all models
    aiccs = {}
    for name, model in models.items():
        if c_hat > c_hat_threshold:
            aiccs[name] = calculate_aicc(model, c_hat)
        else:
            aiccs[name] = calculate_aicc(model, 1.0)

    # Find minimum AICc
    min_aicc = min(aiccs.values())

    # Calculate delta_i and weights
    delta_is = {name: aicc - min_aicc for name, aicc in aiccs.items()}
    sum_exp = sum(np.exp(-0.5 * delta) for delta in delta_is.values())
    weights = {name: np.exp(-0.5 * delta) / sum_exp for name, delta in delta_is.items()}

    # Identify competitive models (delta_i <= threshold)
    competitive = [name for name, delta in delta_is.items() if delta <= delta_i_threshold]

    # Select best model
    if parsimony and len(competitive) > 1:
        # Among competitive models, choose simplest
        best = min(competitive, key=lambda x: models[x].n_params)
    else:
        # Choose model with lowest AICc
        best = min(aiccs, key=aiccs.get)

    best_model = models[best]

    return ModelSelection(
        best_model=best,
        aicc=aiccs[best],
        delta_i=delta_is[best],
        weight=weights[best],
        r_squared=best_model.fitted_model.rsquared if hasattr(best_model.fitted_model, 'rsquared') else 0.0,
        adj_r_squared=best_model.fitted_model.rsquared_adj if hasattr(best_model.fitted_model, 'rsquared_adj') else 0.0,
        competitive_models=competitive,
        model_coefficients=best_model.fitted_model.params.to_dict()
    )


def run_diagnostics(model_result: ModelResult, alpha: float = 0.10) -> DiagnosticResult:
    """
    Run diagnostic tests on model residuals.

    Tests performed:
    1. Shapiro-Wilk: Normality of residuals
    2. Breusch-Pagan: Homoscedasticity
    3. Durbin-Watson: Autocorrelation

    Parameters
    ----------
    model_result : ModelResult
        Fitted model to test
    alpha : float, default 0.10
        Significance level

    Returns
    -------
    DiagnosticResult
        Test statistics and pass/fail indicators

    Notes
    -----
    With only 13 years of data (n=13), these tests have limited power.
    Failed tests should be interpreted as warning signs rather than
    definitive evidence of model inadequacy.
    """
    residuals = model_result.residuals

    # Shapiro-Wilk test (normality)
    try:
        shapiro_w, shapiro_p = shapiro(residuals)
        shapiro_fail = shapiro_p < alpha
    except Exception:
        shapiro_w, shapiro_p, shapiro_fail = np.nan, np.nan, True

    # Breusch-Pagan test (homoscedasticity)
    try:
        from statsmodels.stats.diagnostic import het_breuschpagan
        bp_stat, bp_p, _, _ = het_breuschpagan(
            residuals,
            model_result.fitted_model.model.exog
        )
        bp_fail = bp_p < alpha
    except Exception:
        bp_stat, bp_p, bp_fail = np.nan, np.nan, True

    # Durbin-Watson test (autocorrelation)
    try:
        from statsmodels.stats.stattools import durbin_watson
        dw_stat = durbin_watson(residuals)
        # DW between 1.5 and 2.5 is acceptable
        dw_fail = dw_stat < 1.5 or dw_stat > 2.5
    except Exception:
        dw_stat, dw_fail = np.nan, True

    any_fail = shapiro_fail or bp_fail or dw_fail

    return DiagnosticResult(
        shapiro_w=shapiro_w,
        shapiro_p=shapiro_p,
        shapiro_fail=shapiro_fail,
        bp_stat=bp_stat,
        bp_p=bp_p,
        bp_fail=bp_fail,
        dw_stat=dw_stat,
        dw_fail=dw_fail,
        any_fail=any_fail
    )


def assign_confidence_tier(
    model_selection: ModelSelection,
    diagnostics: DiagnosticResult,
    delta_i_threshold: float = 2.0,
    weight_threshold: float = 0.7
) -> int:
    """
    Assign confidence tier (1-4) based on evidence strength.

    Tier 1 (Strong Evidence):
        - DW test passes
        - Delta_i <= 2
        - Weight >= 0.7

    Tier 2 (Moderate Evidence):
        - Some diagnostics pass
        - Reasonable delta_i

    Tier 3 (Weak Evidence):
        - Multiple diagnostic failures
        - Higher delta_i

    Tier 4 (Exploratory):
        - Severe diagnostic violations
        - Should not be used for conclusions

    Parameters
    ----------
    model_selection : ModelSelection
        Selected model information
    diagnostics : DiagnosticResult
        Diagnostic test results
    delta_i_threshold : float, default 2.0
        Threshold for strong evidence
    weight_threshold : float, default 0.7
        Threshold for strong model support

    Returns
    -------
    int
        Confidence tier (1=strongest, 4=weakest)
    """
    # Tier 1: Strong evidence
    if (not diagnostics.dw_fail and
        model_selection.delta_i <= delta_i_threshold and
        model_selection.weight >= weight_threshold):
        return 1

    # Tier 2: Moderate evidence
    if (not diagnostics.dw_fail or
        (model_selection.delta_i <= delta_i_threshold and
         model_selection.weight >= 0.5)):
        return 2

    # Tier 3: Weak evidence
    if not diagnostics.any_fail or model_selection.delta_i <= 4.0:
        return 3

    # Tier 4: Exploratory only
    return 4


def classify_trajectory(
    model_selection: ModelSelection,
    model_result: ModelResult
) -> str:
    """
    Classify skill trajectory based on selected model.

    Trajectories:
    - Stable: Null model selected
    - Linear Growth/Decline: Linear model with positive/negative slope
    - Exponential Growth/Decline: Exponential model
    - Decelerating Growth: Log-year model with positive coefficient
    - Accelerating: Quadratic with positive second derivative
    - Non-monotonic: Quadratic with inflection point

    Parameters
    ----------
    model_selection : ModelSelection
        Selected model
    model_result : ModelResult
        Fitted model object

    Returns
    -------
    str
        Trajectory classification

    Examples
    --------
    >>> trajectory = classify_trajectory(selection, model)
    >>> print(trajectory)
    'Linear Growth'
    """
    model_name = model_selection.best_model
    fitted = model_result.fitted_model

    if model_name == 'null':
        return 'Stable'

    elif model_name == 'linear':
        slope = fitted.params.get('year_std', 0)
        if slope > 0:
            return 'Linear Growth'
        else:
            return 'Linear Decline'

    elif model_name == 'log_year':
        coef = fitted.params.get('log_year', 0)
        if coef > 0:
            return 'Decelerating Growth'
        else:
            return 'Decelerating Decline'

    elif model_name == 'exponential':
        slope = fitted.params.get('year_std', 0)
        if slope > 0:
            return 'Exponential Growth'
        else:
            return 'Exponential Decline'

    elif model_name == 'quadratic':
        linear_coef = fitted.params.get('year_std', 0)
        quad_coef = fitted.params.get('year_std_sq', 0)

        if quad_coef > 0:
            return 'Accelerating'
        else:
            # Check for inflection point
            if linear_coef * quad_coef < 0:
                return 'Non-monotonic'
            else:
                return 'Decelerating'

    return 'Unknown'


# Example usage
if __name__ == "__main__":
    print("Testing statistics_utils module")
    print("=" * 60)

    # Create synthetic data for testing
    import polars as pl

    years = list(range(2010, 2023))
    np.random.seed(42)

    # Linear growth skill
    linear_prevalence = [0.05 + 0.01 * (y - 2010) + np.random.normal(0, 0.005) for y in years]

    test_data = pl.DataFrame({
        'Skill': ['TestSkill'] * len(years),
        'Year': years,
        'prevalence': linear_prevalence,
        'job_text_length_std': np.random.normal(0, 1, len(years))
    })

    print("\n1. Fitting candidate models...")
    models = fit_candidate_models(test_data, 'TestSkill')
    print(f"   Fitted {len(models)} models")

    print("\n2. Selecting best model...")
    selection = select_best_model(models)
    print(f"   Best model: {selection.best_model}")
    print(f"   AICc: {selection.aicc:.2f}")
    print(f"   Weight: {selection.weight:.3f}")

    print("\n3. Running diagnostics...")
    diagnostics = run_diagnostics(models[selection.best_model])
    print(f"   Shapiro p-value: {diagnostics.shapiro_p:.3f}")
    print(f"   DW statistic: {diagnostics.dw_stat:.2f}")
    print(f"   Any failures: {diagnostics.any_fail}")

    print("\n4. Assigning confidence tier...")
    tier = assign_confidence_tier(selection, diagnostics)
    print(f"   Confidence tier: {tier}")

    print("\n5. Classifying trajectory...")
    trajectory = classify_trajectory(selection, models[selection.best_model])
    print(f"   Trajectory: {trajectory}")

    print("\n" + "=" * 60)
    print("Module tests complete!")
