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
    original_y: Optional[np.ndarray] = None


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
        Delta_i of selected model from the global minimum (0 when best model
        is selected without parsimony; may be > 0 when parsimony promotes a
        simpler model over the raw AICc minimum)
    delta_i_second_best : float
        Gap between the best model's AICc and the second-best model's AICc
        (i.e. the smallest nonzero delta across all candidates). This is the
        Δᵢ used for confidence-tier assignment per B&A (2002). Zero when only
        one model was fitted.
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
    c_hat : float
        Overdispersion parameter estimated from the globally best model.
        1.0 when no overdispersion is detected (AICc used); >1.0 triggers
        QAICc per B&A (2002).
    using_qaic : bool
        True when c_hat > 1.0 and QAICc was used for model selection.
    """
    best_model: str
    aicc: float
    delta_i: float
    delta_i_second_best: float
    weight: float
    r_squared: float
    adj_r_squared: float
    competitive_models: List[str]
    model_coefficients: Dict[str, float]
    c_hat: float = 1.0
    using_qaic: bool = False


def fit_candidate_models(
    prevalence_df: pl.DataFrame,
    skill: str,
    alpha: float = 0.10
) -> Dict[str, ModelResult]:
    """
    Fit all candidate models for a single skill.

    Fits five models to skill prevalence data:
    1. Null: prevalence ~ job_text_length_std
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

    # Center year on global mean (B&A 2002: center only, do not scale)
    df['year_centered'] = df['Year'] - df['Year'].mean()
    df['year_centered_sq'] = df['year_centered'] ** 2

    # Log-year transformation
    min_year = df['Year'].min()
    df['log_year'] = np.log(df['Year'] - min_year + 1)

    models = {}

    # Model 1: Null (covariate only, no temporal trend)
    try:
        null_model = smf.ols('prevalence ~ job_text_length_std', data=df).fit()
        models['null'] = ModelResult(
            model_name='null',
            fitted_model=null_model,
            formula='prevalence ~ job_text_length_std',
            n_params=2,
            loglik=null_model.llf,
            n_obs=null_model.nobs,
            residuals=null_model.resid.values
        )
    except Exception as e:
        print(f"  Warning: Null model failed for {skill}: {e}")

    # Model 2: Linear
    try:
        linear_model = smf.ols('prevalence ~ year_centered + job_text_length_std', data=df).fit()
        models['linear'] = ModelResult(
            model_name='linear',
            fitted_model=linear_model,
            formula='prevalence ~ year_centered + job_text_length_std',
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
        exp_model = smf.ols('log_prevalence ~ year_centered + job_text_length_std', data=df).fit()
        models['exponential'] = ModelResult(
            model_name='exponential',
            fitted_model=exp_model,
            formula='log(prevalence) ~ year_centered + job_text_length_std',
            n_params=3,
            loglik=exp_model.llf,
            n_obs=exp_model.nobs,
            residuals=exp_model.resid.values,
            original_y=df['prevalence'].values
        )
    except Exception as e:
        print(f"  Warning: Exponential model failed for {skill}: {e}")

    # Model 5: Quadratic
    try:
        quad_model = smf.ols('prevalence ~ year_centered + year_centered_sq + job_text_length_std', data=df).fit()
        models['quadratic'] = ModelResult(
            model_name='quadratic',
            fitted_model=quad_model,
            formula='prevalence ~ year_centered + year_centered_sq + job_text_length_std',
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
    Standard AICc = -2*log-likelihood + 2K + (2K(K+1))/(n-K-1)

    QAICc per Burnham & Anderson (2002):
      QAIC  = (-2 * log-likelihood / ĉ) + 2*(K+1)
      QAICc = QAIC + (2*(K+1)*((K+1)+1)) / (n - (K+1) - 1)
    K is incremented by 1 to account for estimation of ĉ. The overdispersion
    correction applies only to the log-likelihood term, not to the penalty.

    For exponential models, add Jacobian correction to log-likelihood.
    """
    n = model_result.n_obs
    k = model_result.n_params
    ll = model_result.loglik

    # Jacobian correction for log-transformed response (B&A 2002).
    # The exponential model is fit on log(y), so its log-likelihood lives in
    # the transformed space. To make it comparable to models fit on y, apply
    # the change-of-variables correction: ll_y = ll_z - sum(log(y_i)).
    if model_result.model_name == 'exponential' and model_result.original_y is not None:
        ll = ll - np.sum(np.log(model_result.original_y))

    if c_hat > 1.0:
        # QAICc: overdispersion correction applied only to log-likelihood;
        # K incremented by 1 for estimation of ĉ (B&A 2002)
        k_q = k + 1
        qaic = (-2 * ll / c_hat) + 2 * k_q
        correction = (2 * k_q * (k_q + 1)) / (n - k_q - 1) if n - k_q - 1 > 0 else 0
        return qaic + correction
    else:
        # Standard AICc
        aic = -2 * ll + 2 * k
        correction = (2 * k * (k + 1)) / (n - k - 1) if n - k - 1 > 0 else 0
        return aic + correction


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
    parsimony: bool = True,
    delta_i_threshold: float = 2.0
) -> ModelSelection:
    """
    Select best model using AICc/QAICc and parsimony principle.

    Uses information-theoretic approach per Burnham & Anderson (2002):
    1. Estimate overdispersion (c-hat) from the globally best model
    2. Clamp c-hat to [1, inf): values below 1 are not meaningful and
       collapse to standard AICc
    3. If c-hat > 1.0, apply QAICc; otherwise use AICc
    4. If c-hat > 4.0, warn that structural lack of fit may be present
    5. Calculate delta_i and Akaike weights
    6. Apply parsimony: if delta_i <= 2, prefer simpler model

    Parameters
    ----------
    models : dict
        Dictionary of ModelResult objects
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

    B&A (2002) recommend ĉ ∈ [1, 4]. Values above 4 suggest structural
    misfit that cannot be corrected by overdispersion scaling alone.
    """
    if not models:
        raise ValueError("No models to select from")

    # Calculate c-hat from globally best model (lowest AICc at c_hat=1)
    temp_aics = {name: calculate_aicc(model, 1.0) for name, model in models.items()}
    best_temp = min(temp_aics, key=temp_aics.get)
    c_hat = calculate_overdispersion(models[best_temp])

    # Clamp c-hat per B&A (2002): values below 1 are not meaningful
    if c_hat < 1.0:
        c_hat = 1.0

    # Warn if c-hat exceeds B&A's practical upper bound
    if c_hat > 4.0:
        warnings.warn(
            f"c-hat = {c_hat:.2f} > 4.0 for model '{best_temp}'. "
            "Burnham & Anderson (2002) suggest this indicates structural lack "
            "of fit. QAICc will still be applied, but results should be "
            "interpreted with caution.",
            RuntimeWarning,
            stacklevel=2
        )

    # Calculate AICc (c_hat == 1.0) or QAICc (c_hat > 1.0) for all models
    aiccs = {name: calculate_aicc(model, c_hat) for name, model in models.items()}

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
        best = min(aiccs, key=lambda name: aiccs[name])

    best_model = models[best]

    # Gap between best and second-best model (used for confidence-tier assignment).
    # delta_is[best] is 0 when best is the raw AICc minimum; the second-best
    # nonzero delta tells us how much better the winner is than its nearest rival.
    other_deltas = [d for name, d in delta_is.items() if name != best]
    delta_i_second_best = min(other_deltas) if other_deltas else 0.0

    return ModelSelection(
        best_model=best,
        aicc=aiccs[best],
        delta_i=delta_is[best],
        delta_i_second_best=delta_i_second_best,
        weight=weights[best],
        r_squared=best_model.fitted_model.rsquared if hasattr(best_model.fitted_model, 'rsquared') else 0.0,
        adj_r_squared=best_model.fitted_model.rsquared_adj if hasattr(best_model.fitted_model, 'rsquared_adj') else 0.0,
        competitive_models=competitive,
        model_coefficients=best_model.fitted_model.params.to_dict(),
        c_hat=c_hat,
        using_qaic=c_hat > 1.0,
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


def assign_discriminability(model_selection: ModelSelection) -> str:
    """
    Classify model discriminability based on Δᵢ to the second-best model.

    Uses ``model_selection.delta_i_second_best`` — the AICc gap between the
    selected model and its nearest competitor — as the sole criterion.  This
    is pure B&A (2002) and answers the question: *how clearly can we pick a
    winner?*

    Decision table
    --------------
    Label               Δᵢ (second-best)
    ------------------- ----------------
    Strong              ≤ 2
    Moderate            2 < Δᵢ < 7
    Weak                7 ≤ Δᵢ < 10
    Indistinguishable   ≥ 10

    Parameters
    ----------
    model_selection : ModelSelection
        Selected model information, including ``delta_i_second_best``.

    Returns
    -------
    str
        One of 'Strong', 'Moderate', 'Weak', or 'Indistinguishable'.
    """
    delta = model_selection.delta_i_second_best
    if delta <= 2.0:
        return 'Strong'
    if delta < 7.0:
        return 'Moderate'
    if delta < 10.0:
        return 'Weak'
    return 'Indistinguishable'


def assign_assumption_quality(
    model_selection: ModelSelection,
    diagnostics: DiagnosticResult,
) -> str:
    """
    Classify the quality of model assumptions for the selected model.

    Answers the question: *how much do we trust the AICc values?*  Three
    mutually exclusive labels are assigned in precedence order.

    Decision table
    --------------
    Label        Condition
    ------------ ----------------------------------------------------------
    Compromised  DW failure  OR  ≥ 2 failures among Shapiro-Wilk + BP
    Corrected    Overdispersion detected and QAICc applied (no other fails)
    Clean        All diagnostics pass, no overdispersion

    Parameters
    ----------
    model_selection : ModelSelection
        Selected model information, including ``using_qaic``.
    diagnostics : DiagnosticResult
        Diagnostic test results for the selected model.

    Returns
    -------
    str
        One of 'Compromised', 'Corrected', or 'Clean'.
    """
    other_failures = int(diagnostics.shapiro_fail) + int(diagnostics.bp_fail)
    if diagnostics.dw_fail or other_failures >= 2:
        return 'Compromised'
    if model_selection.using_qaic:
        return 'Corrected'
    return 'Clean'


def classify_trajectory(
    model_selection: ModelSelection,
    model_result: ModelResult,
) -> str:
    """
    Classify skill trajectory based on selected model per B&A (2002).

    Trajectories:
    - Stable: Null model selected, or implied annual change < 1 percentage
      point per year for any model
    - Linear Growth/Decline: Linear model, |slope| >= 0.01 pp/year
    - Decelerating Growth/Decline: Log-year model, avg annual change >= 0.01
    - Rapidly Increasing/Decreasing: Exponential model, |β₁| > 0.05
    - Exponential Growth/Decline: Exponential model, 0.01 <= |β₁| <= 0.05
    - Accelerating: Quadratic with positive second derivative
    - Non-monotonic: Quadratic with sign change (inflection point)
    - Decelerating: Quadratic with negative second derivative, no sign change

    Skills with 'Indistinguishable' discriminability should not be passed
    to this function — their trajectory is not reported (set to None by the
    caller).

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

    Notes
    -----
    Annual change rate thresholds follow the paper's classification table:
    - Linear model: annual change rate = β₁ (prevalence units/year)
    - Log-year model: avg annual change = β₁ * log(n) / (n - 1), where n is
      the number of observations (years), giving the mean step over the study
      period
    - Exponential model: β₁ approximates the annual relative change rate in
      log space; |β₁| > 0.05 (~5%/year) is required for "Rapidly" labels

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
        slope = fitted.params.get('year_centered', 0)
        # Annual change rate is constant for a linear model
        if abs(slope) < 0.01:
            return 'Stable'
        return 'Linear Growth' if slope > 0 else 'Linear Decline'

    elif model_name == 'log_year':
        coef = fitted.params.get('log_year', 0)
        # Average annual change = total fitted change / (n_years - 1).
        # log_year goes from log(1)=0 to log(n), so total change = coef * log(n).
        n = model_result.n_obs
        avg_annual_change = coef * np.log(n) / (n - 1) if n > 1 else coef
        if abs(avg_annual_change) < 0.01:
            return 'Stable'
        return 'Decelerating Growth' if coef > 0 else 'Decelerating Decline'

    elif model_name == 'exponential':
        slope = fitted.params.get('year_centered', 0)
        # slope is in log-prevalence space; approximates annual relative change
        if abs(slope) < 0.01:
            return 'Stable'
        if abs(slope) > 0.05:
            return 'Rapidly Increasing' if slope > 0 else 'Rapidly Decreasing'
        return 'Exponential Growth' if slope > 0 else 'Exponential Decline'

    elif model_name == 'quadratic':
        linear_coef = fitted.params.get('year_centered', 0)
        quad_coef = fitted.params.get('year_centered_sq', 0)

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
    print(f"   Delta_i (2nd best): {selection.delta_i_second_best:.2f}")
    print(f"   c_hat: {selection.c_hat:.3f}")
    print(f"   Using QAICc: {selection.using_qaic}")

    print("\n3. Running diagnostics...")
    diagnostics = run_diagnostics(models[selection.best_model])
    print(f"   Shapiro p-value: {diagnostics.shapiro_p:.3f}")
    print(f"   DW statistic: {diagnostics.dw_stat:.2f}")
    print(f"   Any failures: {diagnostics.any_fail}")

    print("\n4. Assigning evidence quality dimensions...")
    discriminability = assign_discriminability(selection)
    assumption_quality = assign_assumption_quality(selection, diagnostics)
    print(f"   Discriminability: {discriminability}")
    print(f"   Assumption quality: {assumption_quality}")

    print("\n5. Classifying trajectory...")
    if discriminability == 'Indistinguishable':
        trajectory = None
        print("   Trajectory: None (not reported — indistinguishable models)")
    else:
        trajectory = classify_trajectory(selection, models[selection.best_model])
        print(f"   Trajectory: {trajectory}")

    print("\n" + "=" * 60)
    print("Module tests complete!")
