import numpy as np
import pandas as pd
from scipy import stats
import streamlit as st

def calculate_ab_test_results(df, variant_col, conversion_col, baseline_variant=None):
    """
    Calculate A/B test results.
    
    Parameters:
    df (DataFrame): Data with variant and conversion information
    variant_col (str): Column name for variant
    conversion_col (str): Column name for conversion
    baseline_variant (str): Variant to use as baseline (if None, use first variant)
    
    Returns:
    DataFrame: Results with statistics for each variant
    """
    if df is None or df.empty:
        st.error("No data available for A/B testing analysis.")
        return None
    
    # Ensure conversion column is numeric
    if df[conversion_col].dtype == 'bool':
        df = df.copy()
        df[conversion_col] = df[conversion_col].astype(int)
    
    # Get unique variants
    variants = df[variant_col].unique()
    
    # Set baseline variant if not provided
    if baseline_variant is None:
        baseline_variant = variants[0]
    
    # Initialize results
    results = []
    
    # Calculate statistics for each variant
    for variant in variants:
        # Filter data for this variant
        variant_data = df[df[variant_col] == variant]
        
        # Sample size
        sample_size = len(variant_data)
        
        # Conversions
        conversions = variant_data[conversion_col].sum()
        
        # Conversion rate
        conversion_rate = conversions / sample_size if sample_size > 0 else 0
        
        # Confidence interval (Wilson score interval for binomial proportion)
        if sample_size > 0:
            z = 1.96  # 95% confidence
            p = conversion_rate
            ci_width = z * np.sqrt(p * (1 - p) / sample_size) * 2  # Width of the CI
            ci_lower = max(0, p - ci_width/2)
            ci_upper = min(1, p + ci_width/2)
        else:
            ci_width = 0
            ci_lower = 0
            ci_upper = 0
        
        # Add to results
        results.append({
            'Variant': variant,
            'Sample Size': sample_size,
            'Conversions': conversions,
            'Conversion Rate': conversion_rate,
            'CI Lower': ci_lower,
            'CI Upper': ci_upper,
            'CI Width/2': ci_width/2,
            'Is Baseline': variant == baseline_variant
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate statistical significance compared to baseline
    baseline_data = df[df[variant_col] == baseline_variant]
    baseline_conversions = baseline_data[conversion_col].sum()
    baseline_size = len(baseline_data)
    
    for i, variant in enumerate(variants):
        if variant == baseline_variant:
            results_df.loc[i, 'p-value'] = 1.0
            results_df.loc[i, 'Significant'] = False
            results_df.loc[i, 'Lift'] = 0.0
            continue
        
        variant_data = df[df[variant_col] == variant]
        variant_conversions = variant_data[conversion_col].sum()
        variant_size = len(variant_data)
        
        # Calculate p-value using two-proportion z-test
        if baseline_size > 0 and variant_size > 0:
            # Create contingency table
            # [variant_conversions, variant_non_conversions, baseline_conversions, baseline_non_conversions]
            contingency = [
                variant_conversions, 
                variant_size - variant_conversions,
                baseline_conversions, 
                baseline_size - baseline_conversions
            ]
            
            # Perform chi-square test
            chi2, p_value, _, _ = stats.chi2_contingency([
                [contingency[0], contingency[1]],
                [contingency[2], contingency[3]]
            ])
            
            # Calculate relative lift
            baseline_rate = baseline_conversions / baseline_size if baseline_size > 0 else 0
            variant_rate = variant_conversions / variant_size if variant_size > 0 else 0
            
            if baseline_rate > 0:
                lift = (variant_rate - baseline_rate) / baseline_rate
            else:
                lift = float('inf') if variant_rate > 0 else 0
            
            results_df.loc[i, 'p-value'] = p_value
            results_df.loc[i, 'Significant'] = p_value < 0.05  # Using 5% significance level
            results_df.loc[i, 'Lift'] = lift
        else:
            results_df.loc[i, 'p-value'] = 1.0
            results_df.loc[i, 'Significant'] = False
            results_df.loc[i, 'Lift'] = 0.0
    
    # Format lift as percentage
    results_df['Lift'] = results_df['Lift'].apply(lambda x: f"{x*100:.2f}%" if x != float('inf') else "N/A")
    
    return results_df


def calculate_sample_size(baseline_rate, mde, power=0.8, alpha=0.05):
    """
    Calculate required sample size for A/B test.
    
    Parameters:
    baseline_rate (float): Baseline conversion rate
    mde (float): Minimum detectable effect (relative change)
    power (float): Statistical power (1 - beta)
    alpha (float): Significance level
    
    Returns:
    int: Required sample size per variant
    """
    # Z-scores
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    
    # Absolute effect size
    effect_size = baseline_rate * mde
    
    # Pooled probability
    p_pooled = baseline_rate + effect_size/2
    
    # Calculate sample size per group
    sample_size = (2 * p_pooled * (1 - p_pooled) * (z_alpha + z_beta)**2) / effect_size**2
    
    return int(np.ceil(sample_size))


def estimate_test_duration(sample_size, daily_users, experiment_allocation=0.5):
    """
    Estimate the duration of an A/B test.
    
    Parameters:
    sample_size (int): Required sample size per variant
    daily_users (int): Number of unique users per day
    experiment_allocation (float): Portion of users to include in the experiment
    
    Returns:
    int: Estimated days to complete the test
    """
    # Total sample size needed (for both control and variant)
    total_sample_size = sample_size * 2
    
    # Daily users in the experiment
    daily_experiment_users = daily_users * experiment_allocation
    
    # Estimated days
    days = total_sample_size / daily_experiment_users
    
    return int(np.ceil(days))
