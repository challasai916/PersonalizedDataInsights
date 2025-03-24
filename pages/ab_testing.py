import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from models.ab_testing import calculate_ab_test_results, calculate_sample_size, estimate_test_duration
from utils.data_processor import prepare_for_ab_testing
from utils.visualization import plot_ab_test_results

def show():
    """Display the A/B testing analysis page."""
    st.title("A/B Testing Analysis")
    
    # Check if data exists in session state
    if st.session_state.data is None:
        st.warning("Please upload data first on the 'Data Upload' page.")
        return
    
    df = st.session_state.data
    
    st.markdown("""
    ## Analyze A/B Test Results
    
    A/B testing is essential for evaluating the impact of personalization strategies.
    This page helps you analyze test results to determine what works best for your users.
    """)
    
    # Tabs for A/B testing workflow
    tabs = st.tabs(["Data Preparation", "Test Analysis", "Test Planning"])
    
    # Data Preparation tab
    with tabs[0]:
        st.markdown("### Prepare A/B Test Data")
        
        st.write("""
        For A/B test analysis, we need:
        - Variant information (which version users saw)
        - Conversion data (whether users took the desired action)
        - Optional: other metrics of interest
        """)
        
        # Prepare the data for A/B testing
        with st.spinner("Preparing data for A/B testing analysis..."):
            ab_df, metadata = prepare_for_ab_testing(df)
        
        if ab_df is not None and metadata is not None:
            st.success("Data successfully prepared for A/B testing analysis.")
            
            # Display identified columns
            st.markdown("#### Identified Columns")
            st.write(f"Variant column: {metadata['variant_col']}")
            st.write(f"Conversion column: {metadata['conversion_col']}")
            if metadata.get('experiment_col'):
                st.write(f"Experiment column: {metadata['experiment_col']}")
            
            # Display variant information
            st.markdown("#### Variant Information")
            st.write(f"Variants detected: {', '.join(map(str, metadata['variant_values']))}")
            st.write(f"Baseline variant: {metadata['baseline_variant']}")
            
            # Display data preview
            st.markdown("#### Data Preview")
            st.dataframe(ab_df[[metadata['variant_col'], metadata['conversion_col']]].head())
            
            # Store the prepared data in session state
            st.session_state.ab_df = ab_df
            st.session_state.ab_metadata = metadata
        else:
            st.error("Could not prepare data for A/B testing analysis. Please ensure your data includes variant and conversion information.")
    
    # Test Analysis tab
    with tabs[1]:
        st.markdown("### Analyze A/B Test Results")
        
        if not hasattr(st.session_state, 'ab_df') or st.session_state.ab_df is None:
            st.warning("Please prepare your data in the 'Data Preparation' tab first.")
        else:
            # Get the data and metadata
            ab_df = st.session_state.ab_df
            metadata = st.session_state.ab_metadata
            
            # Test parameters
            st.markdown("#### Analysis Parameters")
            
            # Allow users to select the baseline variant
            baseline_variant = st.selectbox(
                "Select baseline variant:", 
                metadata['variant_values'],
                index=metadata['variant_values'].index(metadata['baseline_variant']) if metadata['baseline_variant'] in metadata['variant_values'] else 0
            )
            
            # Run analysis button
            if st.button("Run A/B Test Analysis"):
                with st.spinner("Analyzing A/B test results..."):
                    # Calculate results
                    results = calculate_ab_test_results(
                        ab_df,
                        metadata['variant_col'],
                        metadata['conversion_col'],
                        baseline_variant
                    )
                    
                    if results is not None:
                        # Store results in session state
                        st.session_state.ab_test_results = results
                        
                        # Display results
                        st.success("A/B test analysis completed!")
                        
                        st.markdown("#### A/B Test Results")
                        st.dataframe(results)
                        
                        # Visualize results
                        st.markdown("#### Results Visualization")
                        plot_ab_test_results(results)
                        
                        # Summary of findings
                        st.markdown("#### Summary of Findings")
                        
                        # Get non-baseline variants
                        test_variants = [v for v in results['Variant'] if v != baseline_variant]
                        
                        # Check if any variants are significantly better
                        significant_better = results[
                            (results['Variant'] != baseline_variant) & 
                            (results['Significant'] == True) & 
                            (results['Conversion Rate'] > results[results['Variant'] == baseline_variant]['Conversion Rate'].values[0])
                        ]
                        
                        if not significant_better.empty:
                            st.success(f"Variant(s) {', '.join(map(str, significant_better['Variant'].values))} performed significantly better than the baseline ({baseline_variant}).")
                            
                            # Best variant
                            best_variant = significant_better.loc[significant_better['Conversion Rate'].idxmax()]
                            st.write(f"The best performing variant is {best_variant['Variant']} with a conversion rate of {best_variant['Conversion Rate']:.4f} (Lift: {best_variant['Lift']}).")
                        else:
                            # Check if any are significantly worse
                            significant_worse = results[
                                (results['Variant'] != baseline_variant) & 
                                (results['Significant'] == True) & 
                                (results['Conversion Rate'] < results[results['Variant'] == baseline_variant]['Conversion Rate'].values[0])
                            ]
                            
                            if not significant_worse.empty:
                                st.error(f"Variant(s) {', '.join(map(str, significant_worse['Variant'].values))} performed significantly worse than the baseline ({baseline_variant}).")
                            else:
                                st.info("No variants showed statistically significant differences from the baseline.")
                    else:
                        st.error("Could not analyze A/B test results. Please check your data.")
            
            # Display existing results if available
            elif hasattr(st.session_state, 'ab_test_results') and st.session_state.ab_test_results is not None:
                st.success("A/B test analysis has already been completed.")
                
                # Display results
                st.markdown("#### A/B Test Results")
                st.dataframe(st.session_state.ab_test_results)
                
                # Visualize results
                st.markdown("#### Results Visualization")
                plot_ab_test_results(st.session_state.ab_test_results)
    
    # Test Planning tab
    with tabs[2]:
        st.markdown("### Plan Your Next A/B Test")
        
        st.write("""
        Planning is crucial for successful A/B tests. This section helps you:
        - Calculate the required sample size
        - Estimate test duration
        - Determine statistical significance thresholds
        """)
        
        # Test planning parameters
        st.markdown("#### Test Planning Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            baseline_rate = st.slider("Baseline conversion rate", min_value=0.001, max_value=0.5, value=0.1, step=0.001, format="%.3f")
            mde = st.slider("Minimum detectable effect", min_value=0.01, max_value=0.5, value=0.1, step=0.01, format="%.2f")
        
        with col2:
            power = st.slider("Statistical power", min_value=0.7, max_value=0.95, value=0.8, step=0.05, format="%.2f")
            alpha = st.slider("Significance level (α)", min_value=0.01, max_value=0.1, value=0.05, step=0.01, format="%.2f")
        
        # Calculate sample size
        sample_size = calculate_sample_size(baseline_rate, mde, power, alpha)
        
        st.markdown("#### Sample Size Calculation")
        st.metric("Required sample size per variant", f"{sample_size:,}")
        st.write(f"For a baseline conversion rate of {baseline_rate:.1%}, to detect a minimum effect of {mde:.1%}, "
                 f"with {power:.0%} power and {alpha:.0%} significance level, you need **{sample_size:,} users per variant**.")
        
        # Estimate test duration
        st.markdown("#### Test Duration Estimation")
        
        daily_users = st.number_input("Daily active users", min_value=10, value=1000, step=100)
        experiment_allocation = st.slider("Percentage of users in experiment", min_value=0.1, max_value=1.0, value=0.5, step=0.1, format="%.1f")
        
        duration = estimate_test_duration(sample_size, daily_users, experiment_allocation)
        
        st.metric("Estimated test duration", f"{duration} days")
        st.write(f"With {daily_users:,} daily users and {experiment_allocation:.0%} allocation to the experiment, "
                 f"your test will take approximately **{duration} days** to complete.")
        
        # Additional test planning guidance
        with st.expander("A/B Testing Best Practices"):
            st.markdown("""
            ### A/B Testing Best Practices
            
            1. **Define clear objectives**
               - What exactly are you testing?
               - What is your success metric?
            
            2. **Test one variable at a time**
               - Focus on testing a single change to isolate its impact
               - Multivariate testing requires much larger sample sizes
            
            3. **Run tests long enough**
               - Ensure statistical significance
               - Account for time-based variations (day of week, seasonality)
            
            4. **Monitor for unexpected impacts**
               - Check secondary metrics
               - Look for segment-specific effects
            
            5. **Document and share learnings**
               - Record methodology and results
               - Build an organizational knowledge base
            """)
