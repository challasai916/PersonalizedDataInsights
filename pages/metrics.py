import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def show():
    """Display the performance metrics page."""
    st.title("Performance Metrics")
    
    # Check if data exists in session state
    if st.session_state.data is None:
        st.warning("Please upload data first on the 'Data Upload' page.")
        return
    
    st.markdown("""
    ## Track Personalization Performance
    
    This dashboard helps you track and visualize the performance of your personalization efforts.
    Compare metrics across different models and approaches to identify what works best.
    """)
    
    # Check which models/analyses have been run
    has_recommendation = st.session_state.recommendation_model is not None
    has_segmentation = hasattr(st.session_state, 'segmentation_model') and st.session_state.segmentation_model is not None
    has_ab_test = hasattr(st.session_state, 'ab_test_results') and st.session_state.ab_test_results is not None
    
    # Create a metrics overview
    st.markdown("### Personalization Performance Overview")
    
    # Create a metrics dashboard
    metrics_dict = {}
    
    # Recommendation metrics
    if has_recommendation:
        metrics_dict["Recommendation System"] = {}
        
        if hasattr(st.session_state, 'recommendation_metrics'):
            metrics = st.session_state.recommendation_metrics
            metrics_dict["Recommendation System"]["Precision"] = metrics['precision']
            metrics_dict["Recommendation System"]["Recall"] = metrics['recall']
            metrics_dict["Recommendation System"]["F1 Score"] = metrics['f1']
            metrics_dict["Recommendation System"]["User Coverage"] = metrics['users_covered']
    
    # Segmentation metrics
    if has_segmentation:
        metrics_dict["User Segmentation"] = {}
        
        model = st.session_state.segmentation_model
        metrics_dict["User Segmentation"]["Number of Segments"] = model['n_clusters']
        
        # Calculate segment balance (entropy)
        labels = model['labels']
        segment_counts = pd.Series(labels).value_counts()
        proportions = segment_counts / len(labels)
        entropy = -sum(p * np.log2(p) for p in proportions)
        max_entropy = np.log2(len(segment_counts))
        balance = entropy / max_entropy if max_entropy > 0 else 0
        
        metrics_dict["User Segmentation"]["Segment Balance"] = balance
        
        # Get silhouette score if available
        if model['silhouette_scores'] and len(model['silhouette_scores']) > 0:
            silhouette = model['silhouette_scores'].get(model['n_clusters'], 0)
            metrics_dict["User Segmentation"]["Silhouette Score"] = silhouette
    
    # A/B testing metrics
    if has_ab_test:
        metrics_dict["A/B Testing"] = {}
        
        results = st.session_state.ab_test_results
        
        # Count significant improvements
        significant_improvements = results[
            (results['Significant'] == True) & 
            (results['Lift'].str.strip('%').astype(float) > 0)
        ]
        
        metrics_dict["A/B Testing"]["Significant Variants"] = len(significant_improvements)
        
        # Get best improvement
        if not significant_improvements.empty:
            # Extract numeric lift value (remove % and convert to float)
            significant_improvements['Numeric Lift'] = significant_improvements['Lift'].str.strip('%').astype(float) / 100
            
            best_lift = significant_improvements['Numeric Lift'].max()
            metrics_dict["A/B Testing"]["Best Improvement"] = best_lift
    
    # Display metrics dashboard
    if metrics_dict:
        # Determine number of columns based on number of models
        n_models = len(metrics_dict)
        cols = st.columns(min(n_models, 3))
        
        # Display metrics for each model
        for i, (model_name, metrics) in enumerate(metrics_dict.items()):
            col_idx = i % len(cols)
            
            with cols[col_idx]:
                st.markdown(f"#### {model_name}")
                
                for metric_name, value in metrics.items():
                    if isinstance(value, float):
                        # Format percentage metrics
                        if "Coverage" in metric_name or "Balance" in metric_name or "Improvement" in metric_name:
                            formatted_value = f"{value:.2%}"
                        else:
                            formatted_value = f"{value:.4f}"
                    else:
                        formatted_value = str(value)
                    
                    st.metric(metric_name, formatted_value)
        
        # Visualize all metrics in a single chart
        st.markdown("### Metrics Visualization")
        
        # Prepare data for visualization
        viz_data = []
        
        for model_name, metrics in metrics_dict.items():
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    viz_data.append({
                        'Model': model_name,
                        'Metric': metric_name,
                        'Value': value
                    })
        
        if viz_data:
            viz_df = pd.DataFrame(viz_data)
            
            # Create a grouped bar chart
            fig = px.bar(
                viz_df, x='Metric', y='Value', color='Model',
                title='Personalization Performance Metrics',
                barmode='group'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No metrics available yet. Please run models or analyses to generate metrics.")
    
    # Historical metrics tracking
    st.markdown("### Historical Performance Tracking")
    
    st.write("""
    Track how your personalization metrics change over time as you refine your approach.
    This helps you understand the impact of your improvements.
    """)
    
    # Add metrics to history
    if metrics_dict and st.button("Add Current Metrics to History"):
        if 'metrics_history' not in st.session_state:
            st.session_state.metrics_history = []
        
        # Add timestamp
        current_metrics = {
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
            'metrics': metrics_dict
        }
        
        st.session_state.metrics_history.append(current_metrics)
        st.success("Current metrics added to history!")
    
    # Display metrics history
    if hasattr(st.session_state, 'metrics_history') and st.session_state.metrics_history:
        history = st.session_state.metrics_history
        
        # Display as a table
        st.markdown("#### Metrics History")
        
        # Prepare history data for display
        history_tables = {}
        
        for entry in history:
            timestamp = entry['timestamp']
            
            for model_name, metrics in entry['metrics'].items():
                if model_name not in history_tables:
                    history_tables[model_name] = {}
                
                for metric_name, value in metrics.items():
                    if metric_name not in history_tables[model_name]:
                        history_tables[model_name][metric_name] = {}
                    
                    history_tables[model_name][metric_name][timestamp] = value
        
        # Display history for each model
        for model_name, metrics in history_tables.items():
            with st.expander(f"{model_name} - Historical Metrics"):
                # Convert to DataFrame
                model_df = pd.DataFrame(metrics)
                
                # Display table
                st.dataframe(model_df)
                
                # Plot metrics over time
                for metric_name, values in metrics.items():
                    # Convert to series
                    metric_series = pd.Series(values)
                    metric_series.index = pd.to_datetime(metric_series.index)
                    metric_series = metric_series.sort_index()
                    
                    # Plot
                    fig = px.line(
                        x=metric_series.index, y=metric_series.values,
                        title=f'{metric_name} Over Time',
                        labels={'x': 'Time', 'y': metric_name}
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    # Personalization success benchmarks
    with st.expander("Personalization Success Benchmarks"):
        st.markdown("""
        ### Personalization Success Benchmarks
        
        These industry benchmarks can help you gauge the success of your personalization efforts:
        
        #### Recommendation Systems
        - **Good Precision@10:** > 0.30
        - **Good Recall@10:** > 0.20
        - **Good F1 Score:** > 0.25
        - **User Coverage:** > 95%
        
        #### User Segmentation
        - **Optimal Number of Segments:** 3-7 for most applications
        - **Good Segment Balance:** > 0.80
        - **Good Silhouette Score:** > 0.50
        
        #### A/B Testing
        - **Meaningful Lift:** > 5%
        - **Significant p-value:** < 0.05
        
        #### Overall Impact
        - **Engagement Increase:** 10-30%
        - **Conversion Rate Increase:** 5-15%
        - **User Retention Improvement:** 5-25%
        """)
