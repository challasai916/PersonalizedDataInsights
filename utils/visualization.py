import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from io import BytesIO

def plot_data_overview(df):
    """Generate plots for data overview."""
    if df is None or df.empty:
        st.error("No data available for visualization.")
        return
    
    # Data types distribution
    dtypes = df.dtypes.value_counts().reset_index()
    dtypes.columns = ['Data Type', 'Count']
    
    fig = px.bar(dtypes, x='Data Type', y='Count', 
                 title='Distribution of Data Types',
                 color='Data Type')
    st.plotly_chart(fig, use_container_width=True)
    
    # Missing values
    missing = df.isna().sum().reset_index()
    missing.columns = ['Column', 'Missing Values']
    missing = missing[missing['Missing Values'] > 0].sort_values('Missing Values', ascending=False)
    
    if not missing.empty:
        fig = px.bar(missing, x='Column', y='Missing Values',
                     title='Missing Values by Column',
                     color='Missing Values')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("No missing values found in the dataset.")
    
    # Preview numeric distributions
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if numeric_cols:
        # Select up to 5 columns to avoid cluttering
        display_cols = numeric_cols[:5]
        
        fig = make_subplots(rows=len(display_cols), cols=1, 
                            subplot_titles=[f"Distribution of {col}" for col in display_cols],
                            vertical_spacing=0.05)
        
        for i, col in enumerate(display_cols):
            fig.add_trace(go.Histogram(x=df[col], name=col), row=i+1, col=1)
        
        fig.update_layout(height=250*len(display_cols), title_text="Numeric Columns Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # Preview categorical distributions
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    if cat_cols:
        # Select up to 5 columns to avoid cluttering
        display_cols = cat_cols[:5]
        
        for col in display_cols:
            value_counts = df[col].value_counts().reset_index()
            value_counts.columns = [col, 'Count']
            
            # Limit to top 10 categories if there are too many
            if len(value_counts) > 10:
                top_values = value_counts.head(9)
                other_values = pd.DataFrame({
                    col: ['Other'],
                    'Count': [value_counts['Count'][9:].sum()]
                })
                value_counts = pd.concat([top_values, other_values])
            
            fig = px.pie(value_counts, values='Count', names=col,
                         title=f'Distribution of {col}')
            st.plotly_chart(fig, use_container_width=True)


def plot_correlation_matrix(df):
    """Generate correlation matrix visualization."""
    if df is None or df.empty:
        st.error("No data available for correlation analysis.")
        return
    
    # Select numeric columns only
    numeric_df = df.select_dtypes(include=['number'])
    
    if numeric_df.shape[1] < 2:
        st.warning("Not enough numeric columns for correlation analysis.")
        return
    
    # Calculate correlation
    corr = numeric_df.corr()
    
    # Create mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Create a custom colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
    plt.title('Correlation Matrix')
    
    # Convert matplotlib figure to image
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    
    # Display the image
    st.image(buf, use_column_width=True)


def plot_user_behavior(df, behavior_cols):
    """Generate plots for user behavior analysis."""
    if df is None or df.empty or not behavior_cols:
        st.error("No data available for user behavior analysis.")
        return
    
    # Activity over time (if time columns exist)
    time_cols = [col for col in df.columns if 'time' in col.lower() 
                 or 'date' in col.lower() or col.lower().endswith('at')]
    
    if time_cols:
        time_col = time_cols[0]  # Take the first time column
        
        if pd.api.types.is_datetime64_any_dtype(df[time_col]):
            # Set the time column as index
            time_df = df.set_index(time_col)
            
            # Resample by day and count occurrences
            daily_activity = time_df.resample('D').size().reset_index()
            daily_activity.columns = [time_col, 'Count']
            
            fig = px.line(daily_activity, x=time_col, y='Count',
                          title='User Activity Over Time')
            st.plotly_chart(fig, use_container_width=True)
    
    # Behavioral metrics distribution
    for col in behavior_cols[:5]:  # Limit to 5 plots
        if df[col].dtype in ['int64', 'float64']:
            fig = px.histogram(df, x=col, title=f'Distribution of {col}')
            st.plotly_chart(fig, use_container_width=True)
    
    # User engagement scatter plot (if we have engagement metrics)
    engagement_cols = [col for col in behavior_cols 
                       if any(term in col.lower() for term in ['count', 'frequency', 'views', 'clicks'])]
    
    if len(engagement_cols) >= 2:
        x_col = engagement_cols[0]
        y_col = engagement_cols[1]
        
        fig = px.scatter(df, x=x_col, y=y_col, 
                          title=f'User Engagement: {x_col} vs {y_col}',
                          opacity=0.7)
        st.plotly_chart(fig, use_container_width=True)


def plot_recommendation_metrics(precision, recall, f1, users_covered):
    """Generate plots for recommendation model metrics."""
    # Create metrics dataframe
    metrics_df = pd.DataFrame({
        'Metric': ['Precision', 'Recall', 'F1 Score', 'User Coverage'],
        'Value': [precision, recall, f1, users_covered]
    })
    
    # Bar chart for metrics
    fig = px.bar(metrics_df, x='Metric', y='Value',
                 title='Recommendation Model Performance Metrics',
                 color='Metric')
    
    # Set y-axis range from 0 to 1
    fig.update_layout(yaxis_range=[0, 1])
    
    st.plotly_chart(fig, use_container_width=True)


def plot_segmentation_results(df, labels, feature_cols):
    """Generate plots for segmentation results."""
    if df is None or df.empty or labels is None:
        st.error("No data available for segmentation visualization.")
        return
    
    # Add segment labels to the dataframe
    df_with_labels = df.copy()
    df_with_labels['Segment'] = labels
    
    # Count number of users per segment
    segment_counts = df_with_labels['Segment'].value_counts().reset_index()
    segment_counts.columns = ['Segment', 'Count']
    
    # Bar chart for segment sizes
    fig = px.bar(segment_counts, x='Segment', y='Count',
                 title='Number of Users per Segment',
                 color='Segment')
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature means by segment
    feature_means = df_with_labels.groupby('Segment')[feature_cols].mean().reset_index()
    
    # Radar chart for segment profiles
    fig = go.Figure()
    
    for segment in feature_means['Segment'].unique():
        segment_data = feature_means[feature_means['Segment'] == segment]
        
        # Add a trace for each segment
        fig.add_trace(go.Scatterpolar(
            r=segment_data[feature_cols].values.flatten(),
            theta=feature_cols,
            fill='toself',
            name=f'Segment {segment}'
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
            )),
        showlegend=True,
        title='Segment Profiles by Feature'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Scatter plot for 2D visualization (if we have at least 2 features)
    if len(feature_cols) >= 2:
        x_col = feature_cols[0]
        y_col = feature_cols[1]
        
        fig = px.scatter(df_with_labels, x=x_col, y=y_col, 
                          color='Segment', title=f'Segments: {x_col} vs {y_col}')
        st.plotly_chart(fig, use_container_width=True)


def plot_ab_test_results(results):
    """Generate plots for A/B testing results."""
    if results is None:
        st.error("No A/B testing results available for visualization.")
        return
    
    # Bar chart for conversion rates
    fig = px.bar(results, x='Variant', y='Conversion Rate',
                 title='Conversion Rate by Variant',
                 color='Variant',
                 error_y='CI Width/2')
    
    # Set y-axis to start from 0
    fig.update_layout(yaxis_range=[0, max(results['Conversion Rate']) * 1.2])
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Sample size and conversions
    metrics_df = results[['Variant', 'Sample Size', 'Conversions']]
    metrics_df = pd.melt(metrics_df, id_vars=['Variant'], 
                         value_vars=['Sample Size', 'Conversions'],
                         var_name='Metric', value_name='Count')
    
    fig = px.bar(metrics_df, x='Variant', y='Count', color='Metric',
                 title='Sample Size and Conversions by Variant',
                 barmode='group')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add significance visualization
    if 'p-value' in results.columns and 'Significant' in results.columns:
        # Create a dataframe for p-values
        pvalue_df = results[['Variant', 'p-value', 'Significant']]
        
        fig = px.bar(pvalue_df, x='Variant', y='p-value',
                     title='Statistical Significance (p-value)',
                     color='Significant')
        
        # Add a horizontal line at p=0.05
        fig.add_shape(type='line', x0=-0.5, x1=len(pvalue_df)-0.5, y0=0.05, y1=0.05,
                      line=dict(color='red', width=2, dash='dash'))
        
        # Add annotation for the significance threshold
        fig.add_annotation(x=len(pvalue_df)-1, y=0.05,
                           text="Significance Threshold (p=0.05)",
                           showarrow=True, arrowhead=1, ax=50, ay=20)
        
        st.plotly_chart(fig, use_container_width=True)
