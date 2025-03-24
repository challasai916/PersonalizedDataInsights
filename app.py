import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os
import json
from datetime import datetime
import time

# Import custom modules
import data_processing as dp
import visualization as viz
import recommendation as rec
import utils

# Page configuration
st.set_page_config(
    page_title="Personalization Data Science Platform",
    page_icon="📊",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'user_segments' not in st.session_state:
    st.session_state.user_segments = None
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'experiment_results' not in st.session_state:
    st.session_state.experiment_results = None

# Main app header
st.title("Personalization Data Science Platform")
st.markdown("""
This platform helps you analyze user behavior data and implement personalization
features for your online platform. Upload your data, explore patterns, segment users,
generate recommendations, and evaluate your personalization strategies.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Data Upload & Processing", "User Behavior Analysis", 
     "User Segmentation", "Recommendation Engine", 
     "A/B Testing & Evaluation", "Documentation"]
)

# Data Upload & Processing Page
if page == "Data Upload & Processing":
    st.header("Data Upload & Processing")
    
    # Data upload section
    st.subheader("Upload Your Data")
    data_file = st.file_uploader("Upload a CSV or JSON file with user behavior data", 
                                type=["csv", "json"])
    
    col1, col2 = st.columns(2)
    with col1:
        if data_file is not None:
            try:
                # Load and display the uploaded data
                if data_file.name.endswith('.csv'):
                    data = pd.read_csv(data_file)
                    st.session_state.data = data
                elif data_file.name.endswith('.json'):
                    data = pd.read_json(data_file)
                    st.session_state.data = data
                
                st.success(f"Successfully loaded data with {len(data)} rows and {len(data.columns)} columns.")
                
                # Display sample data
                st.subheader("Data Preview")
                st.dataframe(data.head(10))
                
                # Display data info
                st.subheader("Data Information")
                buffer = utils.get_data_info(data)
                st.text(buffer)
                
            except Exception as e:
                st.error(f"Error loading data: {e}")
    
    with col2:
        if st.session_state.data is not None:
            # Data processing options
            st.subheader("Data Processing Options")
            
            processing_options = st.multiselect(
                "Select processing steps",
                ["Handle missing values", "Normalize numeric features", 
                 "Encode categorical features", "Feature selection"]
            )
            
            if st.button("Process Data"):
                try:
                    data_to_process = st.session_state.data.copy()
                    
                    with st.spinner("Processing data..."):
                        # Apply selected processing steps
                        if "Handle missing values" in processing_options:
                            data_to_process = dp.handle_missing_values(data_to_process)
                            
                        if "Normalize numeric features" in processing_options:
                            data_to_process = dp.normalize_numeric_features(data_to_process)
                            
                        if "Encode categorical features" in processing_options:
                            data_to_process = dp.encode_categorical_features(data_to_process)
                            
                        if "Feature selection" in processing_options:
                            data_to_process = dp.select_important_features(data_to_process)
                        
                        st.session_state.processed_data = data_to_process
                        st.success("Data processing completed!")
                        
                        # Show processed data
                        st.subheader("Processed Data Preview")
                        st.dataframe(data_to_process.head(10))
                        
                except Exception as e:
                    st.error(f"Error during data processing: {e}")

    # Sample dataset section
    st.subheader("No data? Try a sample dataset")
    sample_dataset = st.selectbox(
        "Select a sample dataset",
        ["E-commerce User Behavior", "Content Platform Engagement", "None"]
    )
    
    if sample_dataset != "None" and st.button("Load Sample Dataset"):
        with st.spinner("Loading sample dataset..."):
            if sample_dataset == "E-commerce User Behavior":
                st.session_state.data = utils.load_ecommerce_sample_data()
            elif sample_dataset == "Content Platform Engagement":
                st.session_state.data = utils.load_content_platform_sample_data()
            
            st.success(f"Loaded {sample_dataset} sample with {len(st.session_state.data)} rows")
            st.dataframe(st.session_state.data.head(10))
            st.info("Note: This is a generated dataset for demonstration purposes. For real applications, use your own data.")

# User Behavior Analysis Page
elif page == "User Behavior Analysis":
    st.header("User Behavior Analysis")
    
    if st.session_state.data is None:
        st.warning("Please upload data first in the 'Data Upload & Processing' page.")
    else:
        data = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
        
        # Feature selection for analysis
        st.subheader("Select Features for Analysis")
        
        # Identify potential numeric and categorical columns
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        analysis_type = st.selectbox(
            "Select Analysis Type",
            ["Univariate Analysis", "Bivariate Analysis", "User Engagement Metrics", 
             "Temporal Patterns", "Behavioral Clusters"]
        )
        
        if analysis_type == "Univariate Analysis":
            col1, col2 = st.columns(2)
            
            with col1:
                feature = st.selectbox("Select a feature to analyze", data.columns)
                
                if feature in numeric_cols:
                    st.subheader(f"Distribution of {feature}")
                    fig = viz.plot_numeric_distribution(data, feature)
                    st.pyplot(fig)
                    
                    st.subheader(f"Statistics for {feature}")
                    st.write(data[feature].describe())
                
                elif feature in categorical_cols:
                    st.subheader(f"Distribution of {feature}")
                    fig = viz.plot_categorical_distribution(data, feature)
                    st.pyplot(fig)
                    
                    st.subheader(f"Value Counts for {feature}")
                    st.write(data[feature].value_counts())
            
            with col2:
                if feature in numeric_cols:
                    st.subheader(f"Box Plot for {feature}")
                    fig = viz.plot_boxplot(data, feature)
                    st.pyplot(fig)
                    
                    # Check for outliers
                    outliers_summary = viz.identify_outliers(data, feature)
                    st.write(outliers_summary)
                
                elif feature in categorical_cols:
                    st.subheader(f"Bar Chart for {feature}")
                    value_counts_df = data[feature].value_counts().reset_index()
                    value_counts_df.columns = [feature, 'count']
                    fig = px.bar(value_counts_df, x=feature, y='count')
                    st.plotly_chart(fig)
        
        elif analysis_type == "Bivariate Analysis":
            col1, col2 = st.columns(2)
            
            with col1:
                feature1 = st.selectbox("Select first feature", data.columns)
                feature2 = st.selectbox("Select second feature", 
                                      [col for col in data.columns if col != feature1])
                
                if feature1 in numeric_cols and feature2 in numeric_cols:
                    st.subheader(f"Scatter Plot: {feature1} vs {feature2}")
                    fig = px.scatter(data, x=feature1, y=feature2, 
                                   opacity=0.6, title=f"{feature1} vs {feature2}")
                    st.plotly_chart(fig)
                    
                    # Correlation
                    corr = data[[feature1, feature2]].corr().iloc[0, 1]
                    st.write(f"Correlation coefficient: {corr:.4f}")
                
                elif feature1 in categorical_cols and feature2 in numeric_cols:
                    st.subheader(f"{feature2} by {feature1}")
                    fig = viz.plot_numeric_by_categorical(data, feature2, feature1)
                    st.pyplot(fig)
                
                elif feature1 in numeric_cols and feature2 in categorical_cols:
                    st.subheader(f"{feature1} by {feature2}")
                    fig = viz.plot_numeric_by_categorical(data, feature1, feature2)
                    st.pyplot(fig)
                
                else:  # Both categorical
                    st.subheader(f"Heatmap: {feature1} vs {feature2}")
                    fig = viz.plot_categorical_relationship(data, feature1, feature2)
                    st.pyplot(fig)
            
            with col2:
                if feature1 in numeric_cols and feature2 in numeric_cols:
                    st.subheader("Hex Bin Density Plot")
                    fig = px.density_heatmap(data, x=feature1, y=feature2, 
                                          title=f"Density: {feature1} vs {feature2}")
                    st.plotly_chart(fig)
                
                elif (feature1 in categorical_cols and feature2 in numeric_cols) or \
                     (feature1 in numeric_cols and feature2 in categorical_cols):
                    num_feature = feature2 if feature2 in numeric_cols else feature1
                    cat_feature = feature1 if feature1 in categorical_cols else feature2
                    
                    st.subheader(f"Violin Plot: {num_feature} by {cat_feature}")
                    fig = px.violin(data, y=num_feature, x=cat_feature, 
                                  box=True, title=f"{num_feature} Distribution by {cat_feature}")
                    st.plotly_chart(fig)
                
                else:  # Both categorical
                    st.subheader("Stacked Bar Chart")
                    pivot = pd.crosstab(data[feature1], data[feature2], normalize='index')
                    fig = px.bar(pivot, barmode='stack', title=f"{feature1} vs {feature2}")
                    st.plotly_chart(fig)
        
        elif analysis_type == "User Engagement Metrics":
            # Identify user ID column if it exists
            potential_id_cols = [col for col in data.columns if 'id' in col.lower() or 'user' in col.lower()]
            
            if potential_id_cols:
                user_id_col = st.selectbox(
                    "Select the user ID column",
                    potential_id_cols
                )
                
                # Identify potential engagement metrics
                engagement_metrics = st.multiselect(
                    "Select engagement metrics to analyze",
                    [col for col in numeric_cols if col != user_id_col],
                    default=[numeric_cols[0]] if numeric_cols and numeric_cols[0] != user_id_col else []
                )
                
                if engagement_metrics and st.button("Generate Engagement Analysis"):
                    with st.spinner("Analyzing user engagement..."):
                        engagement_df = viz.calculate_engagement_metrics(data, user_id_col, engagement_metrics)
                        
                        st.subheader("Top Users by Engagement")
                        st.dataframe(engagement_df.head(10))
                        
                        # Engagement distribution
                        st.subheader("Engagement Score Distribution")
                        fig = px.histogram(engagement_df, x='engagement_score', 
                                         nbins=50, title="Distribution of User Engagement Scores")
                        st.plotly_chart(fig)
                        
                        # Correlation between metrics
                        if len(engagement_metrics) > 1:
                            st.subheader("Correlation Between Engagement Metrics")
                            corr_matrix = engagement_df[engagement_metrics].corr()
                            fig = px.imshow(corr_matrix, text_auto=True, 
                                         color_continuous_scale='RdBu_r', 
                                         title="Correlation Matrix")
                            st.plotly_chart(fig)
            else:
                st.warning("No user ID column detected. Please identify your user ID column in the data.")
        
        elif analysis_type == "Temporal Patterns":
            # Identify potential timestamp columns
            date_cols = [col for col in data.columns if 'date' in col.lower() or 'time' in col.lower()]
            
            if date_cols:
                time_col = st.selectbox("Select timestamp column", date_cols)
                
                # Convert to datetime if not already
                if data[time_col].dtype != 'datetime64[ns]':
                    try:
                        data[time_col] = pd.to_datetime(data[time_col])
                    except Exception as e:
                        st.error(f"Error converting column to datetime: {e}")
                
                # Select metrics to track over time
                temporal_metrics = st.multiselect(
                    "Select metrics to analyze over time",
                    [col for col in numeric_cols if col != time_col]
                )
                
                if temporal_metrics and st.button("Analyze Temporal Patterns"):
                    with st.spinner("Analyzing temporal patterns..."):
                        # Define time aggregation level
                        agg_level = st.selectbox(
                            "Select time aggregation level",
                            ["Day", "Week", "Month"]
                        )
                        
                        # Generate time series visualizations
                        st.subheader(f"Temporal Patterns by {agg_level}")
                        for metric in temporal_metrics:
                            fig = viz.plot_time_series(data, time_col, metric, agg_level.lower())
                            st.plotly_chart(fig)
                        
                        # Heatmap for day of week/hour patterns if data has sufficient granularity
                        if data[time_col].dt.hour.nunique() > 1:
                            st.subheader("Activity Heatmap by Day and Hour")
                            fig = viz.plot_time_heatmap(data, time_col)
                            st.plotly_chart(fig)
            else:
                st.warning("No timestamp column detected. For temporal analysis, your data should include a date/time column.")
        
        elif analysis_type == "Behavioral Clusters":
            st.subheader("User Behavioral Clustering")
            
            # Select features for clustering
            cluster_features = st.multiselect(
                "Select features to use for clustering",
                numeric_cols,
                default=numeric_cols[:min(5, len(numeric_cols))]
            )
            
            if cluster_features and st.button("Generate Behavioral Clusters"):
                with st.spinner("Generating clusters based on user behavior..."):
                    # Number of clusters
                    n_clusters = st.slider("Number of clusters", 2, 10, 5)
                    
                    # Perform clustering
                    cluster_data, cluster_centers, silhouette_avg = viz.perform_clustering(
                        data, cluster_features, n_clusters)
                    
                    # Display silhouette score
                    st.write(f"Silhouette Score: {silhouette_avg:.4f}")
                    
                    # Visualize clusters
                    if len(cluster_features) >= 2:
                        st.subheader("2D Cluster Visualization")
                        fig = viz.plot_clusters_2d(cluster_data, cluster_features)
                        st.plotly_chart(fig)
                    
                    # Show cluster characteristics
                    st.subheader("Cluster Characteristics")
                    cluster_profile = viz.get_cluster_profiles(cluster_data, cluster_features)
                    st.dataframe(cluster_profile)
                    
                    # Radar chart for cluster profiles
                    st.subheader("Cluster Profiles Comparison")
                    fig = viz.plot_cluster_radar(cluster_centers, cluster_features)
                    st.plotly_chart(fig)
                    
                    # Store the user segments for use in recommendation engine
                    st.session_state.user_segments = cluster_data

# User Segmentation Page
elif page == "User Segmentation":
    st.header("User Segmentation")
    
    if st.session_state.data is None:
        st.warning("Please upload data first in the 'Data Upload & Processing' page.")
    else:
        data = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
        
        # Segmentation method selection
        segmentation_method = st.selectbox(
            "Choose Segmentation Method",
            ["Clustering", "Rule-based", "RFM Analysis", "Custom"]
        )
        
        if segmentation_method == "Clustering":
            # Similar to Behavioral Clusters in the Analysis page
            # Select features for clustering
            numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            cluster_features = st.multiselect(
                "Select features to use for clustering",
                numeric_cols,
                default=numeric_cols[:min(5, len(numeric_cols))]
            )
            
            n_clusters = st.slider("Number of segments", 2, 10, 4)
            
            if cluster_features and st.button("Generate Customer Segments"):
                with st.spinner("Generating user segments..."):
                    # Perform clustering
                    cluster_data, cluster_centers, silhouette_avg = viz.perform_clustering(
                        data, cluster_features, n_clusters)
                    
                    # Store in session state
                    st.session_state.user_segments = cluster_data
                    
                    # Display segmentation results
                    st.success(f"Successfully created {n_clusters} user segments!")
                    
                    # Visualize segments
                    st.subheader("User Segment Distribution")
                    fig = px.pie(cluster_data, names='cluster', 
                               title="Distribution of Users Across Segments")
                    st.plotly_chart(fig)
                    
                    # Show segment characteristics
                    st.subheader("Segment Characteristics")
                    segment_profile = viz.get_cluster_profiles(cluster_data, cluster_features)
                    st.dataframe(segment_profile)
                    
                    # Segment comparison
                    st.subheader("Segment Comparison")
                    fig = viz.plot_segment_comparison(cluster_data, cluster_features)
                    st.plotly_chart(fig)
                    
                    # Radar chart for segment profiles
                    st.subheader("Segment Profiles")
                    fig = viz.plot_cluster_radar(cluster_centers, cluster_features)
                    st.plotly_chart(fig)
        
        elif segmentation_method == "Rule-based":
            st.subheader("Rule-based Segmentation")
            
            # Allow user to define segmentation rules
            st.write("Define rules to segment your users:")
            
            # Rule builder
            col1, col2, col3 = st.columns(3)
            
            with col1:
                rule_field = st.selectbox("Select Field", data.columns)
            
            with col2:
                operators = ["==", ">", "<", ">=", "<=", "!=", "contains"]
                rule_operator = st.selectbox("Operator", operators)
            
            with col3:
                if data[rule_field].dtype in ['int64', 'float64']:
                    rule_value = st.number_input("Value", value=0)
                else:
                    # For categorical fields
                    unique_values = data[rule_field].unique().tolist()
                    rule_value = st.selectbox("Value", unique_values)
            
            # Add rules to a list
            if 'rules' not in st.session_state:
                st.session_state.rules = []
            
            if st.button("Add Rule"):
                new_rule = {
                    "field": rule_field,
                    "operator": rule_operator,
                    "value": rule_value
                }
                st.session_state.rules.append(new_rule)
                st.success("Rule added!")
            
            # Display existing rules
            if st.session_state.rules:
                st.subheader("Current Rules")
                for i, rule in enumerate(st.session_state.rules):
                    st.write(f"{i+1}. {rule['field']} {rule['operator']} {rule['value']}")
                
                # Option to clear rules
                if st.button("Clear Rules"):
                    st.session_state.rules = []
                    st.success("Rules cleared!")
                
                # Generate segments based on rules
                if st.button("Apply Rules and Generate Segments"):
                    with st.spinner("Applying segmentation rules..."):
                        segmented_data = dp.apply_rule_based_segmentation(
                            data, st.session_state.rules)
                        
                        # Store in session state
                        st.session_state.user_segments = segmented_data
                        
                        # Display segmentation results
                        st.success("Segmentation completed!")
                        
                        # Show segment distribution
                        st.subheader("Segment Distribution")
                        segment_counts = segmented_data['segment'].value_counts()
                        fig = px.pie(values=segment_counts.values, 
                                   names=segment_counts.index, 
                                   title="Distribution of Users Across Segments")
                        st.plotly_chart(fig)
                        
                        # Show sample users from each segment
                        st.subheader("Sample Users from Each Segment")
                        for segment in segmented_data['segment'].unique():
                            with st.expander(f"Segment: {segment}"):
                                sample = segmented_data[segmented_data['segment'] == segment].head(5)
                                st.dataframe(sample)
        
        elif segmentation_method == "RFM Analysis":
            st.subheader("RFM (Recency, Frequency, Monetary) Analysis")
            
            # Select columns for RFM analysis
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Identify potential date columns for recency
                date_cols = [col for col in data.columns if 'date' in col.lower() or 'time' in col.lower()]
                recency_col = st.selectbox("Select column for Recency (date/time)", date_cols if date_cols else data.columns)
            
            with col2:
                # Identify potential frequency columns
                freq_cols = [col for col in data.columns if 'count' in col.lower() or 'freq' in col.lower() or 'num' in col.lower()]
                frequency_col = st.selectbox("Select column for Frequency", freq_cols if freq_cols else data.columns)
            
            with col3:
                # Identify potential monetary columns
                monetary_cols = [col for col in data.columns if 'value' in col.lower() or 'price' in col.lower() or 'amount' in col.lower() or 'revenue' in col.lower()]
                monetary_col = st.selectbox("Select column for Monetary value", monetary_cols if monetary_cols else data.columns)
            
            # Customer ID column
            id_cols = [col for col in data.columns if 'id' in col.lower() or 'user' in col.lower() or 'customer' in col.lower()]
            customer_id_col = st.selectbox("Select customer ID column", id_cols if id_cols else data.columns)
            
            if st.button("Perform RFM Analysis"):
                with st.spinner("Performing RFM analysis..."):
                    try:
                        # Convert date column to datetime if needed
                        if data[recency_col].dtype != 'datetime64[ns]':
                            data[recency_col] = pd.to_datetime(data[recency_col])
                        
                        # Perform RFM analysis
                        rfm_data = dp.perform_rfm_analysis(
                            data, customer_id_col, recency_col, frequency_col, monetary_col)
                        
                        # Store in session state
                        st.session_state.user_segments = rfm_data
                        
                        # Display segmentation results
                        st.success("RFM Analysis completed!")
                        
                        # Show segment distribution
                        st.subheader("RFM Segment Distribution")
                        segment_counts = rfm_data['rfm_segment'].value_counts()
                        fig = px.bar(x=segment_counts.index, y=segment_counts.values, 
                                   title="Distribution of Users Across RFM Segments")
                        fig.update_layout(xaxis_title="RFM Segment", yaxis_title="Number of Users")
                        st.plotly_chart(fig)
                        
                        # Show RFM scores distribution
                        st.subheader("RFM Scores Distribution")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            fig = px.histogram(rfm_data, x='recency_score', 
                                             title="Recency Score Distribution")
                            st.plotly_chart(fig)
                        
                        with col2:
                            fig = px.histogram(rfm_data, x='frequency_score', 
                                             title="Frequency Score Distribution")
                            st.plotly_chart(fig)
                        
                        with col3:
                            fig = px.histogram(rfm_data, x='monetary_score', 
                                             title="Monetary Score Distribution")
                            st.plotly_chart(fig)
                        
                        # Show RFM segment descriptions
                        st.subheader("RFM Segment Descriptions")
                        segment_descriptions = {
                            "Champions": "Best customers who buy often and recently, with high spending",
                            "Loyal Customers": "Bought most recently, frequently, and spend well",
                            "Potential Loyalists": "Recent customers with average frequency and spending",
                            "Promising": "Recent shoppers, but not frequent ones",
                            "Need Attention": "Above average recency, frequency, and monetary values",
                            "At Risk": "Previous high-value customers who haven't purchased recently",
                            "Can't Lose Them": "Once-loyal customers who haven't purchased recently",
                            "About To Sleep": "Below average recency and frequency, may be lost",
                            "Hibernating": "Low spenders who purchased a long time ago",
                            "Lost": "Lowest scores across all RFM metrics"
                        }
                        
                        for segment, description in segment_descriptions.items():
                            with st.expander(segment):
                                st.write(description)
                                if segment in rfm_data['rfm_segment'].unique():
                                    sample = rfm_data[rfm_data['rfm_segment'] == segment].head(3)
                                    st.write("Sample customers in this segment:")
                                    st.dataframe(sample)
                    
                    except Exception as e:
                        st.error(f"Error performing RFM analysis: {e}")
        
        elif segmentation_method == "Custom":
            st.subheader("Custom Segmentation")
            st.write("Define your own custom segmentation approach.")
            
            # Allow user to select a column to use as segment
            segment_col = st.selectbox(
                "Select an existing column to use as segment",
                data.columns
            )
            
            if st.button("Use Selected Column as Segment"):
                segmented_data = data.copy()
                segmented_data['segment'] = segmented_data[segment_col]
                
                # Store in session state
                st.session_state.user_segments = segmented_data
                
                # Display segmentation summary
                st.success(f"Using '{segment_col}' as segment identifier.")
                
                # Show segment distribution
                st.subheader("Segment Distribution")
                segment_counts = segmented_data['segment'].value_counts()
                fig = px.pie(values=segment_counts.values, 
                           names=segment_counts.index, 
                           title=f"Distribution of Users Across {segment_col} Segments")
                st.plotly_chart(fig)
            
            st.markdown("---")
            
            # Custom formula-based segmentation
            st.subheader("Formula-based Segmentation")
            st.write("Create segments based on a formula or threshold.")
            
            # Feature selection
            numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            segmentation_feature = st.selectbox("Select feature for segmentation", numeric_cols)
            
            # Define thresholds
            st.write("Define segment thresholds:")
            
            thresholds = []
            segment_names = []
            
            col1, col2 = st.columns(2)
            
            with col1:
                threshold1 = st.number_input("Threshold 1", value=data[segmentation_feature].quantile(0.33))
                threshold2 = st.number_input("Threshold 2", value=data[segmentation_feature].quantile(0.67))
                thresholds = [threshold1, threshold2]
            
            with col2:
                segment_name1 = st.text_input("Segment name for values below Threshold 1", "Low")
                segment_name2 = st.text_input("Segment name for values between thresholds", "Medium")
                segment_name3 = st.text_input("Segment name for values above Threshold 2", "High")
                segment_names = [segment_name1, segment_name2, segment_name3]
            
            if st.button("Create Custom Segments"):
                with st.spinner("Creating custom segments..."):
                    segmented_data = dp.create_threshold_segments(
                        data, segmentation_feature, thresholds, segment_names)
                    
                    # Store in session state
                    st.session_state.user_segments = segmented_data
                    
                    # Display segmentation results
                    st.success("Custom segmentation completed!")
                    
                    # Show segment distribution
                    st.subheader("Segment Distribution")
                    segment_counts = segmented_data['segment'].value_counts()
                    fig = px.bar(x=segment_counts.index, y=segment_counts.values, 
                               title="Distribution of Users Across Custom Segments")
                    fig.update_layout(xaxis_title="Segment", yaxis_title="Number of Users")
                    st.plotly_chart(fig)
                    
                    # Show feature distribution by segment
                    st.subheader(f"{segmentation_feature} Distribution by Segment")
                    fig = px.box(segmented_data, x='segment', y=segmentation_feature, 
                               title=f"{segmentation_feature} Values by Segment")
                    st.plotly_chart(fig)

# Recommendation Engine Page
elif page == "Recommendation Engine":
    st.header("Recommendation Engine")
    
    if st.session_state.data is None:
        st.warning("Please upload data first in the 'Data Upload & Processing' page.")
    else:
        data = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
        
        # Recommendation method selection
        rec_method = st.selectbox(
            "Choose Recommendation Method",
            ["Collaborative Filtering", "Content-Based Filtering", "Segment-Based Recommendations", "Popular Items"]
        )
        
        if rec_method == "Collaborative Filtering":
            st.subheader("Collaborative Filtering Recommendations")
            
            # Identify user and item columns
            id_cols = [col for col in data.columns if 'id' in col.lower() or 'user' in col.lower() or 'customer' in col.lower()]
            user_col = st.selectbox("Select user/customer ID column", id_cols if id_cols else data.columns)
            
            item_cols = [col for col in data.columns if 'item' in col.lower() or 'product' in col.lower() or 'content' in col.lower()]
            item_col = st.selectbox("Select item/product ID column", item_cols if item_cols else [col for col in data.columns if col != user_col])
            
            # Rating column
            rating_cols = [col for col in data.columns if 'rating' in col.lower() or 'score' in col.lower() or 'feedback' in col.lower()]
            rating_col = st.selectbox("Select rating/interaction column", rating_cols if rating_cols else [col for col in data.select_dtypes(include=['int64', 'float64']).columns if col not in [user_col, item_col]])
            
            # Algorithm parameters
            st.subheader("Algorithm Parameters")
            n_factors = st.slider("Number of latent factors", 5, 100, 20)
            n_recommendations = st.slider("Number of recommendations to generate", 3, 20, 5)
            
            # Generate recommendations
            if st.button("Generate Collaborative Filtering Recommendations"):
                with st.spinner("Training recommendation model and generating recommendations..."):
                    try:
                        # Prepare data for collaborative filtering
                        cf_data = data[[user_col, item_col, rating_col]].copy()
                        
                        # Train model and get recommendations
                        user_recs, model_evaluation = rec.collaborative_filtering_recommendations(
                            cf_data, user_col, item_col, rating_col, n_factors, n_recommendations)
                        
                        # Store recommendations in session state
                        st.session_state.recommendations = user_recs
                        
                        # Display model evaluation
                        st.subheader("Model Evaluation")
                        st.write(f"RMSE: {model_evaluation['rmse']:.4f}")
                        st.write(f"MAE: {model_evaluation['mae']:.4f}")
                        
                        # Display recommendations for a sample user
                        st.subheader("Sample Recommendations")
                        
                        # Let user select a specific user to see recommendations for
                        all_users = cf_data[user_col].unique()
                        selected_user = st.selectbox("Select a user to view recommendations", all_users)
                        
                        if selected_user in user_recs:
                            recs_for_user = user_recs[selected_user]
                            st.write(f"Top {len(recs_for_user)} recommendations for user {selected_user}:")
                            
                            # Display as table
                            rec_df = pd.DataFrame(recs_for_user, columns=["Item", "Predicted Rating"])
                            st.dataframe(rec_df)
                            
                            # Bar chart of recommendations
                            fig = px.bar(rec_df, x="Item", y="Predicted Rating", 
                                       title=f"Recommended Items for User {selected_user}")
                            st.plotly_chart(fig)
                        else:
                            st.warning(f"No recommendations available for user {selected_user}.")
                    
                    except Exception as e:
                        st.error(f"Error generating recommendations: {e}")
        
        elif rec_method == "Content-Based Filtering":
            st.subheader("Content-Based Filtering Recommendations")
            
            # Identify item columns
            item_cols = [col for col in data.columns if 'item' in col.lower() or 'product' in col.lower() or 'content' in col.lower()]
            item_col = st.selectbox("Select item/product ID column", item_cols if item_cols else data.columns)
            
            # Feature columns
            feature_cols = st.multiselect(
                "Select feature columns to use for content filtering",
                [col for col in data.columns if col != item_col],
                default=[col for col in data.columns if col != item_col][:min(5, len(data.columns)-1)]
            )
            
            # Generate recommendations
            if feature_cols and st.button("Generate Content-Based Recommendations"):
                with st.spinner("Computing content similarities and generating recommendations..."):
                    try:
                        # Create content profiles and compute similarities
                        item_similarities, item_features = rec.content_based_recommendations(
                            data, item_col, feature_cols)
                        
                        # Store in session state (in a different format than CF)
                        st.session_state.recommendations = {"content_based": item_similarities}
                        
                        # Display sample item similarities
                        st.subheader("Item Similarity Analysis")
                        
                        # Let user select an item to see similar items
                        all_items = data[item_col].unique()
                        selected_item = st.selectbox("Select an item to find similar items", all_items)
                        
                        if selected_item in item_similarities:
                            similar_items = item_similarities[selected_item]
                            
                            st.write(f"Top similar items to {selected_item}:")
                            
                            # Display as table
                            sim_df = pd.DataFrame(similar_items, columns=["Item", "Similarity Score"])
                            st.dataframe(sim_df)
                            
                            # Bar chart of similar items
                            fig = px.bar(sim_df, x="Item", y="Similarity Score", 
                                       title=f"Items Similar to {selected_item}")
                            st.plotly_chart(fig)
                            
                            # Show feature comparison
                            st.subheader("Feature Comparison")
                            item_feature_values = item_features.loc[[selected_item] + sim_df["Item"].tolist()[:3]]
                            
                            # Radar chart of features
                            features_to_plot = item_feature_values.columns
                            fig = go.Figure()
                            
                            for idx, item_id in enumerate(item_feature_values.index):
                                fig.add_trace(go.Scatterpolar(
                                    r=item_feature_values.loc[item_id].values,
                                    theta=features_to_plot,
                                    fill='toself',
                                    name=f"Item {item_id}"
                                ))
                            
                            fig.update_layout(
                                polar=dict(
                                    radialaxis=dict(
                                        visible=True,
                                        range=[0, 1]
                                    )),
                                showlegend=True,
                                title="Feature Comparison of Similar Items"
                            )
                            
                            st.plotly_chart(fig)
                        else:
                            st.warning(f"Item {selected_item} not found in similarity matrix.")
                    
                    except Exception as e:
                        st.error(f"Error generating content-based recommendations: {e}")
        
        elif rec_method == "Segment-Based Recommendations":
            st.subheader("Segment-Based Recommendations")
            
            if st.session_state.user_segments is None:
                st.warning("Please create user segments first in the 'User Segmentation' page.")
            else:
                segmented_data = st.session_state.user_segments
                
                # Check if segment column exists
                if 'segment' not in segmented_data.columns and 'cluster' not in segmented_data.columns and 'rfm_segment' not in segmented_data.columns:
                    st.error("Segment information not found in the segmented data. Please perform segmentation first.")
                else:
                    # Determine segment column
                    segment_col = 'segment' if 'segment' in segmented_data.columns else ('cluster' if 'cluster' in segmented_data.columns else 'rfm_segment')
                    
                    # Display available segments
                    st.write(f"Available segments: {', '.join(segmented_data[segment_col].unique())}")
                    
                    # Identify item columns
                    item_cols = [col for col in segmented_data.columns if 'item' in col.lower() or 'product' in col.lower() or 'content' in col.lower()]
                    item_col = st.selectbox("Select item/product ID column", item_cols if item_cols else [col for col in segmented_data.columns if col != segment_col])
                    
                    # Rating/interaction column
                    interaction_cols = [col for col in segmented_data.columns if 'rating' in col.lower() or 'score' in col.lower() or 'view' in col.lower() or 'purchase' in col.lower()]
                    interaction_col = st.selectbox("Select interaction column", interaction_cols if interaction_cols else [col for col in segmented_data.select_dtypes(include=['int64', 'float64']).columns if col != segment_col and col != item_col])
                    
                    # User ID column
                    id_cols = [col for col in segmented_data.columns if 'id' in col.lower() or 'user' in col.lower() or 'customer' in col.lower()]
                    user_col = st.selectbox("Select user ID column", id_cols if id_cols else [col for col in segmented_data.columns if col not in [segment_col, item_col, interaction_col]])
                    
                    if st.button("Generate Segment-Based Recommendations"):
                        with st.spinner("Generating segment-based recommendations..."):
                            try:
                                # Get recommendations by segment
                                segment_recommendations = rec.segment_based_recommendations(
                                    segmented_data, segment_col, user_col, item_col, interaction_col)
                                
                                # Store in session state
                                st.session_state.recommendations = segment_recommendations
                                
                                # Display recommendations by segment
                                st.subheader("Top Items by Segment")
                                
                                for segment, items in segment_recommendations.items():
                                    with st.expander(f"Segment: {segment}"):
                                        # Display as table
                                        rec_df = pd.DataFrame(items, columns=["Item", "Score"])
                                        st.dataframe(rec_df)
                                        
                                        # Bar chart of top items
                                        fig = px.bar(rec_df, x="Item", y="Score", 
                                                   title=f"Top Items for Segment: {segment}")
                                        st.plotly_chart(fig)
                                
                                # Allow recommending for a specific user
                                st.subheader("Recommendations for Specific User")
                                
                                # Let user select a specific user
                                all_users = segmented_data[user_col].unique()
                                selected_user = st.selectbox("Select a user to get recommendations", all_users)
                                
                                # Get user's segment
                                user_segment = segmented_data[segmented_data[user_col] == selected_user][segment_col].iloc[0]
                                
                                st.write(f"User {selected_user} belongs to segment: {user_segment}")
                                
                                if user_segment in segment_recommendations:
                                    st.write(f"Recommended items for user {selected_user} based on segment {user_segment}:")
                                    
                                    # Display recommendations
                                    rec_df = pd.DataFrame(segment_recommendations[user_segment], columns=["Item", "Score"])
                                    st.dataframe(rec_df)
                            
                            except Exception as e:
                                st.error(f"Error generating segment-based recommendations: {e}")
        
        elif rec_method == "Popular Items":
            st.subheader("Popular Items Recommendation")
            
            # Identify item and interaction columns
            item_cols = [col for col in data.columns if 'item' in col.lower() or 'product' in col.lower() or 'content' in col.lower()]
            item_col = st.selectbox("Select item/product ID column", item_cols if item_cols else data.columns)
            
            interaction_cols = [col for col in data.columns if 'rating' in col.lower() or 'score' in col.lower() or 'view' in col.lower() or 'purchase' in col.lower()]
            interaction_col = st.selectbox("Select interaction column", interaction_cols if interaction_cols else [col for col in data.select_dtypes(include=['int64', 'float64']).columns if col != item_col])
            
            n_recommendations = st.slider("Number of popular items to show", 5, 50, 10)
            
            if st.button("Find Popular Items"):
                with st.spinner("Finding popular items..."):
                    try:
                        # Get popular items
                        popular_items = rec.popular_items_recommendations(
                            data, item_col, interaction_col, n_recommendations)
                        
                        # Store in session state
                        st.session_state.recommendations = {"popular": popular_items}
                        
                        # Display popular items
                        st.subheader(f"Top {n_recommendations} Popular Items")
                        
                        # Display as table
                        pop_df = pd.DataFrame(popular_items, columns=["Item", "Popularity Score"])
                        st.dataframe(pop_df)
                        
                        # Bar chart of popular items
                        fig = px.bar(pop_df, x="Item", y="Popularity Score", 
                                   title=f"Top {n_recommendations} Popular Items")
                        st.plotly_chart(fig)
                    
                    except Exception as e:
                        st.error(f"Error finding popular items: {e}")

# A/B Testing & Evaluation Page
elif page == "A/B Testing & Evaluation":
    st.header("A/B Testing & Evaluation")
    
    # Main tabs
    evaluation_tab, ab_testing_tab = st.tabs(["Recommendation Evaluation", "A/B Testing Simulation"])
    
    with evaluation_tab:
        st.subheader("Evaluate Recommendation Performance")
        
        if st.session_state.recommendations is None:
            st.warning("Please generate recommendations first in the 'Recommendation Engine' page.")
        else:
            # Let user select evaluation metrics
            metrics = st.multiselect(
                "Select evaluation metrics",
                ["Precision", "Recall", "NDCG", "Diversity", "Coverage"],
                default=["Precision", "Recall"]
            )
            
            if st.session_state.data is not None:
                data = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
                
                # Identify user, item, and rating columns
                id_cols = [col for col in data.columns if 'id' in col.lower() or 'user' in col.lower() or 'customer' in col.lower()]
                user_col = st.selectbox("Select user ID column", id_cols if id_cols else data.columns)
                
                item_cols = [col for col in data.columns if 'item' in col.lower() or 'product' in col.lower() or 'content' in col.lower()]
                item_col = st.selectbox("Select item ID column", item_cols if item_cols else [col for col in data.columns if col != user_col])
                
                rating_cols = [col for col in data.columns if 'rating' in col.lower() or 'score' in col.lower() or 'feedback' in col.lower()]
                rating_col = st.selectbox("Select rating column", rating_cols if rating_cols else [col for col in data.select_dtypes(include=['int64', 'float64']).columns if col not in [user_col, item_col]])
                
                if st.button("Evaluate Recommendations"):
                    with st.spinner("Evaluating recommendation performance..."):
                        try:
                            # Get recommendations
                            recommendations = st.session_state.recommendations
                            
                            # Evaluate recommendations
                            eval_results = rec.evaluate_recommendations(
                                data, recommendations, user_col, item_col, rating_col, metrics)
                            
                            # Display evaluation results
                            st.subheader("Evaluation Results")
                            
                            for metric, value in eval_results.items():
                                st.metric(label=metric, value=f"{value:.4f}")
                            
                            # Plot results
                            eval_df = pd.DataFrame({
                                'Metric': list(eval_results.keys()),
                                'Value': list(eval_results.values())
                            })
                            
                            fig = px.bar(eval_df, x='Metric', y='Value', 
                                       title="Recommendation Evaluation Metrics")
                            st.plotly_chart(fig)
                            
                            # Detailed evaluation by user segment if available
                            if st.session_state.user_segments is not None:
                                st.subheader("Evaluation by User Segment")
                                
                                segmented_data = st.session_state.user_segments
                                segment_col = 'segment' if 'segment' in segmented_data.columns else ('cluster' if 'cluster' in segmented_data.columns else 'rfm_segment')
                                
                                if segment_col in segmented_data.columns:
                                    segment_eval = rec.evaluate_by_segment(
                                        data, segmented_data, segment_col, recommendations, 
                                        user_col, item_col, rating_col, metrics)
                                    
                                    # Display segment evaluation
                                    for segment, segment_results in segment_eval.items():
                                        with st.expander(f"Segment: {segment}"):
                                            for metric, value in segment_results.items():
                                                st.metric(label=metric, value=f"{value:.4f}")
                        
                        except Exception as e:
                            st.error(f"Error evaluating recommendations: {e}")
    
    with ab_testing_tab:
        st.subheader("A/B Testing Simulation")
        
        # A/B testing options
        test_scenario = st.selectbox(
            "Select A/B Testing Scenario",
            ["Compare Recommendation Algorithms", "Test Different Personalization Strategies", 
             "Upload A/B Test Results"]
        )
        
        if test_scenario == "Compare Recommendation Algorithms":
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Algorithm A (Control)")
                algo_a = st.selectbox(
                    "Select first algorithm",
                    ["Collaborative Filtering", "Content-Based", "Popularity-Based"]
                )
                
                # Parameters for Algorithm A
                if algo_a == "Collaborative Filtering":
                    n_factors_a = st.slider("Number of factors (A)", 10, 100, 20)
                elif algo_a == "Content-Based":
                    similarity_a = st.selectbox("Similarity metric (A)", ["cosine", "jaccard", "euclidean"])
                else:
                    timeframe_a = st.selectbox("Popularity timeframe (A)", ["1 day", "1 week", "1 month", "all time"])
            
            with col2:
                st.subheader("Algorithm B (Test)")
                algo_b = st.selectbox(
                    "Select second algorithm",
                    ["Collaborative Filtering", "Content-Based", "Popularity-Based"],
                    index=1
                )
                
                # Parameters for Algorithm B
                if algo_b == "Collaborative Filtering":
                    n_factors_b = st.slider("Number of factors (B)", 10, 100, 50)
                elif algo_b == "Content-Based":
                    similarity_b = st.selectbox("Similarity metric (B)", ["cosine", "jaccard", "euclidean"])
                else:
                    timeframe_b = st.selectbox("Popularity timeframe (B)", ["1 day", "1 week", "1 month", "all time"])
            
            # Simulation parameters
            st.subheader("Simulation Parameters")
            
            sample_size = st.slider("User sample size", 100, 10000, 1000)
            success_metric = st.selectbox(
                "Success metric",
                ["Click-through rate", "Conversion rate", "Average order value", "User engagement"]
            )
            
            if st.button("Run A/B Test Simulation"):
                with st.spinner("Simulating A/B test..."):
                    # Simulate A/B test results
                    ab_results = rec.simulate_ab_test(
                        algo_a, algo_b, sample_size, success_metric)
                    
                    # Store in session state
                    st.session_state.experiment_results = ab_results
                    
                    # Display test results
                    st.subheader("A/B Test Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            label=f"{success_metric} (A)",
                            value=f"{ab_results['control_rate']:.4f}"
                        )
                    
                    with col2:
                        st.metric(
                            label=f"{success_metric} (B)",
                            value=f"{ab_results['treatment_rate']:.4f}",
                            delta=f"{ab_results['lift']:.2%}"
                        )
                    
                    with col3:
                        st.metric(
                            label="Statistical Significance",
                            value=f"p = {ab_results['p_value']:.4f}"
                        )
                    
                    # Visualization of results
                    st.subheader("Visual Comparison")
                    
                    # Bar chart comparison
                    comparison_df = pd.DataFrame({
                        'Algorithm': ['Algorithm A (Control)', 'Algorithm B (Test)'],
                        success_metric: [ab_results['control_rate'], ab_results['treatment_rate']]
                    })
                    
                    fig = px.bar(comparison_df, x='Algorithm', y=success_metric,
                               title=f"Comparison of {success_metric}")
                    st.plotly_chart(fig)
                    
                    # Distribution plot
                    if 'control_samples' in ab_results and 'treatment_samples' in ab_results:
                        st.subheader("Distribution of Individual Results")
                        
                        # Combine samples for plotting
                        control_df = pd.DataFrame({
                            'value': ab_results['control_samples'],
                            'group': 'Control'
                        })
                        
                        treatment_df = pd.DataFrame({
                            'value': ab_results['treatment_samples'],
                            'group': 'Treatment'
                        })
                        
                        combined_df = pd.concat([control_df, treatment_df])
                        
                        fig = px.histogram(combined_df, x="value", color="group", 
                                         barmode="overlay", opacity=0.7,
                                         title=f"Distribution of {success_metric}")
                        st.plotly_chart(fig)
                    
                    # Statistical analysis
                    st.subheader("Statistical Analysis")
                    
                    significance = "significant" if ab_results['p_value'] < 0.05 else "not significant"
                    st.write(f"The difference between Algorithm A and Algorithm B is statistically {significance} (p = {ab_results['p_value']:.4f}).")
                    
                    st.write(f"Lift: {ab_results['lift']:.2%}")
                    st.write(f"Confidence interval: [{ab_results['confidence_interval'][0]:.4f}, {ab_results['confidence_interval'][1]:.4f}]")
                    
                    # Recommendation
                    st.subheader("Recommendation")
                    
                    if ab_results['lift'] > 0 and ab_results['p_value'] < 0.05:
                        st.success(f"Algorithm B outperforms Algorithm A with a statistically significant lift of {ab_results['lift']:.2%}. Consider implementing Algorithm B.")
                    elif ab_results['lift'] < 0 and ab_results['p_value'] < 0.05:
                        st.error(f"Algorithm A outperforms Algorithm B with a statistically significant lift of {-ab_results['lift']:.2%}. Stick with Algorithm A.")
                    else:
                        st.info("No statistically significant difference detected between the algorithms. Consider running the test longer or with more users.")
        
        elif test_scenario == "Test Different Personalization Strategies":
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Strategy A (Control)")
                strategy_a = st.selectbox(
                    "Select first strategy",
                    ["No personalization", "Segment-based personalization", "Individual personalization"]
                )
                
                # Parameters for Strategy A
                if strategy_a == "Segment-based personalization":
                    n_segments_a = st.slider("Number of segments (A)", 2, 10, 4)
                elif strategy_a == "Individual personalization":
                    depth_a = st.slider("Personalization depth (A)", 1, 10, 5)
            
            with col2:
                st.subheader("Strategy B (Test)")
                strategy_b = st.selectbox(
                    "Select second strategy",
                    ["No personalization", "Segment-based personalization", "Individual personalization"],
                    index=1
                )
                
                # Parameters for Strategy B
                if strategy_b == "Segment-based personalization":
                    n_segments_b = st.slider("Number of segments (B)", 2, 10, 6)
                elif strategy_b == "Individual personalization":
                    depth_b = st.slider("Personalization depth (B)", 1, 10, 7)
            
            # Simulation parameters
            st.subheader("Simulation Parameters")
            
            sample_size = st.slider("User sample size for strategies", 100, 10000, 2000)
            success_metrics = st.multiselect(
                "Success metrics",
                ["Click-through rate", "Conversion rate", "Average order value", "User engagement time", "Retention"],
                default=["Click-through rate", "Conversion rate"]
            )
            
            if st.button("Run Strategy A/B Test"):
                with st.spinner("Simulating personalization strategy test..."):
                    # Simulate A/B test results for each metric
                    strategy_results = {}
                    
                    for metric in success_metrics:
                        result = rec.simulate_ab_test(
                            strategy_a, strategy_b, sample_size, metric)
                        strategy_results[metric] = result
                    
                    # Store in session state
                    st.session_state.experiment_results = strategy_results
                    
                    # Display overview
                    st.subheader("Test Results Overview")
                    
                    # Create summary table
                    summary_data = []
                    
                    for metric, result in strategy_results.items():
                        summary_data.append({
                            'Metric': metric,
                            'Control Value': result['control_rate'],
                            'Test Value': result['treatment_rate'],
                            'Lift': result['lift'],
                            'p-value': result['p_value'],
                            'Significant': result['p_value'] < 0.05
                        })
                    
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df)
                    
                    # Visualize results
                    st.subheader("Visual Comparison by Metric")
                    
                    for metric, result in strategy_results.items():
                        with st.expander(f"{metric} Results"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric(
                                    label=f"{metric} (Strategy A)",
                                    value=f"{result['control_rate']:.4f}"
                                )
                                
                                st.metric(
                                    label=f"{metric} (Strategy B)",
                                    value=f"{result['treatment_rate']:.4f}",
                                    delta=f"{result['lift']:.2%}"
                                )
                                
                                significance = "significant" if result['p_value'] < 0.05 else "not significant"
                                st.write(f"Statistically {significance} (p = {result['p_value']:.4f})")
                            
                            with col2:
                                # Comparison bar chart
                                comparison_df = pd.DataFrame({
                                    'Strategy': ['Strategy A', 'Strategy B'],
                                    metric: [result['control_rate'], result['treatment_rate']]
                                })
                                
                                fig = px.bar(comparison_df, x='Strategy', y=metric,
                                           title=f"Comparison of {metric}")
                                st.plotly_chart(fig)
                    
                    # Overall recommendation
                    st.subheader("Overall Recommendation")
                    
                    # Count significant improvements
                    sig_improvements = sum(1 for r in strategy_results.values() 
                                         if r['lift'] > 0 and r['p_value'] < 0.05)
                    sig_declines = sum(1 for r in strategy_results.values() 
                                     if r['lift'] < 0 and r['p_value'] < 0.05)
                    
                    if sig_improvements > sig_declines:
                        st.success(f"Strategy B shows significant improvements in {sig_improvements} out of {len(success_metrics)} metrics. Consider implementing Strategy B.")
                    elif sig_declines > sig_improvements:
                        st.error(f"Strategy B shows significant declines in {sig_declines} out of {len(success_metrics)} metrics. Stick with Strategy A.")
                    else:
                        st.info("Results are mixed or not significant. Consider refining the strategies or running a longer test.")
        
        elif test_scenario == "Upload A/B Test Results":
            st.subheader("Upload External A/B Test Results")
            st.write("If you have conducted A/B tests externally, you can upload the results for analysis.")
            
            ab_file = st.file_uploader("Upload A/B test results (CSV format)", type="csv")
            
            if ab_file is not None:
                try:
                    ab_data = pd.read_csv(ab_file)
                    
                    # Display uploaded data
                    st.subheader("Uploaded A/B Test Data")
                    st.dataframe(ab_data.head())
                    
                    # Columns to use for analysis
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        group_col = st.selectbox(
                            "Column identifying test groups",
                            ab_data.columns
                        )
                    
                    with col2:
                        metric_col = st.selectbox(
                            "Column with the metric value",
                            [c for c in ab_data.columns if c != group_col]
                        )
                    
                    with col3:
                        user_col = st.selectbox(
                            "Column with user ID (optional)",
                            ["None"] + [c for c in ab_data.columns if c not in [group_col, metric_col]]
                        )
                    
                    # Check for test group values
                    group_values = ab_data[group_col].unique()
                    
                    if len(group_values) != 2:
                        st.warning(f"Expected 2 test groups, but found {len(group_values)}. Please ensure your data has exactly 2 groups (control and treatment).")
                    
                    control_group = st.selectbox(
                        "Select control group value",
                        group_values
                    )
                    
                    treatment_group = [g for g in group_values if g != control_group][0]
                    
                    if st.button("Analyze A/B Test Results"):
                        with st.spinner("Analyzing test results..."):
                            # Analyze the test results
                            analysis_results = rec.analyze_ab_test_results(
                                ab_data, group_col, metric_col, control_group, treatment_group, 
                                user_col if user_col != "None" else None)
                            
                            # Store results
                            st.session_state.experiment_results = analysis_results
                            
                            # Display results
                            st.subheader("A/B Test Analysis Results")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric(
                                    label=f"Control Group ({control_group})",
                                    value=f"{analysis_results['control_mean']:.4f}"
                                )
                            
                            with col2:
                                st.metric(
                                    label=f"Treatment Group ({treatment_group})",
                                    value=f"{analysis_results['treatment_mean']:.4f}",
                                    delta=f"{analysis_results['relative_difference']:.2%}"
                                )
                            
                            with col3:
                                st.metric(
                                    label="p-value",
                                    value=f"{analysis_results['p_value']:.4f}"
                                )
                            
                            # Visualizations
                            st.subheader("Visual Analysis")
                            
                            # Group comparison
                            fig = px.bar(
                                x=[f"Control ({control_group})", f"Treatment ({treatment_group})"],
                                y=[analysis_results['control_mean'], analysis_results['treatment_mean']],
                                error_y=[analysis_results['control_se'], analysis_results['treatment_se']],
                                title="Comparison of Group Means with Standard Error"
                            )
                            st.plotly_chart(fig)
                            
                            # Distribution comparison if user data is available
                            if user_col != "None":
                                fig = px.histogram(
                                    ab_data, x=metric_col, color=group_col,
                                    barmode="overlay", opacity=0.7,
                                    title="Distribution of Metric by Test Group"
                                )
                                st.plotly_chart(fig)
                            
                            # Statistical details
                            st.subheader("Statistical Details")
                            
                            significance = "significant" if analysis_results['p_value'] < 0.05 else "not significant"
                            st.write(f"The difference between the control and treatment groups is statistically {significance} (p = {analysis_results['p_value']:.4f}).")
                            
                            st.write(f"Absolute difference: {analysis_results['absolute_difference']:.4f}")
                            st.write(f"Relative difference: {analysis_results['relative_difference']:.2%}")
                            st.write(f"95% Confidence Interval: [{analysis_results['ci_lower']:.4f}, {analysis_results['ci_upper']:.4f}]")
                            
                            # Sample sizes
                            st.write(f"Control group size: {analysis_results['control_size']}")
                            st.write(f"Treatment group size: {analysis_results['treatment_size']}")
                            
                            # Conclusion and recommendation
                            st.subheader("Conclusion")
                            
                            if analysis_results['p_value'] < 0.05:
                                if analysis_results['relative_difference'] > 0:
                                    st.success(f"The treatment ({treatment_group}) shows a statistically significant improvement of {analysis_results['relative_difference']:.2%} over the control ({control_group}). Recommend implementing the treatment.")
                                else:
                                    st.error(f"The treatment ({treatment_group}) shows a statistically significant decline of {-analysis_results['relative_difference']:.2%} compared to the control ({control_group}). Recommend staying with the control.")
                            else:
                                st.info("No statistically significant difference detected between the control and treatment groups. Consider running the test longer or with more users.")
                
                except Exception as e:
                    st.error(f"Error analyzing A/B test results: {e}")

# Documentation Page
elif page == "Documentation":
    st.header("Documentation")
    
    st.write("""
    ## About This Platform
    
    This data science platform is designed to help you analyze user behavior and implement 
    personalization features for your online platform. It provides a set of tools for data 
    processing, analysis, segmentation, recommendation generation, and A/B testing.
    
    ### Features
    
    1. **Data Upload & Processing**
       - Support for CSV and JSON data formats
       - Data cleaning and preprocessing options
       - Sample datasets for demonstration
    
    2. **User Behavior Analysis**
       - Univariate and bivariate analysis
       - User engagement metrics
       - Temporal pattern analysis
       - Behavioral clustering
    
    3. **User Segmentation**
       - Clustering-based segmentation
       - Rule-based segmentation
       - RFM (Recency, Frequency, Monetary) analysis
       - Custom segmentation approaches
    
    4. **Recommendation Engine**
       - Collaborative filtering
       - Content-based filtering
       - Segment-based recommendations
       - Popular items recommendations
    
    5. **A/B Testing & Evaluation**
       - Recommendation performance evaluation
       - A/B testing simulation
       - External A/B test results analysis
    
    ### Getting Started
    
    1. Begin by uploading your data in the "Data Upload & Processing" page
    2. Explore your data in the "User Behavior Analysis" page
    3. Create user segments in the "User Segmentation" page
    4. Generate recommendations in the "Recommendation Engine" page
    5. Evaluate and test your personalization strategies in the "A/B Testing & Evaluation" page
    """)
    
    # Methodologies
    with st.expander("Methodologies"):
        st.write("""
        ### Data Processing
        
        - **Missing Value Handling**: Implemented through imputation techniques such as mean/median replacement for numerical variables and mode replacement for categorical variables
        - **Feature Encoding**: Categorical variables are encoded using one-hot encoding or label encoding as appropriate
        - **Normalization**: Numeric features are normalized using min-max scaling or standardization
        
        ### User Segmentation
        
        - **Clustering**: K-means clustering is used to group users based on similar behaviors and characteristics
        - **RFM Analysis**: Users are segmented based on their Recency (how recently they interacted), Frequency (how often they interact), and Monetary value (how much they spend)
        - **Rule-based Segmentation**: Custom rules defined by business logic to segment users
        
        ### Recommendation Algorithms
        
        - **Collaborative Filtering**: Uses matrix factorization to find latent factors that explain observed user-item interactions
        - **Content-based Filtering**: Recommends items similar to those a user has shown interest in, based on item features
        - **Segment-based Recommendations**: Leverages user segments to provide recommendations based on segment-level preferences
        - **Popular Items**: Simple approach recommending the most popular items, serving as a baseline
        
        ### Evaluation Metrics
        
        - **Precision**: Proportion of recommended items that are relevant
        - **Recall**: Proportion of relevant items that were recommended
        - **NDCG (Normalized Discounted Cumulative Gain)**: Measures ranking quality, emphasizing the importance of relevant items appearing at the top of the recommendation list
        - **Diversity**: Variety of items in the recommendation set
        - **Coverage**: Proportion of available items that the system can recommend
        
        ### A/B Testing
        
        - **Statistical Significance**: Two-sample t-tests for comparing means between control and treatment groups
        - **Confidence Intervals**: Bootstrap methods for estimating confidence intervals
        - **Effect Size**: Cohen's d for measuring the standardized difference between groups
        """)
    
    # Best Practices
    with st.expander("Best Practices"):
        st.write("""
        ### Data Quality
        
        - **Ensure data consistency**: Check for and handle outliers, inconsistent values, and data anomalies
        - **Appropriate data sampling**: If working with large datasets, ensure representative sampling
        - **Feature selection**: Focus on relevant features to avoid noise and overfitting
        
        ### User Segmentation
        
        - **Select appropriate number of segments**: Too few segments might not capture important differences, while too many can lead to overfitting
        - **Validate segment stability**: Ensure segments are stable over time and across different data samples
        - **Interpret segments meaningfully**: Give segments business-relevant names and descriptions
        
        ### Recommendation Systems
        
        - **Cold start handling**: Have strategies for new users and new items
        - **Balance personalization and diversity**: Highly personalized recommendations may create filter bubbles
        - **Consider context**: Time of day, device, location, and other contextual factors can improve recommendation relevance
        
        ### A/B Testing
        
        - **Define clear success metrics**: Decide on primary and secondary metrics before testing
        - **Calculate appropriate sample size**: Ensure sufficient statistical power
        - **Avoid peeking**: Don't stop tests early based on preliminary results
        - **Control for external factors**: Be aware of seasonality and other external influences
        """)
    
    # Use Cases
    with st.expander("Example Use Cases"):
        st.write("""
        ### E-commerce
        
        - **Product recommendations**: Personalized product suggestions based on browsing and purchase history
        - **Category affinity**: Identify which product categories each user has the strongest affinity for
        - **Personalized search ranking**: Adjust search results based on user preferences
        - **Strategic discounting**: Target promotions to specific user segments
        
        ### Content Platforms
        
        - **Content recommendations**: Suggest articles, videos, or other content based on consumption history
        - **User journey optimization**: Understand typical content consumption patterns
        - **Engagement prediction**: Predict which content will drive the most engagement for each user
        - **Churn prevention**: Identify users at risk of churning and target them with relevant content
        
        ### Financial Services
        
        - **Service recommendations**: Suggest financial products based on customer profile and behavior
        - **Risk segmentation**: Group customers based on risk profiles
        - **Personalized financial advice**: Provide tailored recommendations based on financial situation
        - **Fraud detection**: Identify unusual patterns that may indicate fraudulent activity
        """)
    
    # FAQ
    with st.expander("Frequently Asked Questions"):
        st.write("""
        ### General Questions
        
        **Q: What kind of data do I need to use this platform?**
        
        A: At minimum, you need data that identifies users, items/products/content they've interacted with, and some measure of those interactions (views, purchases, ratings, etc.). Additional user and item attributes will improve personalization quality.
        
        **Q: How much data is needed for good recommendations?**
        
        A: This varies by algorithm, but generally:
        - Collaborative filtering typically needs at least a few thousand interactions
        - Content-based filtering can work with less interaction data but needs good item attributes
        - Popular items recommendations can work with any amount of data
        
        **Q: How frequently should I update my models?**
        
        A: This depends on your business and data velocity. E-commerce sites with high traffic might update daily, while more stable domains might update weekly or monthly.
        
        ### Technical Questions
        
        **Q: Which recommendation algorithm should I choose?**
        
        A: It depends on your specific case:
        - Collaborative filtering works well when you have rich interaction data
        - Content-based works well when you have good item attribute data
        - Popular items serve as a good baseline and fallback option
        - Often a hybrid approach works best
        
        **Q: How do I handle the cold start problem?**
        
        A: For new users, start with popular items or content-based recommendations. For new items, use content-based approaches until sufficient interaction data is available.
        
        **Q: What metrics should I use to evaluate my recommendations?**
        
        A: Consider both offline metrics (precision, recall, NDCG) and online metrics from A/B tests (click-through rate, conversion rate). The most important metrics align with your business objectives.
        
        ### Implementation Questions
        
        **Q: How can I deploy these models to production?**
        
        A: The models can be exported and deployed in several ways:
        - As a batch process generating recommendations periodically
        - As an API service generating real-time recommendations
        - Integrated directly into your application backend
        
        **Q: How do I interpret the segmentation results?**
        
        A: Look for distinguishing features of each segment. What behaviors or attributes make each segment unique? How do they differ in their interaction patterns? Assign business-relevant names to help stakeholders understand.
        
        **Q: How long should I run A/B tests?**
        
        A: Tests should run until they reach statistical significance. This depends on effect size, traffic volume, and conversion rates, but typically at least 1-2 weeks to account for weekly patterns.
        """)
    
    # Resources
    with st.expander("Additional Resources"):
        st.write("""
        ### Books
        
        - "Recommender Systems: The Textbook" by Charu C. Aggarwal
        - "Mining of Massive Datasets" by Jure Leskovec, Anand Rajaraman, and Jeff Ullman
        - "Python Machine Learning" by Sebastian Raschka and Vahid Mirjalili
        
        ### Online Courses
        
        - "Recommender Systems Specialization" on Coursera
        - "Machine Learning for Recommender Systems" on edX
        - "A/B Testing" by Google on Udacity
        
        ### Libraries and Tools
        
        - Surprise: A Python library for recommender systems
        - Implicit: Fast collaborative filtering for implicit feedback datasets
        - LightFM: A hybrid recommendation algorithm in Python
        - Scikit-learn: Machine learning library for Python
        - TensorFlow Recommenders: Deep learning-based recommendation models
        
        ### Research Papers
        
        - "Matrix Factorization Techniques for Recommender Systems" by Y. Koren, R. Bell, and C. Volinsky
        - "BPR: Bayesian Personalized Ranking from Implicit Feedback" by S. Rendle et al.
        - "Deep Neural Networks for YouTube Recommendations" by P. Covington et al.
        - "Item-Based Collaborative Filtering Recommendation Algorithms" by B. Sarwar et al.
        """)

# Footer
st.markdown("---")
st.markdown("Personalization Data Science Platform | Built with Streamlit")
