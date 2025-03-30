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
     "User Segmentation", "Documentation"]
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
                    # Sample data if it's too large (for better performance)
                    if len(data) > 10000:
                        sample_size = min(10000, int(len(data) * 0.3))
                        data_sample = data.sample(n=sample_size, random_state=42)
                    else:
                        data_sample = data

                    # Perform clustering on sample
                    cluster_data, cluster_centers, silhouette_avg = viz.perform_clustering(
                        data_sample, cluster_features, n_clusters)

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
elif page == "Documentation":
    st.header("Documentation")
    
    st.subheader("About This Platform")
    st.write("""
    The Personalization Data Science Platform is a tool designed to help analyze user behavior 
    and implement personalization features on your online platform. 
    
    This application allows you to:
    1. Upload and process user behavior data
    2. Analyze patterns and trends in user behavior
    3. Create user segments based on behavior or characteristics
    """)
    
    st.subheader("How to Use the Platform")
    
    st.markdown("""
    #### 1. Data Upload & Processing
    - Upload your own CSV or JSON data file
    - Use sample datasets if you don't have your own data
    - Apply data processing operations like handling missing values and normalization
    
    #### 2. User Behavior Analysis
    - Explore distributions and relationships in your data
    - Analyze user engagement metrics
    - Identify temporal patterns in user activity
    - Discover behavioral clusters in your user base
    
    #### 3. User Segmentation
    - Create user segments using different methods:
      - RFM Analysis (Recency, Frequency, Monetary)
      - K-means clustering
      - Rule-based segmentation
    
    #### Need Help?
    If you need assistance or have questions about using this platform, 
    please contact support at support@personalization-platform.com.
    """)
    
    st.subheader("Downloading Data")
    st.markdown("""
    Our platform now includes functionality to collect data from online sources. Using web scraping 
    technologies, you can gather data from websites to analyze and enhance your personalization strategies.
    
    #### How to Use Web Scraping:
    1. Enter the URL of the website you want to collect data from
    2. The system will extract the main text content
    3. Process and analyze this data to identify user preferences and behaviors
    
    #### Example Use Cases:
    - Collect product reviews to understand customer preferences
    - Gather news articles related to user interests
    - Extract social media trends relevant to your audience
    """)

    # Web Scraping Demo
    st.subheader("Try Web Scraping")
    
    url = st.text_input("Enter a website URL to extract content", "https://example.com")
    
    if st.button("Extract Content"):
        with st.spinner("Extracting content from the website..."):
            try:
                # Import the web scraping function
                import trafilatura
                
                # Fetch and extract the content
                downloaded = trafilatura.fetch_url(url)
                if downloaded:
                    text = trafilatura.extract(downloaded)
                    
                    # Display the extracted content
                    st.subheader("Extracted Content")
                    st.write(text[:1000] + "..." if len(text) > 1000 else text)
                    
                    # Option to download the full content
                    full_text = text
                    st.download_button(
                        label="Download Full Content",
                        data=full_text,
                        file_name="extracted_content.txt",
                        mime="text/plain"
                    )
                    
                    # Show data analysis options
                    st.subheader("Analyze This Content")
                    st.write("The extracted content can be used for various personalization analyses:")
                    
                    analysis_option = st.selectbox(
                        "Select analysis type",
                        ["Basic Text Statistics", "Sentiment Analysis", "Topic Extraction"]
                    )
                    
                    if analysis_option == "Basic Text Statistics":
                        # Simple text statistics
                        word_count = len(text.split())
                        sent_count = len(text.split('.'))
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Word Count", word_count)
                        with col2:
                            st.metric("Sentence Count", sent_count)
                        with col3:
                            st.metric("Avg. Words per Sentence", round(word_count/max(1, sent_count), 1))
                    
                    elif analysis_option == "Sentiment Analysis":
                        st.info("Sentiment analysis would provide emotion and tone insights from the content.")
                        
                    elif analysis_option == "Topic Extraction":
                        st.info("Topic extraction would identify key themes and subjects in the content.")
                else:
                    st.error("Could not download content from the provided URL.")
            except Exception as e:
                st.error(f"Error extracting content: {e}")

    # Web Data Application
    st.subheader("Example Applications")
    
    st.markdown("""
    ### Data Collection Use Cases
    
    Here are some examples of how web scraping can be applied for personalization:
    
    1. **Content Analysis**
       - Scrape articles from news sites to analyze trending topics
       - Extract product descriptions to enhance content-based recommendations
       - Gather social media content for sentiment analysis
       
    2. **Competitive Intelligence**
       - Monitor competitor pricing
       - Track product features across different websites
       - Analyze market trends and new product launches
       
    3. **User-Generated Content**
       - Collect reviews and ratings to understand user preferences
       - Gather social media mentions for brand sentiment analysis
       - Extract forum discussions to identify emerging user needs
    
    ### Getting Started with Web Data Collection
    
    1. Identify relevant data sources for your specific use case
    2. Use the web scraping tool to extract content
    3. Process and clean the extracted data
    4. Integrate the data into your personalization strategy
    """)
    
    # Add a divider
    st.markdown("---")

    # Create a web scraper utility function
    st.subheader("Web Scraper Utility")
    
    st.write("""
    This utility helps you create a reusable web scraper function that you can 
    incorporate into your data collection pipeline.
    """)
    
    code_sample = '''
# Example code for a web scraper function
import trafilatura

def scrape_website(url):
    """
    Extract main text content from a website URL.
    
    Parameters:
        url (str): The URL of the website to scrape
        
    Returns:
        str: The extracted text content
    """
    downloaded = trafilatura.fetch_url(url)
    if downloaded:
        return trafilatura.extract(downloaded)
    else:
        return "Could not download content from URL"
    '''
    
    st.code(code_sample, language="python")
    
    st.write("""
    #### How to Use This Function:
    
    ```python
    # Example usage
    url = "https://example.com/article"
    content = scrape_website(url)
    
    # Process the content
    # ...
    ```
    
    #### Important Considerations:
    
    - Always respect robots.txt and website terms of service
    - Add appropriate delays between requests to avoid overloading servers
    - Consider using an API if available instead of scraping
    - Be aware of legal and ethical considerations when scraping content
    """)
    
    # Add a divider for visual separation
    st.markdown("---")

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

        - **Select appropriate numberof segments**: Too few segments might not capture important differences, while too many can lead to overfitting
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