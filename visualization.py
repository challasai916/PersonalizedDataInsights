import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import io

# Set style for matplotlib
plt.style.use('seaborn-v0_8-whitegrid')
colors = list(mcolors.TABLEAU_COLORS.values())

def plot_numeric_distribution(data, feature):
    """
    Plot distribution of a numeric feature.
    
    Parameters:
        data (pd.DataFrame): Input data
        feature (str): Feature to plot
        
    Returns:
        matplotlib.figure.Figure: Figure with distribution plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram with KDE
    sns.histplot(data[feature].dropna(), kde=True, ax=ax, color=colors[0])
    
    # Add mean and median lines
    mean_val = data[feature].mean()
    median_val = data[feature].median()
    
    ax.axvline(mean_val, color=colors[1], linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    ax.axvline(median_val, color=colors[2], linestyle=':', linewidth=2, label=f'Median: {median_val:.2f}')
    
    # Add labels and title
    ax.set_xlabel(feature)
    ax.set_ylabel('Frequency')
    ax.set_title(f'Distribution of {feature}')
    ax.legend()
    
    plt.tight_layout()
    return fig

def plot_categorical_distribution(data, feature):
    """
    Plot distribution of a categorical feature.
    
    Parameters:
        data (pd.DataFrame): Input data
        feature (str): Feature to plot
        
    Returns:
        matplotlib.figure.Figure: Figure with distribution plot
    """
    # Get value counts and limit to top 15 categories if there are too many
    value_counts = data[feature].value_counts()
    if len(value_counts) > 15:
        value_counts = value_counts.head(15)
        title = f'Top 15 Categories for {feature}'
    else:
        title = f'Distribution of {feature}'
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Bar plot
    value_counts.plot(kind='bar', ax=ax, color=colors[0])
    
    # Add labels and title
    ax.set_xlabel(feature)
    ax.set_ylabel('Count')
    ax.set_title(title)
    
    # Add value labels on top of bars
    for i, v in enumerate(value_counts):
        ax.text(i, v + 0.1, str(v), ha='center')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def plot_boxplot(data, feature):
    """
    Create a box plot for a numeric feature.
    
    Parameters:
        data (pd.DataFrame): Input data
        feature (str): Feature to plot
        
    Returns:
        matplotlib.figure.Figure: Figure with box plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Box plot
    sns.boxplot(x=data[feature].dropna(), ax=ax, color=colors[0])
    
    # Add labels and title
    ax.set_xlabel(feature)
    ax.set_title(f'Box Plot of {feature}')
    
    plt.tight_layout()
    return fig

def identify_outliers(data, feature):
    """
    Identify outliers in a numeric feature using IQR method.
    
    Parameters:
        data (pd.DataFrame): Input data
        feature (str): Feature to analyze
        
    Returns:
        str: Summary of outliers
    """
    # Calculate IQR
    Q1 = data[feature].quantile(0.25)
    Q3 = data[feature].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define outlier boundaries
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Identify outliers
    outliers = data[(data[feature] < lower_bound) | (data[feature] > upper_bound)][feature]
    
    # Generate summary
    buffer = io.StringIO()
    buffer.write(f"Outlier Analysis for {feature}:\n")
    buffer.write(f"IQR: {IQR:.2f}\n")
    buffer.write(f"Lower bound: {lower_bound:.2f}\n")
    buffer.write(f"Upper bound: {upper_bound:.2f}\n")
    buffer.write(f"Number of outliers: {len(outliers)} ({(len(outliers) / len(data) * 100):.2f}% of data)\n")
    
    if len(outliers) > 0:
        buffer.write(f"Min outlier value: {outliers.min():.2f}\n")
        buffer.write(f"Max outlier value: {outliers.max():.2f}\n")
    
    return buffer.getvalue()

def plot_numeric_by_categorical(data, numeric_feature, categorical_feature):
    """
    Plot a numeric feature grouped by a categorical feature.
    
    Parameters:
        data (pd.DataFrame): Input data
        numeric_feature (str): Numeric feature to plot
        categorical_feature (str): Categorical feature to group by
        
    Returns:
        matplotlib.figure.Figure: Figure with grouped plot
    """
    # Get value counts and limit to top 10 categories if there are too many
    top_categories = data[categorical_feature].value_counts().head(10).index
    filtered_data = data[data[categorical_feature].isin(top_categories)]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Box plot
    sns.boxplot(x=categorical_feature, y=numeric_feature, data=filtered_data, ax=ax, palette='Set3')
    
    # Add labels and title
    ax.set_xlabel(categorical_feature)
    ax.set_ylabel(numeric_feature)
    ax.set_title(f'{numeric_feature} by {categorical_feature}')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def plot_categorical_relationship(data, feature1, feature2):
    """
    Plot relationship between two categorical features using a heatmap.
    
    Parameters:
        data (pd.DataFrame): Input data
        feature1 (str): First categorical feature
        feature2 (str): Second categorical feature
        
    Returns:
        matplotlib.figure.Figure: Figure with heatmap
    """
    # Create cross-tabulation
    cross_tab = pd.crosstab(data[feature1], data[feature2], normalize='index')
    
    # Limit to top categories if there are too many
    if cross_tab.shape[0] > 10 or cross_tab.shape[1] > 10:
        # Get top 10 categories for each feature
        top_feature1 = data[feature1].value_counts().head(10).index
        top_feature2 = data[feature2].value_counts().head(10).index
        
        # Filter crosstab
        cross_tab = cross_tab.loc[cross_tab.index.isin(top_feature1), 
                               cross_tab.columns.isin(top_feature2)]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Heatmap
    sns.heatmap(cross_tab, annot=True, cmap='YlGnBu', fmt='.2f', ax=ax)
    
    # Add labels and title
    ax.set_xlabel(feature2)
    ax.set_ylabel(feature1)
    ax.set_title(f'Relationship between {feature1} and {feature2}')
    
    plt.tight_layout()
    return fig

def calculate_engagement_metrics(data, user_id_col, engagement_metrics):
    """
    Calculate engagement metrics for each user.
    
    Parameters:
        data (pd.DataFrame): Input data
        user_id_col (str): Column name for user ID
        engagement_metrics (list): List of columns to use as engagement metrics
        
    Returns:
        pd.DataFrame: DataFrame with engagement metrics by user
    """
    # Group by user ID and calculate metrics
    engagement_df = data.groupby(user_id_col)[engagement_metrics].sum().reset_index()
    
    # Calculate overall engagement score
    # First, normalize each metric
    for metric in engagement_metrics:
        max_val = engagement_df[metric].max()
        if max_val > 0:  # Avoid division by zero
            engagement_df[f'{metric}_normalized'] = engagement_df[metric] / max_val
    
    # Calculate engagement score as average of normalized metrics
    normalized_cols = [f'{metric}_normalized' for metric in engagement_metrics]
    engagement_df['engagement_score'] = engagement_df[normalized_cols].mean(axis=1)
    
    # Sort by engagement score
    engagement_df = engagement_df.sort_values('engagement_score', ascending=False)
    
    return engagement_df

def plot_time_series(data, time_col, metric, agg_level='day'):
    """
    Plot time series data.
    
    Parameters:
        data (pd.DataFrame): Input data
        time_col (str): Column name for timestamp
        metric (str): Column name for metric to plot
        agg_level (str): Aggregation level ('day', 'week', 'month')
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure with time series plot
    """
    # Ensure timestamp column is datetime
    if data[time_col].dtype != 'datetime64[ns]':
        data[time_col] = pd.to_datetime(data[time_col])
    
    # Create copy of data
    df = data.copy()
    
    # Create time period column based on aggregation level
    if agg_level == 'day':
        df['time_period'] = df[time_col].dt.date
    elif agg_level == 'week':
        df['time_period'] = df[time_col].dt.isocalendar().week
    elif agg_level == 'month':
        df['time_period'] = df[time_col].dt.to_period('M').astype(str)
    
    # Aggregate data
    agg_data = df.groupby('time_period')[metric].agg(['mean', 'sum', 'count']).reset_index()
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Add traces for mean, sum, and count
    fig.add_trace(go.Scatter(
        x=agg_data['time_period'],
        y=agg_data['mean'],
        mode='lines+markers',
        name=f'Mean {metric}'
    ))
    
    fig.add_trace(go.Scatter(
        x=agg_data['time_period'],
        y=agg_data['sum'],
        mode='lines+markers',
        name=f'Sum {metric}',
        visible='legendonly'  # Hide by default
    ))
    
    fig.add_trace(go.Scatter(
        x=agg_data['time_period'],
        y=agg_data['count'],
        mode='lines+markers',
        name=f'Count',
        visible='legendonly'  # Hide by default
    ))
    
    # Update layout
    fig.update_layout(
        title=f'{metric} Over Time (by {agg_level})',
        xaxis_title=f'Time ({agg_level})',
        yaxis_title=f'{metric}',
        hovermode='x unified'
    )
    
    return fig

def plot_time_heatmap(data, time_col):
    """
    Create a heatmap of activity by day of week and hour of day.
    
    Parameters:
        data (pd.DataFrame): Input data
        time_col (str): Column name for timestamp
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure with heatmap
    """
    # Ensure timestamp column is datetime
    if data[time_col].dtype != 'datetime64[ns]':
        data[time_col] = pd.to_datetime(data[time_col])
    
    # Create copy of data
    df = data.copy()
    
    # Extract day of week and hour
    df['day_of_week'] = df[time_col].dt.day_name()
    df['hour_of_day'] = df[time_col].dt.hour
    
    # Count occurrences for each day-hour combination
    heatmap_data = df.groupby(['day_of_week', 'hour_of_day']).size().reset_index(name='count')
    
    # Convert to pivot table for heatmap
    pivot_data = heatmap_data.pivot(index='day_of_week', columns='hour_of_day', values='count')
    
    # Ensure proper day of week order
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot_data = pivot_data.reindex(day_order)
    
    # Create Plotly heatmap
    fig = px.imshow(
        pivot_data,
        labels=dict(x="Hour of Day", y="Day of Week", color="Activity Count"),
        x=list(range(24)),
        y=day_order,
        color_continuous_scale='Viridis',
        title='Activity Heatmap by Day and Hour'
    )
    
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(24)),
            ticktext=[f'{h}:00' for h in range(24)]
        )
    )
    
    return fig

def perform_clustering(data, features, n_clusters):
    """
    Perform clustering on the data using selected features.
    
    Parameters:
        data (pd.DataFrame): Input data
        features (list): List of features to use for clustering
        n_clusters (int): Number of clusters
        
    Returns:
        pd.DataFrame: Data with cluster assignments
        np.ndarray: Cluster centers
        float: Silhouette score
    """
    # Create copy of data
    df = data.copy()
    
    # Select only the specified features
    X = df[features].copy()
    
    # Handle missing values
    for col in X.columns:
        X[col] = X[col].fillna(X[col].median())
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Get cluster centers and convert back to original scale
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(X_scaled, df['cluster'])
    
    return df, cluster_centers, silhouette_avg

def plot_clusters_2d(cluster_data, features, method='pca'):
    """
    Create a 2D visualization of clusters.
    
    Parameters:
        cluster_data (pd.DataFrame): Data with cluster assignments
        features (list): Features used for clustering
        method (str): Dimensionality reduction method ('pca' or 'select')
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure with cluster visualization
    """
    # Select data for visualization
    X = cluster_data[features].copy()
    
    # Handle missing values
    for col in X.columns:
        X[col] = X[col].fillna(X[col].median())
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Reduce to 2D for visualization
    if method == 'pca' or len(features) > 2:
        # Use PCA to reduce to 2D
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X_scaled)
        
        # Create visualization data
        vis_df = pd.DataFrame({
            'x': X_2d[:, 0],
            'y': X_2d[:, 1],
            'cluster': cluster_data['cluster']
        })
        
        # Create scatter plot
        fig = px.scatter(
            vis_df, x='x', y='y', color='cluster',
            color_continuous_scale='Viridis',
            title='Cluster Visualization (PCA)',
            labels={'x': 'PCA Component 1', 'y': 'PCA Component 2'}
        )
    else:
        # Use the first two features directly
        vis_df = pd.DataFrame({
            'x': X[features[0]],
            'y': X[features[1]],
            'cluster': cluster_data['cluster']
        })
        
        # Create scatter plot
        fig = px.scatter(
            vis_df, x='x', y='y', color='cluster',
            color_continuous_scale='Viridis',
            title=f'Cluster Visualization: {features[0]} vs {features[1]}',
            labels={'x': features[0], 'y': features[1]}
        )
    
    fig.update_layout(
        legend_title_text='Cluster'
    )
    
    return fig

def get_cluster_profiles(cluster_data, features):
    """
    Generate profiles for each cluster.
    
    Parameters:
        cluster_data (pd.DataFrame): Data with cluster assignments
        features (list): Features used for clustering
        
    Returns:
        pd.DataFrame: Cluster profiles
    """
    # Group by cluster and calculate statistics for each feature
    profiles = cluster_data.groupby('cluster')[features].agg(['mean', 'median', 'std', 'count'])
    
    # Calculate cluster sizes and percentages
    cluster_sizes = cluster_data['cluster'].value_counts().sort_index()
    total_size = len(cluster_data)
    
    cluster_pcts = cluster_sizes / total_size * 100
    
    # Create a summary DataFrame
    summary = pd.DataFrame({
        'Size': cluster_sizes,
        'Percentage': cluster_pcts
    })
    
    # Add feature means for easier comparison
    for feature in features:
        summary[feature] = cluster_data.groupby('cluster')[feature].mean()
    
    return summary

def plot_cluster_radar(cluster_centers, features):
    """
    Create a radar chart comparing cluster profiles.
    
    Parameters:
        cluster_centers (np.ndarray): Cluster centers
        features (list): Features used for clustering
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure with radar chart
    """
    # Number of clusters and features
    n_clusters = cluster_centers.shape[0]
    
    # Normalize values for radar chart (0-1 scale)
    centers_df = pd.DataFrame(cluster_centers, columns=features)
    
    # Min-max scaling for each feature
    for feature in features:
        min_val = centers_df[feature].min()
        max_val = centers_df[feature].max()
        if max_val > min_val:
            centers_df[feature] = (centers_df[feature] - min_val) / (max_val - min_val)
    
    # Create radar chart
    fig = go.Figure()
    
    for i in range(n_clusters):
        fig.add_trace(go.Scatterpolar(
            r=centers_df.iloc[i].values,
            theta=features,
            fill='toself',
            name=f'Cluster {i}'
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Cluster Profiles Comparison"
    )
    
    return fig

def plot_segment_comparison(segmented_data, features):
    """
    Compare segments/clusters across different features.
    
    Parameters:
        segmented_data (pd.DataFrame): Data with segment/cluster assignments
        features (list): Features to compare
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure with segment comparison
    """
    # Determine segment column
    segment_col = 'segment' if 'segment' in segmented_data.columns else 'cluster'
    
    # Create a long-form DataFrame for easier plotting
    plot_data = []
    
    for feature in features:
        feature_data = segmented_data.groupby(segment_col)[feature].mean().reset_index()
        feature_data['Feature'] = feature
        feature_data.rename(columns={feature: 'Value'}, inplace=True)
        plot_data.append(feature_data)
    
    plot_df = pd.concat(plot_data)
    
    # Create comparison plot
    fig = px.bar(
        plot_df, 
        x='Feature', 
        y='Value', 
        color=segment_col,
        barmode='group',
        title='Segment Comparison by Feature',
        labels={'Value': 'Average Value'}
    )
    
    fig.update_layout(
        xaxis_title='Feature',
        yaxis_title='Average Value',
        legend_title=segment_col.capitalize()
    )
    
    return fig
