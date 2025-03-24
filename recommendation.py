import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split as surprise_train_test_split
from surprise import accuracy

def collaborative_filtering_recommendations(data, user_col, item_col, rating_col, n_factors=20, n_recommendations=5):
    """
    Generate recommendations using collaborative filtering.
    
    Parameters:
        data (pd.DataFrame): DataFrame with user-item interactions
        user_col (str): Column name for user IDs
        item_col (str): Column name for item IDs
        rating_col (str): Column name for ratings
        n_factors (int): Number of latent factors to use in the model
        n_recommendations (int): Number of recommendations to generate per user
        
    Returns:
        dict: Dictionary of recommendations for each user
        dict: Model evaluation metrics
    """
    # Create a copy of the data
    df = data.copy()
    
    # Prepare the data for Surprise
    reader = Reader(rating_scale=(df[rating_col].min(), df[rating_col].max()))
    data = Dataset.load_from_df(df[[user_col, item_col, rating_col]], reader)
    
    # Take a smaller sample for faster training
    raw_trainset = data.build_full_trainset()
    sample_size = min(10000, len(list(raw_trainset.all_ratings())))
    
    # Use a smaller test size and fewer iterations for faster training
    trainset, testset = surprise_train_test_split(data, test_size=0.1, random_state=42)
    
    # Train the model with fewer epochs
    model = SVD(n_factors=min(n_factors, 10), n_epochs=5, random_state=42)
    model.fit(trainset)
    
    # Evaluate the model
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions)
    mae = accuracy.mae(predictions)
    
    # Generate recommendations for all users
    all_users = df[user_col].unique()
    all_items = df[item_col].unique()
    
    # Get user-item pairs that already exist in the data
    existing_interactions = set(zip(df[user_col], df[item_col]))
    
    # Generate recommendations
    user_recommendations = {}
    
    for user in all_users:
        # Find items the user hasn't interacted with
        user_items = df[df[user_col] == user][item_col].unique()
        new_items = [item for item in all_items if item not in user_items]
        
        # If user has interacted with all items, use existing items
        if not new_items:
            new_items = all_items
        
        # Predict ratings for new items
        item_preds = [(item, model.predict(user, item).est) for item in new_items]
        
        # Sort by predicted rating and get top N
        top_recommendations = sorted(item_preds, key=lambda x: x[1], reverse=True)[:n_recommendations]
        
        user_recommendations[user] = top_recommendations
    
    # Model evaluation
    evaluation = {
        'rmse': rmse,
        'mae': mae
    }
    
    return user_recommendations, evaluation

def content_based_recommendations(data, item_col, feature_cols, n_recommendations=5):
    """
    Generate content-based recommendations.
    
    Parameters:
        data (pd.DataFrame): Input data
        item_col (str): Column name for item IDs
        feature_cols (list): List of feature columns to use
        n_recommendations (int): Number of similar items to recommend
        
    Returns:
        dict: Dictionary of similar items for each item
        pd.DataFrame: Item features matrix
    """
    # Create a copy of the data
    df = data.copy()
    
    # Create item-feature matrix
    item_features = df.groupby(item_col)[feature_cols].mean().reset_index()
    
    # Set item_col as index
    item_features = item_features.set_index(item_col)
    
    # Handle missing values
    for col in feature_cols:
        item_features[col] = item_features[col].fillna(0)
    
    # Normalize features
    scaler = StandardScaler()
    item_features_scaled = pd.DataFrame(
        scaler.fit_transform(item_features),
        index=item_features.index,
        columns=item_features.columns
    )
    
    # Calculate similarity between items
    similarity_matrix = cosine_similarity(item_features_scaled)
    
    # Convert to DataFrame for easier indexing
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=item_features.index,
        columns=item_features.index
    )
    
    # Generate recommendations for each item
    item_similarities = {}
    
    for item in item_features.index:
        # Get similarity scores
        similar_items = similarity_df[item].sort_values(ascending=False)
        
        # Exclude the item itself
        similar_items = similar_items.drop(item, errors='ignore')
        
        # Get top N similar items
        top_similar = similar_items.head(n_recommendations)
        
        # Store as (item, similarity) tuples
        item_similarities[item] = [(idx, score) for idx, score in top_similar.items()]
    
    return item_similarities, item_features

def segment_based_recommendations(data, segment_col, user_col, item_col, interaction_col, n_recommendations=5):
    """
    Generate recommendations based on user segments.
    
    Parameters:
        data (pd.DataFrame): Segmented data
        segment_col (str): Column name for segment
        user_col (str): Column name for user ID
        item_col (str): Column name for item ID
        interaction_col (str): Column name for interaction value
        n_recommendations (int): Number of top items to recommend per segment
        
    Returns:
        dict: Dictionary of top items for each segment
    """
    # Create a copy of the data
    df = data.copy()
    
    # Calculate average interaction by segment and item
    segment_item_scores = df.groupby([segment_col, item_col])[interaction_col].agg(['mean', 'count']).reset_index()
    
    # Only consider items with sufficient interactions (at least 5)
    segment_item_scores = segment_item_scores[segment_item_scores['count'] >= 5]
    
    # Rank items within each segment
    segment_recommendations = {}
    
    for segment in df[segment_col].unique():
        # Get items for this segment
        segment_items = segment_item_scores[segment_item_scores[segment_col] == segment]
        
        # Sort by mean interaction value
        top_items = segment_items.sort_values('mean', ascending=False).head(n_recommendations)
        
        # Store as (item, score) tuples
        segment_recommendations[segment] = [(item, score) for item, score in zip(top_items[item_col], top_items['mean'])]
    
    return segment_recommendations

def popular_items_recommendations(data, item_col, interaction_col, n_recommendations=10):
    """
    Generate recommendations based on item popularity.
    
    Parameters:
        data (pd.DataFrame): Input data
        item_col (str): Column name for item ID
        interaction_col (str): Column name for interaction value
        n_recommendations (int): Number of popular items to recommend
        
    Returns:
        list: List of (item, score) tuples for the most popular items
    """
    # Create a copy of the data
    df = data.copy()
    
    # Calculate item popularity (count and average interaction value)
    item_popularity = df.groupby(item_col)[interaction_col].agg(['count', 'mean']).reset_index()
    
    # Create a combined popularity score
    # Normalize count and mean
    count_max = item_popularity['count'].max()
    mean_max = item_popularity['mean'].max()
    
    if count_max > 0 and mean_max > 0:
        item_popularity['count_norm'] = item_popularity['count'] / count_max
        item_popularity['mean_norm'] = item_popularity['mean'] / mean_max
        
        # Popularity score is a weighted combination of normalized count and mean
        item_popularity['popularity_score'] = 0.7 * item_popularity['count_norm'] + 0.3 * item_popularity['mean_norm']
    else:
        item_popularity['popularity_score'] = item_popularity['count']
    
    # Sort by popularity score
    top_items = item_popularity.sort_values('popularity_score', ascending=False).head(n_recommendations)
    
    # Return as (item, score) tuples
    popular_items = [(item, score) for item, score in zip(top_items[item_col], top_items['popularity_score'])]
    
    return popular_items

def evaluate_recommendations(data, recommendations, user_col, item_col, rating_col, metrics=['Precision', 'Recall']):
    """
    Evaluate recommendation performance.
    
    Parameters:
        data (pd.DataFrame): Original data with user-item interactions
        recommendations (dict): Dictionary of recommendations
        user_col (str): Column name for user ID
        item_col (str): Column name for item ID
        rating_col (str): Column name for rating/interaction
        metrics (list): List of metrics to calculate
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Create a copy of the data
    df = data.copy()
    
    # Define a threshold for "relevant" items (e.g., items with rating > 3.5)
    relevance_threshold = df[rating_col].mean()
    
    # Create a set of "ground truth" relevant items for each user
    user_relevant_items = {}
    
    for user, group in df.groupby(user_col):
        # Consider items with rating above threshold as relevant
        relevant_items = set(group[group[rating_col] >= relevance_threshold][item_col])
        user_relevant_items[user] = relevant_items
    
    # Initialize evaluation metrics
    eval_results = {}
    
    # Calculate metrics
    if 'Precision' in metrics:
        precision_sum = 0
        precision_count = 0
        
        for user, user_recs in recommendations.items():
            # Skip users without ground truth data
            if user not in user_relevant_items:
                continue
                
            # Get recommended items
            rec_items = [item for item, _ in user_recs]
            
            # Get relevant items
            relevant_items = user_relevant_items[user]
            
            # Calculate precision
            if len(rec_items) > 0:
                precision = len(set(rec_items) & relevant_items) / len(rec_items)
                precision_sum += precision
                precision_count += 1
        
        # Average precision
        eval_results['Precision'] = precision_sum / max(1, precision_count)
    
    if 'Recall' in metrics:
        recall_sum = 0
        recall_count = 0
        
        for user, user_recs in recommendations.items():
            # Skip users without ground truth data
            if user not in user_relevant_items:
                continue
                
            # Get recommended items
            rec_items = [item for item, _ in user_recs]
            
            # Get relevant items
            relevant_items = user_relevant_items[user]
            
            # Calculate recall
            if len(relevant_items) > 0:
                recall = len(set(rec_items) & relevant_items) / len(relevant_items)
                recall_sum += recall
                recall_count += 1
        
        # Average recall
        eval_results['Recall'] = recall_sum / max(1, recall_count)
    
    if 'NDCG' in metrics:
        ndcg_sum = 0
        ndcg_count = 0
        
        for user, user_recs in recommendations.items():
            # Skip users without ground truth data
            if user not in user_relevant_items:
                continue
                
            # Get recommended items and their scores
            rec_items = [item for item, _ in user_recs]
            
            # Get relevant items
            relevant_items = user_relevant_items[user]
            
            # Calculate DCG
            dcg = 0
            for i, item in enumerate(rec_items):
                if item in relevant_items:
                    # Use binary relevance (1 if relevant, 0 otherwise)
                    # Position is i+1 (1-indexed)
                    dcg += 1 / np.log2(i + 2)
            
            # Calculate IDCG (ideal DCG)
            idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant_items), len(rec_items))))
            
            # Calculate NDCG
            if idcg > 0:
                ndcg = dcg / idcg
                ndcg_sum += ndcg
                ndcg_count += 1
        
        # Average NDCG
        eval_results['NDCG'] = ndcg_sum / max(1, ndcg_count)
    
    if 'Diversity' in metrics:
        # Calculate diversity as average pairwise dissimilarity of recommended items
        diversity_sum = 0
        diversity_count = 0
        
        for user, user_recs in recommendations.items():
            # Get recommended items
            rec_items = [item for item, _ in user_recs]
            
            # If less than 2 items, diversity is not applicable
            if len(rec_items) < 2:
                continue
            
            # Get item features if available
            if 'content_based' in recommendations and hasattr(evaluate_recommendations, 'item_features'):
                # Use content-based similarity
                item_features = evaluate_recommendations.item_features
                
                # Calculate pairwise dissimilarity
                dissimilarity_sum = 0
                pair_count = 0
                
                for i in range(len(rec_items)):
                    for j in range(i+1, len(rec_items)):
                        item_i, item_j = rec_items[i], rec_items[j]
                        
                        if item_i in item_features.index and item_j in item_features.index:
                            # Calculate cosine similarity
                            sim = cosine_similarity(
                                [item_features.loc[item_i].values],
                                [item_features.loc[item_j].values]
                            )[0][0]
                            
                            # Dissimilarity = 1 - similarity
                            dissimilarity_sum += (1 - sim)
                            pair_count += 1
                
                if pair_count > 0:
                    diversity = dissimilarity_sum / pair_count
                    diversity_sum += diversity
                    diversity_count += 1
            else:
                # Use item IDs as proxy for diversity
                # More unique items = higher diversity
                diversity = len(set(rec_items)) / len(rec_items)
                diversity_sum += diversity
                diversity_count += 1
        
        # Average diversity
        eval_results['Diversity'] = diversity_sum / max(1, diversity_count)
    
    if 'Coverage' in metrics:
        # Calculate coverage as percentage of items that are recommended at least once
        all_items = set(df[item_col].unique())
        recommended_items = set()
        
        for user, user_recs in recommendations.items():
            recommended_items.update([item for item, _ in user_recs])
        
        # Coverage
        eval_results['Coverage'] = len(recommended_items) / max(1, len(all_items))
    
    return eval_results

def evaluate_by_segment(data, segmented_data, segment_col, recommendations, user_col, item_col, rating_col, metrics):
    """
    Evaluate recommendations by user segment.
    
    Parameters:
        data (pd.DataFrame): Original data with user-item interactions
        segmented_data (pd.DataFrame): Data with user segment information
        segment_col (str): Column name for segment
        recommendations (dict): Dictionary of recommendations
        user_col (str): Column name for user ID
        item_col (str): Column name for item ID
        rating_col (str): Column name for rating/interaction
        metrics (list): List of metrics to calculate
        
    Returns:
        dict: Dictionary of evaluation metrics by segment
    """
    # Create a mapping of user to segment
    user_segments = dict(zip(segmented_data[user_col], segmented_data[segment_col]))
    
    # Group recommendations by segment
    segment_recommendations = {}
    
    for user, user_recs in recommendations.items():
        if user in user_segments:
            segment = user_segments[user]
            
            if segment not in segment_recommendations:
                segment_recommendations[segment] = {}
                
            segment_recommendations[segment][user] = user_recs
    
    # Evaluate for each segment
    segment_eval = {}
    
    for segment, segment_recs in segment_recommendations.items():
        # Get users in this segment
        segment_users = [user for user, seg in user_segments.items() if seg == segment]
        
        # Filter data to only include users in this segment
        segment_data = data[data[user_col].isin(segment_users)]
        
        # Evaluate recommendations for this segment
        segment_eval[segment] = evaluate_recommendations(
            segment_data, segment_recs, user_col, item_col, rating_col, metrics)
    
    return segment_eval

def simulate_ab_test(algorithm_a, algorithm_b, sample_size, success_metric):
    """
    Simulate an A/B test comparing two recommendation algorithms or strategies.
    
    Parameters:
        algorithm_a (str): Name or description of algorithm A (control)
        algorithm_b (str): Name or description of algorithm B (treatment)
        sample_size (int): Number of simulated users
        success_metric (str): Metric name (e.g., 'Click-through rate')
        
    Returns:
        dict: A/B test results and statistics
    """
    # Set up parameters based on algorithm types
    # These are hypothetical performance characteristics
    if algorithm_a == "Collaborative Filtering":
        control_mean = 0.12  # Example CTR for collaborative filtering
        control_std = 0.04
    elif algorithm_a == "Content-Based":
        control_mean = 0.10  # Example CTR for content-based
        control_std = 0.03
    elif algorithm_a == "Popularity-Based":
        control_mean = 0.08  # Example CTR for popularity-based
        control_std = 0.02
    elif algorithm_a == "No personalization":
        control_mean = 0.05  # Example CTR for no personalization
        control_std = 0.01
    elif algorithm_a == "Segment-based personalization":
        control_mean = 0.09  # Example CTR for segment-based
        control_std = 0.03
    elif algorithm_a == "Individual personalization":
        control_mean = 0.14  # Example CTR for individual personalization
        control_std = 0.05
    else:
        control_mean = 0.10
        control_std = 0.03
    
    if algorithm_b == "Collaborative Filtering":
        treatment_mean = 0.12  # Example CTR for collaborative filtering
        treatment_std = 0.04
    elif algorithm_b == "Content-Based":
        treatment_mean = 0.10  # Example CTR for content-based
        treatment_std = 0.03
    elif algorithm_b == "Popularity-Based":
        treatment_mean = 0.08  # Example CTR for popularity-based
        treatment_std = 0.02
    elif algorithm_b == "No personalization":
        treatment_mean = 0.05  # Example CTR for no personalization
        treatment_std = 0.01
    elif algorithm_b == "Segment-based personalization":
        treatment_mean = 0.09  # Example CTR for segment-based
        treatment_std = 0.03
    elif algorithm_b == "Individual personalization":
        treatment_mean = 0.14  # Example CTR for individual personalization
        treatment_std = 0.05
    else:
        treatment_mean = 0.11  # Default is 10% improvement
        treatment_std = 0.03
    
    # Adjust values based on success metric
    if success_metric == "Conversion rate":
        control_mean /= 4  # Conversion rates are typically lower than CTR
        treatment_mean /= 4
        control_std /= 4
        treatment_std /= 4
    elif success_metric == "Average order value":
        control_mean *= 50  # AOV in dollars
        treatment_mean *= 50
        control_std *= 20
        treatment_std *= 20
    elif success_metric == "User engagement":
        control_mean *= 10  # Engagement time in minutes
        treatment_mean *= 10
        control_std *= 5
        treatment_std *= 5
    elif success_metric == "User engagement time":
        control_mean *= 15  # Time in minutes
        treatment_mean *= 15
        control_std *= 7
        treatment_std *= 7
    elif success_metric == "Retention":
        control_mean *= 4  # Retention percentage
        treatment_mean *= 4
        control_std *= 1
        treatment_std *= 1
    
    # Add a slight improvement for algorithm B (treatment)
    # If the means are the same, add a small improvement
    if abs(treatment_mean - control_mean) < 0.001:
        treatment_mean *= 1.1  # 10% improvement
    
    # Generate simulated data for control and treatment groups
    np.random.seed(42)  # For reproducibility
    control_samples = np.random.normal(control_mean, control_std, sample_size)
    treatment_samples = np.random.normal(treatment_mean, treatment_std, sample_size)
    
    # Ensure values make sense for the given metric
    if success_metric in ["Click-through rate", "Conversion rate", "Retention"]:
        # These are percentages/rates between 0 and 1
        control_samples = np.clip(control_samples, 0, 1)
        treatment_samples = np.clip(treatment_samples, 0, 1)
    elif success_metric in ["Average order value", "User engagement", "User engagement time"]:
        # These should be positive values
        control_samples = np.maximum(control_samples, 0)
        treatment_samples = np.maximum(treatment_samples, 0)
    
    # Calculate statistics
    control_rate = control_samples.mean()
    treatment_rate = treatment_samples.mean()
    
    # Absolute and relative difference
    abs_diff = treatment_rate - control_rate
    rel_diff = abs_diff / control_rate if control_rate > 0 else 0
    
    # Statistical significance test (t-test)
    t_stat, p_value = stats.ttest_ind(treatment_samples, control_samples)
    
    # Calculate confidence interval for the difference
    # 95% confidence level (alpha = 0.05)
    alpha = 0.05
    df = 2 * sample_size - 2  # degrees of freedom
    
    # Standard error of the difference
    se_diff = np.sqrt((control_std**2 / sample_size) + (treatment_std**2 / sample_size))
    
    # Critical value from t-distribution
    t_crit = stats.t.ppf(1 - alpha/2, df)
    
    # Confidence interval
    ci_lower = abs_diff - t_crit * se_diff
    ci_upper = abs_diff + t_crit * se_diff
    
    # Return results
    return {
        'control_rate': control_rate,
        'treatment_rate': treatment_rate,
        'absolute_difference': abs_diff,
        'relative_difference': rel_diff,
        'lift': rel_diff,  # Alias for relative difference
        'p_value': p_value,
        'significant': p_value < 0.05,
        'confidence_interval': (ci_lower, ci_upper),
        'control_samples': control_samples,
        'treatment_samples': treatment_samples,
        'control_mean': control_mean,
        'treatment_mean': treatment_mean,
        'control_std': control_std,
        'treatment_std': treatment_std,
        'sample_size': sample_size
    }

def analyze_ab_test_results(data, group_col, metric_col, control_value, treatment_value, user_col=None):
    """
    Analyze uploaded A/B test results.
    
    Parameters:
        data (pd.DataFrame): A/B test data
        group_col (str): Column identifying test groups
        metric_col (str): Column with the metric values
        control_value: Value in group_col that identifies the control group
        treatment_value: Value in group_col that identifies the treatment group
        user_col (str, optional): Column with user IDs
        
    Returns:
        dict: A/B test analysis results
    """
    # Filter data for control and treatment groups
    control_data = data[data[group_col] == control_value][metric_col]
    treatment_data = data[data[group_col] == treatment_value][metric_col]
    
    # Calculate statistics
    control_mean = control_data.mean()
    treatment_mean = treatment_data.mean()
    
    control_size = len(control_data)
    treatment_size = len(treatment_data)
    
    # Standard errors
    control_se = control_data.std() / np.sqrt(control_size) if control_size > 0 else 0
    treatment_se = treatment_data.std() / np.sqrt(treatment_size) if treatment_size > 0 else 0
    
    # Absolute and relative difference
    abs_diff = treatment_mean - control_mean
    rel_diff = abs_diff / control_mean if control_mean > 0 else 0
    
    # Statistical significance test (t-test)
    t_stat, p_value = stats.ttest_ind(treatment_data, control_data)
    
    # Calculate confidence interval for the difference
    # 95% confidence level (alpha = 0.05)
    alpha = 0.05
    df = control_size + treatment_size - 2  # degrees of freedom
    
    # Standard error of the difference
    se_diff = np.sqrt((control_data.std()**2 / control_size) + 
                     (treatment_data.std()**2 / treatment_size))
    
    # Critical value from t-distribution
    t_crit = stats.t.ppf(1 - alpha/2, df)
    
    # Confidence interval
    ci_lower = abs_diff - t_crit * se_diff
    ci_upper = abs_diff + t_crit * se_diff
    
    # Return results
    return {
        'control_mean': control_mean,
        'treatment_mean': treatment_mean,
        'control_size': control_size,
        'treatment_size': treatment_size,
        'control_se': control_se,
        'treatment_se': treatment_se,
        'absolute_difference': abs_diff,
        'relative_difference': rel_diff,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }
