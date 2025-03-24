import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

def preprocess_user_data(df):
    """
    Preprocess user data for analysis.
    
    Parameters:
    df (DataFrame): Raw user data
    
    Returns:
    DataFrame: Processed data
    dict: Summary statistics
    """
    if df is None or df.empty:
        st.error("No data available for processing.")
        return None, None
    
    # Create a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Collect summary stats
    summary = {
        "row_count": len(processed_df),
        "column_count": len(processed_df.columns),
        "missing_values": processed_df.isna().sum().sum(),
        "duplicates": processed_df.duplicated().sum()
    }
    
    # Handle common tasks
    
    # 1. Convert timestamp columns if present
    time_columns = [col for col in processed_df.columns if 'time' in col.lower() 
                    or 'date' in col.lower() or col.lower().endswith('at')]
    
    for col in time_columns:
        if processed_df[col].dtype == 'object':
            try:
                processed_df[col] = pd.to_datetime(processed_df[col])
                summary[f"{col}_min"] = processed_df[col].min()
                summary[f"{col}_max"] = processed_df[col].max()
            except:
                pass  # Skip if conversion fails
    
    # 2. Handle missing values
    numeric_cols = processed_df.select_dtypes(include=['number']).columns
    processed_df[numeric_cols] = processed_df[numeric_cols].fillna(processed_df[numeric_cols].median())
    
    categorical_cols = processed_df.select_dtypes(include=['object']).columns
    processed_df[categorical_cols] = processed_df[categorical_cols].fillna('Unknown')
    
    # 3. Remove duplicates if any
    if summary["duplicates"] > 0:
        processed_df = processed_df.drop_duplicates()
        summary["after_dedup_count"] = len(processed_df)
    
    # 4. Detect user ID column if present
    id_columns = [col for col in processed_df.columns if 'id' in col.lower() 
                  or 'user' in col.lower() or '_id' in col.lower()]
    if id_columns:
        summary["potential_user_id_columns"] = id_columns
    
    # 5. Detect behavioral features
    behavioral_cols = [col for col in processed_df.columns if any(term in col.lower() 
                      for term in ['click', 'view', 'purchase', 'rating', 'like', 'share', 
                                   'comment', 'time', 'duration', 'count'])]
    if behavioral_cols:
        summary["behavioral_columns"] = behavioral_cols
    
    return processed_df, summary


def prepare_for_recommendation(df):
    """
    Prepare data for recommendation system.
    
    Parameters:
    df (DataFrame): Processed user data
    
    Returns:
    DataFrame: Data ready for recommendation algorithms
    dict: Mapping dictionaries and metadata
    """
    if df is None or df.empty:
        st.error("No data available for recommendations.")
        return None, None
    
    # Try to identify user, item, and rating columns
    columns = df.columns
    user_col = None
    item_col = None
    rating_col = None
    
    # Look for user column
    for col in columns:
        if 'user' in col.lower() or col.lower() == 'uid':
            user_col = col
            break
    
    # Look for item column
    for col in columns:
        if any(term in col.lower() for term in ['item', 'product', 'movie', 'book', 'article']):
            item_col = col
            break
    
    # Look for rating column
    for col in columns:
        if any(term in col.lower() for term in ['rating', 'score', 'preference', 'grade']):
            if df[col].dtype in ['int64', 'float64']:
                rating_col = col
                break
    
    # If we couldn't find the expected columns, try to make educated guesses
    if user_col is None or item_col is None:
        # Try to identify columns based on cardinality and data types
        object_cols = df.select_dtypes(include=['object']).columns
        id_cols = [col for col in columns if 'id' in col.lower()]
        
        if not user_col and id_cols:
            user_col = id_cols[0]  # Take first ID column as user
        
        if not item_col and len(object_cols) > 0:
            # Take an object column that's not the user_col
            for col in object_cols:
                if col != user_col:
                    item_col = col
                    break
    
    # If we still don't have a rating column, try to find a numeric column
    if rating_col is None:
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if col not in [user_col, item_col]:
                rating_col = col
                break
    
    # If we still don't have all required columns, create a binary interaction
    if user_col and item_col and not rating_col:
        df['interaction'] = 1
        rating_col = 'interaction'
    
    # If we identified the necessary columns, prepare the data
    if user_col and item_col and rating_col:
        # Create a dataset with only the necessary columns
        rec_df = df[[user_col, item_col, rating_col]].copy()
        
        # Create mappings for user and item IDs
        user_mapping = {id: i for i, id in enumerate(rec_df[user_col].unique())}
        item_mapping = {id: i for i, id in enumerate(rec_df[item_col].unique())}
        
        # Map IDs to integers
        rec_df['user_idx'] = rec_df[user_col].map(user_mapping)
        rec_df['item_idx'] = rec_df[item_col].map(item_mapping)
        
        # Create reverse mappings
        reverse_user_mapping = {v: k for k, v in user_mapping.items()}
        reverse_item_mapping = {v: k for k, v in item_mapping.items()}
        
        mappings = {
            'user_mapping': user_mapping,
            'item_mapping': item_mapping,
            'reverse_user_mapping': reverse_user_mapping,
            'reverse_item_mapping': reverse_item_mapping,
            'user_col': user_col,
            'item_col': item_col,
            'rating_col': rating_col
        }
        
        return rec_df, mappings
    else:
        st.error("Could not identify required columns for recommendation system. Please ensure your data contains user, item, and rating information.")
        return None, None


def prepare_for_segmentation(df):
    """
    Prepare data for user segmentation.
    
    Parameters:
    df (DataFrame): Processed user data
    
    Returns:
    DataFrame: Data ready for segmentation algorithms
    list: Feature columns used for segmentation
    """
    if df is None or df.empty:
        st.error("No data available for segmentation.")
        return None, None
    
    # Create a copy to avoid modifying the original
    seg_df = df.copy()
    
    # Try to identify user ID column
    user_col = None
    for col in seg_df.columns:
        if 'user' in col.lower() or col.lower() == 'uid' or col.lower() == 'id':
            user_col = col
            break
    
    # Identify potential feature columns for segmentation
    # Exclude ID columns, timestamp columns, and other non-feature columns
    exclude_patterns = ['id', 'name', 'email', 'phone', 'address', 'date', 'time', 'created', 'updated']
    
    feature_cols = []
    for col in seg_df.columns:
        # Skip columns that match exclude patterns
        if any(pattern in col.lower() for pattern in exclude_patterns):
            continue
        
        # Keep numeric columns and boolean columns
        if seg_df[col].dtype in ['int64', 'float64', 'bool']:
            feature_cols.append(col)
    
    # If we have feature columns, create the segmentation dataframe
    if feature_cols:
        # Handle any remaining missing values
        for col in feature_cols:
            if seg_df[col].isna().any():
                if seg_df[col].dtype in ['int64', 'float64']:
                    seg_df[col] = seg_df[col].fillna(seg_df[col].median())
                else:
                    seg_df[col] = seg_df[col].fillna(seg_df[col].mode()[0])
        
        # Create a dataset with only the feature columns
        features_df = seg_df[feature_cols].copy()
        
        # Store the user IDs if available
        if user_col:
            user_ids = seg_df[user_col].tolist()
            return features_df, feature_cols, user_ids
        
        return features_df, feature_cols
    else:
        st.error("Could not identify suitable feature columns for segmentation.")
        return None, None


def prepare_for_ab_testing(df):
    """
    Prepare data for A/B testing analysis.
    
    Parameters:
    df (DataFrame): Processed user data
    
    Returns:
    DataFrame: Data ready for A/B testing
    dict: Metadata about the experiment
    """
    if df is None or df.empty:
        st.error("No data available for A/B testing analysis.")
        return None, None
    
    # Create a copy to avoid modifying the original
    ab_df = df.copy()
    
    # Try to identify experiment, variant, and conversion columns
    experiment_col = None
    variant_col = None
    conversion_col = None
    
    # Look for experiment/variant columns
    for col in ab_df.columns:
        col_lower = col.lower()
        if 'experiment' in col_lower or 'test' in col_lower:
            experiment_col = col
        elif 'variant' in col_lower or 'group' in col_lower or 'version' in col_lower:
            variant_col = col
        elif any(term in col_lower for term in ['conversion', 'success', 'click', 'purchase']):
            if ab_df[col].dtype in ['int64', 'float64', 'bool']:
                conversion_col = col
    
    # If we couldn't find the expected columns, try to make educated guesses
    # Assume any binary column could be a variant or conversion
    binary_cols = []
    for col in ab_df.columns:
        if ab_df[col].nunique() == 2:
            binary_cols.append(col)
    
    if not variant_col and len(binary_cols) >= 1:
        variant_col = binary_cols[0]
    
    if not conversion_col and len(binary_cols) >= 2:
        conversion_col = binary_cols[1]
    
    # If we identified the necessary columns, prepare the data
    if variant_col and conversion_col:
        # Ensure conversion column is numeric
        if ab_df[conversion_col].dtype == 'bool':
            ab_df[conversion_col] = ab_df[conversion_col].astype(int)
        
        # Create metadata
        metadata = {
            'variant_col': variant_col,
            'conversion_col': conversion_col,
            'experiment_col': experiment_col,
            'variant_values': ab_df[variant_col].unique().tolist(),
            'baseline_variant': ab_df[variant_col].mode()[0],
            'metric_type': 'binary' if ab_df[conversion_col].nunique() <= 2 else 'continuous'
        }
        
        return ab_df, metadata
    else:
        st.error("Could not identify required columns for A/B testing analysis. Please ensure your data contains variant and conversion information.")
        return None, None
