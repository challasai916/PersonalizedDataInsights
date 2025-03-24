import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from datetime import datetime

def handle_missing_values(data):
    """
    Handle missing values in the dataset.
    
    Parameters:
        data (pd.DataFrame): Input data
        
    Returns:
        pd.DataFrame: Data with handled missing values
    """
    # Make a copy of the data
    df = data.copy()
    
    # For numeric columns, impute with median
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if not numeric_cols.empty:
        imputer = SimpleImputer(strategy='median')
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    # For categorical columns, impute with mode
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
    
    return df

def normalize_numeric_features(data):
    """
    Normalize numeric features in the dataset.
    
    Parameters:
        data (pd.DataFrame): Input data
        
    Returns:
        pd.DataFrame: Data with normalized numeric features
    """
    # Make a copy of the data
    df = data.copy()
    
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    # Skip columns that appear to be IDs
    numeric_cols = [col for col in numeric_cols if not (col.lower().endswith('id') or col.lower() == 'id')]
    
    if not numeric_cols.empty:
        # Use MinMaxScaler to normalize features to [0, 1] range
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df

def encode_categorical_features(data):
    """
    Encode categorical features in the dataset.
    
    Parameters:
        data (pd.DataFrame): Input data
        
    Returns:
        pd.DataFrame: Data with encoded categorical features
    """
    # Make a copy of the data
    df = data.copy()
    
    # Get categorical columns, excluding potential ID columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    categorical_cols = [col for col in categorical_cols if not (col.lower().endswith('id') or col.lower() == 'id')]
    
    # Encode each categorical column
    for col in categorical_cols:
        # If the column has many unique values, use pd.get_dummies (one-hot encoding)
        if df[col].nunique() < 10:  # threshold for one-hot encoding
            # Create dummy variables
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
            # Add dummy variables to the dataframe
            df = pd.concat([df, dummies], axis=1)
            # Drop the original column
            df = df.drop(col, axis=1)
        else:
            # For high cardinality features, use label encoding
            df[col] = df[col].astype('category').cat.codes
    
    return df

def select_important_features(data, k=10):
    """
    Select the k most important features from the dataset.
    
    Parameters:
        data (pd.DataFrame): Input data
        k (int): Number of features to select
        
    Returns:
        pd.DataFrame: Data with only the important features
    """
    # Make a copy of the data
    df = data.copy()
    
    # Try to identify a potential target variable
    # This is just a heuristic approach
    target_candidates = [col for col in df.columns if col.lower() in 
                         ['target', 'label', 'class', 'response', 'outcome', 
                          'revenue', 'purchase', 'conversion', 'click']]
    
    if target_candidates:
        target_col = target_candidates[0]
        features = [col for col in df.columns if col != target_col]
        
        # Select only numeric features for feature selection
        numeric_features = df[features].select_dtypes(include=['int64', 'float64']).columns
        
        if len(numeric_features) > k and not df[target_col].isna().any():
            # Use SelectKBest to get k best features
            selector = SelectKBest(f_classif, k=min(k, len(numeric_features)))
            X_new = selector.fit_transform(df[numeric_features], df[target_col])
            
            # Get the selected feature names
            selected_features = numeric_features[selector.get_support()]
            
            # Keep only selected features and the target
            selected_cols = list(selected_features) + [target_col]
            df = df[selected_cols]
            
            # Print the selected features and their scores
            feature_scores = pd.DataFrame({
                'Feature': numeric_features,
                'Score': selector.scores_
            })
            print("Selected features and their scores:")
            print(feature_scores.sort_values(by='Score', ascending=False).head(k))
    
    return df

def apply_rule_based_segmentation(data, rules):
    """
    Apply rule-based segmentation to the dataset.
    
    Parameters:
        data (pd.DataFrame): Input data
        rules (list): List of rule dictionaries with field, operator, and value
        
    Returns:
        pd.DataFrame: Data with segment column added
    """
    # Make a copy of the data
    df = data.copy()
    
    # Default segment
    df['segment'] = 'Other'
    
    # Apply each rule in sequence
    for i, rule in enumerate(rules):
        field = rule['field']
        operator = rule['operator']
        value = rule['value']
        
        # Create a segment name based on the rule
        segment_name = f"Segment_{i+1}"
        
        # Apply the rule based on the operator
        if operator == "==":
            df.loc[df[field] == value, 'segment'] = segment_name
        elif operator == ">":
            df.loc[df[field] > value, 'segment'] = segment_name
        elif operator == "<":
            df.loc[df[field] < value, 'segment'] = segment_name
        elif operator == ">=":
            df.loc[df[field] >= value, 'segment'] = segment_name
        elif operator == "<=":
            df.loc[df[field] <= value, 'segment'] = segment_name
        elif operator == "!=":
            df.loc[df[field] != value, 'segment'] = segment_name
        elif operator == "contains":
            if df[field].dtype == 'object':  # Only apply contains to string columns
                df.loc[df[field].str.contains(str(value), na=False), 'segment'] = segment_name
    
    return df

def perform_rfm_analysis(data, customer_id_col, recency_col, frequency_col, monetary_col):
    """
    Perform RFM (Recency, Frequency, Monetary) analysis.
    
    Parameters:
        data (pd.DataFrame): Input data
        customer_id_col (str): Column name for customer ID
        recency_col (str): Column name for recency (date/time)
        frequency_col (str): Column name for frequency
        monetary_col (str): Column name for monetary value
        
    Returns:
        pd.DataFrame: RFM analysis results with segments
    """
    # Make a copy of the data
    df = data.copy()
    
    # Ensure frequency and monetary columns are numeric
    df[frequency_col] = pd.to_numeric(df[frequency_col], errors='coerce')
    df[monetary_col] = pd.to_numeric(df[monetary_col], errors='coerce')
    
    # Calculate recency in days
    if df[recency_col].dtype != 'datetime64[ns]':
        df[recency_col] = pd.to_datetime(df[recency_col], errors='coerce')
        
    # Get the maximum date in the data to calculate recency
    max_date = df[recency_col].max()
    
    # Group by customer and calculate RFM metrics
    rfm = df.groupby(customer_id_col).agg({
        recency_col: lambda x: (max_date - x.max()).days,  # Recency in days
        frequency_col: 'sum',  # Frequency
        monetary_col: 'sum'  # Monetary
    }).reset_index()
    
    # Rename columns
    rfm.columns = [customer_id_col, 'recency', 'frequency', 'monetary']
    
    # Create RFM scores (1-5) - higher is better for F and M, lower is better for R
    rfm['recency_score'] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm['frequency_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
    rfm['monetary_score'] = pd.qcut(rfm['monetary'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
    
    # Convert scores to numeric
    rfm['recency_score'] = rfm['recency_score'].astype(int)
    rfm['frequency_score'] = rfm['frequency_score'].astype(int)
    rfm['monetary_score'] = rfm['monetary_score'].astype(int)
    
    # Calculate RFM combined score
    rfm['rfm_score'] = rfm['recency_score'] * 100 + rfm['frequency_score'] * 10 + rfm['monetary_score']
    
    # Define RFM segments
    def rfm_segment(row):
        if row['rfm_score'] >= 444:
            return 'Champions'
        elif row['recency_score'] >= 5 and (row['frequency_score'] + row['monetary_score'] >= 8):
            return 'Loyal Customers'
        elif row['recency_score'] >= 4 and (row['frequency_score'] + row['monetary_score'] >= 6):
            return 'Potential Loyalists'
        elif row['recency_score'] >= 5 and (row['frequency_score'] + row['monetary_score'] < 6):
            return 'Promising'
        elif row['recency_score'] >= 3 and (row['frequency_score'] + row['monetary_score'] >= 5):
            return 'Need Attention'
        elif row['recency_score'] <= 2 and (row['frequency_score'] + row['monetary_score'] >= 8):
            return 'At Risk'
        elif row['recency_score'] <= 2 and (row['frequency_score'] + row['monetary_score'] >= 5):
            return 'Can\'t Lose Them'
        elif row['recency_score'] <= 3 and (row['frequency_score'] + row['monetary_score'] < 5):
            return 'About To Sleep'
        elif row['recency_score'] <= 2 and (row['frequency_score'] + row['monetary_score'] < 5):
            return 'Hibernating'
        else:
            return 'Lost'
    
    rfm['rfm_segment'] = rfm.apply(rfm_segment, axis=1)
    
    return rfm

def create_threshold_segments(data, feature, thresholds, segment_names):
    """
    Create segments based on thresholds for a specific feature.
    
    Parameters:
        data (pd.DataFrame): Input data
        feature (str): Feature to use for segmentation
        thresholds (list): List of threshold values
        segment_names (list): List of segment names (should be len(thresholds) + 1)
        
    Returns:
        pd.DataFrame: Data with segment column added
    """
    # Make a copy of the data
    df = data.copy()
    
    # Ensure feature is numeric
    df[feature] = pd.to_numeric(df[feature], errors='coerce')
    
    # Create segment column
    df['segment'] = pd.cut(
        df[feature],
        bins=[-float('inf')] + thresholds + [float('inf')],
        labels=segment_names,
        include_lowest=True
    )
    
    return df
