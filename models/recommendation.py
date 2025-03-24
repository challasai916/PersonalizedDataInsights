import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import streamlit as st

class CollaborativeFilteringRecommender:
    """
    A simple collaborative filtering recommendation engine based on matrix factorization.
    """
    
    def __init__(self, n_factors=20, regularization=0.1, learning_rate=0.005, n_epochs=20):
        """
        Initialize the recommendation engine with hyperparameters.
        
        Parameters:
        n_factors (int): Number of latent factors
        regularization (float): Regularization term to prevent overfitting
        learning_rate (float): Learning rate for gradient descent
        n_epochs (int): Number of training epochs
        """
        self.n_factors = n_factors
        self.regularization = regularization
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        
        self.user_factors = None
        self.item_factors = None
        self.global_mean = None
        self.user_biases = None
        self.item_biases = None
        
        self.user_mapping = None
        self.item_mapping = None
        self.reverse_user_mapping = None
        self.reverse_item_mapping = None
        
        self.metrics = {}
    
    def fit(self, df, user_col='user_idx', item_col='item_idx', rating_col='rating'):
        """
        Train the recommendation model.
        
        Parameters:
        df (DataFrame): Training data with user, item, and rating columns
        user_col (str): User ID column name
        item_col (str): Item ID column name
        rating_col (str): Rating column name
        
        Returns:
        self: Trained model
        """
        # Store mappings
        self.user_mapping = {user: idx for idx, user in enumerate(df[user_col].unique())}
        self.item_mapping = {item: idx for idx, item in enumerate(df[item_col].unique())}
        self.reverse_user_mapping = {idx: user for user, idx in self.user_mapping.items()}
        self.reverse_item_mapping = {idx: item for item, idx in self.item_mapping.items()}
        
        # Get dimensions
        n_users = len(self.user_mapping)
        n_items = len(self.item_mapping)
        
        # Map user and item IDs to indices
        users = df[user_col].map(self.user_mapping).values
        items = df[item_col].map(self.item_mapping).values
        ratings = df[rating_col].values
        
        # Calculate global mean rating
        self.global_mean = ratings.mean()
        
        # Initialize model parameters
        # Initialize user and item factors with small random values
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        
        # Initialize biases
        self.user_biases = np.zeros(n_users)
        self.item_biases = np.zeros(n_items)
        
        # Perform stochastic gradient descent
        for epoch in range(self.n_epochs):
            # Shuffle the data
            indices = np.arange(len(ratings))
            np.random.shuffle(indices)
            
            # Training loop
            for idx in indices:
                u, i, r = users[idx], items[idx], ratings[idx]
                
                # Compute prediction
                pred = self.global_mean + self.user_biases[u] + self.item_biases[i] + \
                       np.dot(self.user_factors[u], self.item_factors[i])
                
                # Compute error
                err = r - pred
                
                # Update biases
                self.user_biases[u] += self.learning_rate * (err - self.regularization * self.user_biases[u])
                self.item_biases[i] += self.learning_rate * (err - self.regularization * self.item_biases[i])
                
                # Update factors
                u_factors_prev = self.user_factors[u].copy()
                self.user_factors[u] += self.learning_rate * (err * self.item_factors[i] - self.regularization * self.user_factors[u])
                self.item_factors[i] += self.learning_rate * (err * u_factors_prev - self.regularization * self.item_factors[i])
        
        # Evaluate on training data
        self._evaluate(users, items, ratings)
        
        return self
    
    def _evaluate(self, users, items, ratings):
        """
        Evaluate the model and calculate metrics
        """
        predictions = []
        for u, i, r in zip(users, items, ratings):
            pred = self.global_mean + self.user_biases[u] + self.item_biases[i] + \
                  np.dot(self.user_factors[u], self.item_factors[i])
            predictions.append(max(0.5, min(5, pred)))  # Clip predictions to valid range
        
        # Calculate metrics
        rmse = np.sqrt(np.mean((np.array(ratings) - np.array(predictions)) ** 2))
        mae = np.mean(np.abs(np.array(ratings) - np.array(predictions)))
        
        self.metrics = {
            'rmse': rmse,
            'mae': mae,
            'users_covered': len(np.unique(users)) / len(self.user_mapping),
            'items_covered': len(np.unique(items)) / len(self.item_mapping)
        }
    
    def predict(self, user_idx, item_idx):
        """
        Predict rating for a user-item pair.
        
        Parameters:
        user_idx (int): User index
        item_idx (int): Item index
        
        Returns:
        float: Predicted rating
        """
        if user_idx >= len(self.user_factors) or item_idx >= len(self.item_factors):
            return self.global_mean
        
        pred = self.global_mean + self.user_biases[user_idx] + self.item_biases[item_idx] + \
               np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        
        return max(0.5, min(5, pred))  # Clip to valid rating range
    
    def recommend_for_user(self, user_idx, n_recommendations=10, exclude_rated=True, df=None):
        """
        Generate recommendations for a user.
        
        Parameters:
        user_idx (int): User index
        n_recommendations (int): Number of recommendations to generate
        exclude_rated (bool): Whether to exclude already rated items
        df (DataFrame): Original data with user-item interactions
        
        Returns:
        DataFrame: Recommendations with item IDs and predicted ratings
        """
        if user_idx >= len(self.user_factors):
            return pd.DataFrame(columns=['item_idx', 'predicted_rating'])
        
        # Get all items
        all_items = np.arange(len(self.item_factors))
        
        # Exclude items that the user has already rated
        if exclude_rated and df is not None:
            # Map the user_idx to original user ID
            original_user_id = self.reverse_user_mapping[user_idx]
            
            # Get all items rated by this user
            user_col = [col for col in df.columns if col.endswith('_idx') and col != 'item_idx'][0]
            rated_items = df[df[user_col] == original_user_id]['item_idx'].unique()
            
            # Map back to model indices
            rated_indices = [self.item_mapping.get(item, -1) for item in rated_items]
            rated_indices = [idx for idx in rated_indices if idx != -1]
            
            # Create a mask for unrated items
            mask = np.ones(len(all_items), dtype=bool)
            mask[rated_indices] = False
            
            # Filter to only unrated items
            all_items = all_items[mask]
        
        # Generate predictions for all (unrated) items
        predictions = []
        for item_idx in all_items:
            pred = self.predict(user_idx, item_idx)
            predictions.append((item_idx, pred))
        
        # Sort by predicted rating (descending)
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Take top N
        top_n = predictions[:n_recommendations]
        
        # Convert to DataFrame
        recommendations = pd.DataFrame(top_n, columns=['item_idx', 'predicted_rating'])
        
        # Map item indices back to original item IDs
        recommendations['item_id'] = recommendations['item_idx'].map(self.reverse_item_mapping)
        
        return recommendations


def evaluate_recommendations(model, test_df, user_col='user_idx', item_col='item_idx', rating_col='rating', k=10):
    """
    Evaluate recommendations on a test set.
    
    Parameters:
    model (CollaborativeFilteringRecommender): Trained model
    test_df (DataFrame): Test data with user, item, and rating columns
    user_col (str): User ID column name
    item_col (str): Item ID column name
    rating_col (str): Rating column name
    k (int): Number of recommendations to consider
    
    Returns:
    dict: Evaluation metrics
    """
    # Filter to only include users and items in the model
    model_users = set(model.user_mapping.keys())
    model_items = set(model.item_mapping.keys())
    
    filtered_df = test_df[
        test_df[user_col].isin(model_users) &
        test_df[item_col].isin(model_items)
    ]
    
    if len(filtered_df) == 0:
        return {
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'users_covered': 0
        }
    
    # Define a threshold for considering an item "relevant"
    # For explicit feedback, we might consider ratings above average as relevant
    # For implicit feedback, all interactions might be considered relevant
    if filtered_df[rating_col].nunique() > 2:
        # Explicit feedback case
        threshold = filtered_df[rating_col].mean()
        filtered_df['relevant'] = filtered_df[rating_col] >= threshold
    else:
        # Implicit feedback case (binary interactions)
        filtered_df['relevant'] = True
    
    # Calculate metrics for each user
    precision_sum = 0
    recall_sum = 0
    f1_sum = 0
    user_count = 0
    
    for user in filtered_df[user_col].unique():
        # Get actual relevant items for this user
        actual_relevant = set(filtered_df[
            (filtered_df[user_col] == user) & 
            (filtered_df['relevant'])
        ][item_col].values)
        
        if len(actual_relevant) == 0:
            continue
        
        # Get model recommendations for this user
        user_idx = model.user_mapping.get(user)
        if user_idx is None:
            continue
        
        recommendations = model.recommend_for_user(user_idx, n_recommendations=k, exclude_rated=False)
        recommended_items = set(recommendations['item_id'].values)
        
        # Calculate metrics
        n_relevant_and_recommended = len(actual_relevant.intersection(recommended_items))
        
        precision = n_relevant_and_recommended / len(recommended_items) if recommended_items else 0
        recall = n_relevant_and_recommended / len(actual_relevant) if actual_relevant else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        precision_sum += precision
        recall_sum += recall
        f1_sum += f1
        user_count += 1
    
    if user_count == 0:
        return {
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'users_covered': 0
        }
    
    # Average metrics across users
    return {
        'precision': precision_sum / user_count,
        'recall': recall_sum / user_count,
        'f1': f1_sum / user_count,
        'users_covered': user_count / len(model_users)
    }
