import pandas as pd
import numpy as np
import io
from datetime import datetime, timedelta
import random

def get_data_info(data):
    """
    Get information about a DataFrame.
    
    Parameters:
        data (pd.DataFrame): Input data
        
    Returns:
        str: String containing data information
    """
    # Create a buffer to capture the output
    buffer = io.StringIO()
    
    # Write general information
    buffer.write(f"DataFrame Shape: {data.shape[0]} rows, {data.shape[1]} columns\n\n")
    
    # Write column information
    buffer.write("Column Information:\n")
    buffer.write("-" * 80 + "\n")
    buffer.write(f"{'Column Name':<30} {'Data Type':<15} {'Non-Null':<10} {'Unique Values':<15} {'Range/Examples':<20}\n")
    buffer.write("-" * 80 + "\n")
    
    for col in data.columns:
        # Get data type and non-null count
        dtype = str(data[col].dtype)
        non_null = data[col].count()
        unique_count = data[col].nunique()
        
        # Get range for numeric columns or examples for categorical
        if pd.api.types.is_numeric_dtype(data[col]):
            range_info = f"{data[col].min()} to {data[col].max()}"
        else:
            # Get a few examples for categorical columns
            examples = ', '.join(data[col].dropna().sample(min(3, unique_count)).astype(str).values)
            if len(examples) > 20:
                examples = examples[:17] + "..."
            range_info = examples
        
        # Write column info
        buffer.write(f"{col:<30} {dtype:<15} {non_null:<10} {unique_count:<15} {range_info:<20}\n")
    
    # Missing values
    buffer.write("\nMissing Values:\n")
    buffer.write("-" * 40 + "\n")
    missing = data.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        for col, count in missing.items():
            buffer.write(f"{col}: {count} ({count/len(data):.2%})\n")
    else:
        buffer.write("No missing values found.\n")
    
    # Memory usage
    buffer.write("\nMemory Usage:\n")
    buffer.write("-" * 40 + "\n")
    memory_usage = data.memory_usage(deep=True)
    total_memory = memory_usage.sum()
    buffer.write(f"Total memory usage: {total_memory / 1024**2:.2f} MB\n")
    
    return buffer.getvalue()

def load_ecommerce_sample_data(n_users=1000, n_products=100, n_orders=5000):
    """
    Load a sample e-commerce dataset.
    
    Parameters:
        n_users (int): Number of users to generate
        n_products (int): Number of products to generate
        n_orders (int): Number of orders to generate
        
    Returns:
        pd.DataFrame: Sample e-commerce data
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Generate user data
    user_ids = [f"U{i:05d}" for i in range(1, n_users + 1)]
    
    # Generate product data
    product_ids = [f"P{i:05d}" for i in range(1, n_products + 1)]
    product_categories = ["Electronics", "Clothing", "Home & Kitchen", "Books", "Beauty"]
    product_prices = np.random.uniform(10, 200, n_products)
    
    products = pd.DataFrame({
        'product_id': product_ids,
        'product_category': [random.choice(product_categories) for _ in range(n_products)],
        'product_price': product_prices
    })
    
    # Generate orders
    order_data = []
    
    # Current date
    current_date = datetime.now()
    
    for _ in range(n_orders):
        # Random user
        user_id = random.choice(user_ids)
        
        # Random date (within last 1 year)
        days_ago = random.randint(1, 365)
        order_date = current_date - timedelta(days=days_ago)
        
        # Number of products in this order
        n_items = random.randint(1, 5)
        
        # Select products for this order
        order_products = random.sample(product_ids, n_items)
        
        for product_id in order_products:
            # Get product price
            product_price = products[products['product_id'] == product_id]['product_price'].values[0]
            
            # Quantity
            quantity = random.randint(1, 3)
            
            # Order value
            order_value = product_price * quantity
            
            # Rating (only some orders have ratings)
            has_rating = random.random() < 0.7  # 70% of orders have ratings
            rating = random.randint(1, 5) if has_rating else np.nan
            
            # Add to order data
            order_data.append({
                'user_id': user_id,
                'product_id': product_id,
                'order_date': order_date,
                'quantity': quantity,
                'order_value': order_value,
                'rating': rating
            })
    
    # Create DataFrame
    orders_df = pd.DataFrame(order_data)
    
    # Add some user demographic data
    age_groups = ["18-24", "25-34", "35-44", "45-54", "55+"]
    genders = ["Male", "Female", "Other"]
    
    user_data = []
    
    for user_id in user_ids:
        user_data.append({
            'user_id': user_id,
            'age_group': random.choice(age_groups),
            'gender': random.choice(genders),
            'signup_date': current_date - timedelta(days=random.randint(30, 500))
        })
    
    users_df = pd.DataFrame(user_data)
    
    # Merge orders with user data
    ecommerce_data = pd.merge(orders_df, users_df, on='user_id')
    
    # Merge with product data
    ecommerce_data = pd.merge(ecommerce_data, products, on='product_id')
    
    # Add additional user behavior metrics
    # Frequency: number of orders per user
    user_frequency = ecommerce_data.groupby('user_id').size().reset_index(name='order_frequency')
    
    # Recency: days since last order
    user_recency = ecommerce_data.groupby('user_id')['order_date'].max().reset_index()
    user_recency['days_since_last_order'] = (current_date - user_recency['order_date']).dt.days
    
    # Monetary: total spend per user
    user_monetary = ecommerce_data.groupby('user_id')['order_value'].sum().reset_index(name='total_spend')
    
    # Merge user metrics back
    ecommerce_data = pd.merge(ecommerce_data, user_frequency, on='user_id')
    ecommerce_data = pd.merge(ecommerce_data, user_recency[['user_id', 'days_since_last_order']], on='user_id')
    ecommerce_data = pd.merge(ecommerce_data, user_monetary, on='user_id')
    
    return ecommerce_data

def load_content_platform_sample_data(n_users=1000, n_content=200, n_interactions=8000):
    """
    Load a sample content platform dataset.
    
    Parameters:
        n_users (int): Number of users to generate
        n_content (int): Number of content items to generate
        n_interactions (int): Number of interactions to generate
        
    Returns:
        pd.DataFrame: Sample content platform data
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Generate user data
    user_ids = [f"U{i:05d}" for i in range(1, n_users + 1)]
    
    # Generate content data
    content_ids = [f"C{i:05d}" for i in range(1, n_content + 1)]
    content_types = ["Article", "Video", "Podcast", "Image"]
    content_categories = ["Technology", "Business", "Entertainment", "Sports", "Science", "Health"]
    
    content_data = []
    
    for content_id in content_ids:
        content_type = random.choice(content_types)
        
        # Content duration or length varies by type
        if content_type == "Article":
            length = random.randint(500, 5000)  # word count
            duration = None
        elif content_type == "Video":
            length = None
            duration = random.randint(30, 3600)  # seconds
        elif content_type == "Podcast":
            length = None
            duration = random.randint(600, 3600)  # seconds
        else:  # Image
            length = None
            duration = None
        
        content_data.append({
            'content_id': content_id,
            'content_type': content_type,
            'content_category': random.choice(content_categories),
            'publication_date': datetime.now() - timedelta(days=random.randint(1, 365)),
            'length': length,
            'duration': duration
        })
    
    content_df = pd.DataFrame(content_data)
    
    # Generate user interaction data
    interaction_data = []
    
    # Current time
    current_time = datetime.now()
    
    for _ in range(n_interactions):
        # Random user and content
        user_id = random.choice(user_ids)
        content_id = random.choice(content_ids)
        
        # Get content details
        content_row = content_df[content_df['content_id'] == content_id].iloc[0]
        
        # Interaction time
        interaction_time = current_time - timedelta(days=random.randint(0, 90), 
                                                    hours=random.randint(0, 23),
                                                    minutes=random.randint(0, 59))
        
        # Interaction type
        interaction_type = random.choice(["View", "Like", "Share", "Comment"])
        
        # Interaction duration (for View type)
        if interaction_type == "View":
            if content_row['content_type'] == "Article":
                # Time to read article (based on length)
                max_duration = content_row['length'] / 200  # ~200 words per minute
                view_duration = random.uniform(10, max_duration)
            elif content_row['content_type'] in ["Video", "Podcast"]:
                # Time watching/listening (based on content duration)
                if content_row['duration']:
                    max_duration = content_row['duration']
                    view_duration = random.uniform(10, max_duration)
                else:
                    view_duration = None
            else:
                # Time viewing image
                view_duration = random.uniform(5, 60)
        else:
            view_duration = None
        
        # Interaction score (e.g., rating)
        if random.random() < 0.8:  # 80% of interactions have a score
            if interaction_type == "Like":
                score = 1
            elif interaction_type == "View":
                score = random.randint(1, 5)
            else:
                score = None
        else:
            score = None
        
        # Add to interaction data
        interaction_data.append({
            'user_id': user_id,
            'content_id': content_id,
            'interaction_type': interaction_type,
            'interaction_time': interaction_time,
            'view_duration': view_duration,
            'score': score
        })
    
    # Create DataFrame
    interactions_df = pd.DataFrame(interaction_data)
    
    # Add user demographic data
    user_data = []
    
    age_groups = ["18-24", "25-34", "35-44", "45-54", "55+"]
    interests = ["Technology", "Business", "Entertainment", "Sports", "Science", "Health", 
                 "Politics", "Travel", "Food", "Fashion"]
    devices = ["Mobile", "Desktop", "Tablet", "Smart TV"]
    
    for user_id in user_ids:
        # Random number of interests (1-4)
        n_interests = random.randint(1, 4)
        user_interests = random.sample(interests, n_interests)
        
        user_data.append({
            'user_id': user_id,
            'age_group': random.choice(age_groups),
            'primary_device': random.choice(devices),
            'interests': ','.join(user_interests),
            'signup_date': current_time - timedelta(days=random.randint(1, 500))
        })
    
    users_df = pd.DataFrame(user_data)
    
    # Calculate engagement metrics
    
    # Total interactions per user
    user_interaction_counts = interactions_df.groupby('user_id').size().reset_index(name='total_interactions')
    
    # Average view duration per user (for view interactions)
    view_durations = interactions_df[interactions_df['interaction_type'] == 'View']
    user_avg_view_duration = view_durations.groupby('user_id')['view_duration'].mean().reset_index(name='avg_view_duration')
    
    # Interaction frequency (interactions per day since signup)
    user_tenure = users_df.copy()
    user_tenure['days_since_signup'] = (current_time - user_tenure['signup_date']).dt.days
    user_tenure = user_tenure[['user_id', 'days_since_signup']]
    
    # Merge with interaction counts
    user_frequency = pd.merge(user_interaction_counts, user_tenure, on='user_id')
    user_frequency['interaction_frequency'] = user_frequency['total_interactions'] / user_frequency['days_since_signup']
    
    # Merge all data
    # First, merge interactions with content data
    platform_data = pd.merge(interactions_df, content_df, on='content_id')
    
    # Merge with user data
    platform_data = pd.merge(platform_data, users_df, on='user_id')
    
    # Merge with engagement metrics
    platform_data = pd.merge(platform_data, user_interaction_counts, on='user_id')
    platform_data = pd.merge(platform_data, user_avg_view_duration, on='user_id', how='left')
    platform_data = pd.merge(platform_data, user_frequency[['user_id', 'interaction_frequency']], on='user_id')
    
    return platform_data
