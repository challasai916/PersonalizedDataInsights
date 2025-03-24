import streamlit as st

def show():
    """Display the home page."""
    st.title("Data Science Platform for Personalization")
    
    st.markdown("""
    ## Welcome to the Personalization Data Science Platform
    
    This platform helps you analyze user behavior and implement personalization features for online platforms.
    
    ### Key Features:
    
    1. **Data Upload and Processing**
       - Support for CSV and JSON formats
       - Basic data preprocessing capabilities
    
    2. **Exploratory Data Analysis**
       - Data overview and summary statistics
       - Correlation analysis
       - Feature distributions
    
    3. **User Behavior Analysis**
       - Activity patterns visualization
       - Behavioral metrics analysis
       - Engagement insights
    
    4. **Recommendation Engine**
       - Collaborative filtering algorithm
       - User-based recommendations
       - Performance evaluation
    
    5. **User Segmentation**
       - Automated clustering
       - Segment profiling
       - Visual segment comparison
    
    6. **A/B Testing Analysis**
       - Conversion rate comparison
       - Statistical significance testing
       - Sample size calculator
    
    7. **Performance Metrics**
       - Model evaluation metrics
       - Personalization impact assessment
       - Visualization of key indicators
    
    ### Getting Started:
    
    1. Upload your user behavior data in CSV or JSON format
    2. Explore and analyze patterns in your data
    3. Apply machine learning models for personalization
    4. Measure and optimize performance
    
    Navigate through the different sections using the sidebar.
    """)
    
    # Add an example data description
    with st.expander("Example Data Structure"):
        st.markdown("""
        ### Example User Behavior Data Format
        
        This platform works best with data that includes:
        
        - **User identifiers**: user_id, customer_id, etc.
        - **Item identifiers**: product_id, article_id, content_id, etc.
        - **Interaction data**: views, clicks, purchases, ratings, etc.
        - **Timestamps**: when interactions occurred
        - **User attributes**: demographics, preferences, etc.
        - **Item attributes**: categories, tags, features, etc.
        
        #### Example CSV structure:
        
        