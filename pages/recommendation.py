import streamlit as st
import pandas as pd
import numpy as np
from models.recommendation import CollaborativeFilteringRecommender, evaluate_recommendations
from utils.data_processor import prepare_for_recommendation
from utils.visualization import plot_recommendation_metrics

def show():
    """Display the recommendation engine page."""
    st.title("Recommendation Engine")
    
    # Check if data exists in session state
    if st.session_state.data is None:
        st.warning("Please upload data first on the 'Data Upload' page.")
        return
    
    df = st.session_state.data
    
    st.markdown("""
    ## Build a Recommendation System
    
    This page helps you create a collaborative filtering recommendation system based on your user data.
    The system analyzes user-item interactions to provide personalized recommendations.
    """)
    
    # Check if we already have a trained model
    has_model = st.session_state.recommendation_model is not None
    
    # Tabs for train/test and recommendations
    tabs = st.tabs(["Data Preparation", "Model Training", "Generate Recommendations", "Model Performance"])
    
    # Data Preparation tab
    with tabs[0]:
        st.markdown("### Prepare Data for Recommendations")
        
        st.write("""
        To build a recommendation system, we need data that shows how users interact with items.
        This typically includes:
        - User identifiers
        - Item identifiers
        - Ratings or interaction data
        """)
        
        # Prepare the data for the recommendation system
        with st.spinner("Preparing data for recommendation system..."):
            rec_df, mappings = prepare_for_recommendation(df)
        
        if rec_df is not None and mappings is not None:
            st.success("Data successfully prepared for recommendation system.")
            
            # Display the identified columns
            st.markdown("#### Identified Columns")
            st.write(f"User column: {mappings['user_col']}")
            st.write(f"Item column: {mappings['item_col']}")
            st.write(f"Rating/interaction column: {mappings['rating_col']}")
            
            # Display data preview
            st.markdown("#### Data Preview")
            st.dataframe(rec_df.head())
            
            # Data statistics
            st.markdown("#### Data Statistics")
            st.write(f"Number of users: {len(mappings['user_mapping'])}")
            st.write(f"Number of items: {len(mappings['item_mapping'])}")
            st.write(f"Number of interactions: {len(rec_df)}")
            
            # Density calculation
            density = len(rec_df) / (len(mappings['user_mapping']) * len(mappings['item_mapping'])) * 100
            st.write(f"Data density: {density:.4f}% (higher density means more data for better recommendations)")
            
            # Store the prepared data in session state
            st.session_state.rec_df = rec_df
            st.session_state.rec_mappings = mappings
        else:
            st.error("Could not prepare data for recommendations. Please ensure your data includes user-item interactions.")
    
    # Model Training tab
    with tabs[1]:
        st.markdown("### Train Recommendation Model")
        
        if not hasattr(st.session_state, 'rec_df') or st.session_state.rec_df is None:
            st.warning("Please prepare your data in the 'Data Preparation' tab first.")
        else:
            # Model parameters
            st.markdown("#### Model Parameters")
            
            col1, col2 = st.columns(2)
            with col1:
                n_factors = st.slider("Number of factors", min_value=5, max_value=100, value=20, step=5)
                learning_rate = st.slider("Learning rate", min_value=0.001, max_value=0.1, value=0.005, step=0.001, format="%.3f")
            
            with col2:
                regularization = st.slider("Regularization", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
                n_epochs = st.slider("Training epochs", min_value=5, max_value=50, value=20, step=5)
            
            # Train model button
            if st.button("Train Recommendation Model"):
                with st.spinner("Training recommendation model... This may take a moment."):
                    # Train model
                    model = CollaborativeFilteringRecommender(
                        n_factors=n_factors,
                        regularization=regularization,
                        learning_rate=learning_rate,
                        n_epochs=n_epochs
                    )
                    
                    # Split data for evaluation
                    rec_df = st.session_state.rec_df
                    train_ratio = 0.8
                    train_idx = np.random.choice(len(rec_df), size=int(len(rec_df) * train_ratio), replace=False)
                    train_mask = np.zeros(len(rec_df), dtype=bool)
                    train_mask[train_idx] = True
                    
                    train_df = rec_df[train_mask]
                    test_df = rec_df[~train_mask]
                    
                    # Fit the model
                    model.fit(train_df)
                    
                    # Evaluate the model
                    eval_metrics = evaluate_recommendations(model, test_df)
                    
                    # Store model and metrics in session state
                    st.session_state.recommendation_model = model
                    st.session_state.recommendation_metrics = eval_metrics
                    
                    # Display training results
                    st.success("Model trained successfully!")
                    st.markdown("#### Training Results")
                    st.write(f"RMSE: {model.metrics['rmse']:.4f}")
                    st.write(f"MAE: {model.metrics['mae']:.4f}")
                    
                    # Display evaluation metrics
                    st.markdown("#### Evaluation Metrics")
                    st.write(f"Precision@10: {eval_metrics['precision']:.4f}")
                    st.write(f"Recall@10: {eval_metrics['recall']:.4f}")
                    st.write(f"F1 Score@10: {eval_metrics['f1']:.4f}")
                    st.write(f"User Coverage: {eval_metrics['users_covered']*100:.2f}%")
            
            # Display existing model info if available
            elif has_model:
                st.success("A recommendation model is already trained.")
                if hasattr(st.session_state, 'recommendation_metrics'):
                    metrics = st.session_state.recommendation_metrics
                    
                    st.markdown("#### Model Performance")
                    st.write(f"Precision@10: {metrics['precision']:.4f}")
                    st.write(f"Recall@10: {metrics['recall']:.4f}")
                    st.write(f"F1 Score@10: {metrics['f1']:.4f}")
                    st.write(f"User Coverage: {metrics['users_covered']*100:.2f}%")
    
    # Generate Recommendations tab
    with tabs[2]:
        st.markdown("### Generate Recommendations")
        
        if not has_model:
            st.warning("Please train a recommendation model in the 'Model Training' tab first.")
        else:
            # Get the model and mappings
            model = st.session_state.recommendation_model
            mappings = st.session_state.rec_mappings
            rec_df = st.session_state.rec_df
            
            # Reverse mappings for display
            reverse_user_mapping = model.reverse_user_mapping
            reverse_item_mapping = model.reverse_item_mapping
            
            # Select a user for recommendations
            st.markdown("#### Generate Recommendations for a User")
            
            # Get list of users
            user_ids = list(mappings['user_mapping'].keys())
            selected_user = st.selectbox("Select a user:", user_ids)
            
            if selected_user:
                # Convert to model index
                user_idx = mappings['user_mapping'].get(selected_user)
                
                # Generate recommendations
                n_recommendations = st.slider("Number of recommendations", min_value=5, max_value=20, value=10)
                
                exclude_rated = st.checkbox("Exclude already rated items", value=True)
                
                if st.button("Generate Recommendations"):
                    with st.spinner("Generating recommendations..."):
                        # Get recommendations
                        recommendations = model.recommend_for_user(
                            user_idx, 
                            n_recommendations=n_recommendations,
                            exclude_rated=exclude_rated,
                            df=rec_df
                        )
                        
                        if recommendations is not None and not recommendations.empty:
                            # Display recommendations
                            st.markdown("#### Top Recommendations")
                            
                            # Format the recommendations for display
                            display_recs = recommendations.copy()
                            display_recs.rename(columns={
                                'item_id': mappings['item_col'],
                                'predicted_rating': 'Predicted Rating'
                            }, inplace=True)
                            
                            # Drop the item_idx column
                            display_recs = display_recs.drop(columns=['item_idx'])
                            
                            st.dataframe(display_recs)
                        else:
                            st.warning("Could not generate recommendations for this user.")
    
    # Model Performance tab
    with tabs[3]:
        st.markdown("### Model Performance")
        
        if not has_model:
            st.warning("Please train a recommendation model in the 'Model Training' tab first.")
        elif not hasattr(st.session_state, 'recommendation_metrics'):
            st.warning("No performance metrics available. Please retrain the model.")
        else:
            # Get metrics
            metrics = st.session_state.recommendation_metrics
            
            # Display metrics
            st.markdown("#### Performance Metrics")
            
            # Plot metrics
            plot_recommendation_metrics(
                metrics['precision'],
                metrics['recall'],
                metrics['f1'],
                metrics['users_covered']
            )
            
            # Advanced metrics explanation
            with st.expander("Understanding Recommendation Metrics"):
                st.markdown("""
                ### Recommendation Metrics Explained
                
                - **Precision@10**: The proportion of recommended items that are relevant to the user. Higher is better.
                
                - **Recall@10**: The proportion of relevant items that are recommended to the user. Higher is better.
                
                - **F1 Score**: The harmonic mean of precision and recall, providing a balance between the two metrics. Higher is better.
                
                - **User Coverage**: The proportion of users for whom the system can generate recommendations. Higher coverage means the system can serve more users.
                
                These metrics help evaluate how well the recommendation system performs in terms of relevance and coverage.
                """)
