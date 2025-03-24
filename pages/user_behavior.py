import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.visualization import plot_user_behavior

def show():
    """Display the user behavior analysis page."""
    st.title("User Behavior Analysis")
    
    # Check if data exists in session state
    if st.session_state.data is None:
        st.warning("Please upload data first on the 'Data Upload' page.")
        return
    
    df = st.session_state.data
    
    st.markdown("""
    ## Analyze User Behavior
    
    Understanding how users interact with your platform is crucial for personalization.
    This page helps you analyze user behaviors, identify patterns, and gain insights.
    """)
    
    # Try to identify behavioral columns
    time_cols = [col for col in df.columns if 'time' in col.lower() 
                 or 'date' in col.lower() or col.lower().endswith('at')]
    
    behavior_cols = [col for col in df.columns if any(term in col.lower() 
                     for term in ['click', 'view', 'purchase', 'rating', 'like', 'share', 
                                  'comment', 'count', 'visit', 'score', 'duration', 'frequency'])]
    
    user_id_cols = [col for col in df.columns if 'user' in col.lower() 
                    or 'id' in col.lower() or 'customer' in col.lower()]
    
    # User behavior visualizations
    if behavior_cols:
        st.markdown("### User Behavior Visualizations")
        plot_user_behavior(df, behavior_cols)
    else:
        st.warning("No clear behavioral columns detected. Please select columns for analysis.")
        behavior_cols = st.multiselect("Select behavioral metrics columns:", df.columns)
        if behavior_cols:
            plot_user_behavior(df, behavior_cols)
    
    # User activity patterns
    st.markdown("### User Activity Patterns")
    
    # Check if we have time-related columns
    if time_cols:
        time_col = st.selectbox("Select timestamp column:", time_cols)
        
        # Check if the column is already a datetime
        if not pd.api.types.is_datetime64_dtype(df[time_col]):
            # Try to convert to datetime
            try:
                df[time_col] = pd.to_datetime(df[time_col])
                st.success(f"Converted {time_col} to datetime format.")
            except:
                st.error(f"Could not convert {time_col} to datetime. Please choose another column.")
                return
        
        # Aggregate by time periods
        st.markdown("#### Activity Over Time")
        
        # Set the time column as index
        time_df = df.set_index(time_col)
        
        # Time period selector
        time_period = st.selectbox("Select time period:", ["Hour", "Day", "Week", "Month"])
        
        # Resample based on selected period
        if time_period == "Hour":
            activity = time_df.resample('H').size().reset_index()
            time_format = '%Y-%m-%d %H:00'
        elif time_period == "Day":
            activity = time_df.resample('D').size().reset_index()
            time_format = '%Y-%m-%d'
        elif time_period == "Week":
            activity = time_df.resample('W').size().reset_index()
            time_format = '%Y-%m-%d'
        else:  # Month
            activity = time_df.resample('M').size().reset_index()
            time_format = '%Y-%m'
        
        activity.columns = [time_col, 'Count']
        
        # Format the time column for display
        activity[time_col] = activity[time_col].dt.strftime(time_format)
        
        # Plot activity
        fig = px.line(activity, x=time_col, y='Count',
                      title=f'User Activity by {time_period}',
                      labels={'Count': 'Number of Actions'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No timestamp columns detected in the data.")
    
    # User engagement analysis
    st.markdown("### User Engagement Analysis")
    
    if user_id_cols and behavior_cols:
        # Select a user ID column
        user_id_col = st.selectbox("Select user ID column:", user_id_cols)
        
        # Select behavioral metrics to analyze
        selected_metrics = st.multiselect(
            "Select behavioral metrics:", 
            behavior_cols,
            default=behavior_cols[:min(3, len(behavior_cols))]
        )
        
        if selected_metrics:
            # Group by user and calculate aggregate metrics
            user_metrics = df.groupby(user_id_col)[selected_metrics].agg(['count', 'mean', 'sum']).reset_index()
            
            # Format column names
            user_metrics.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in user_metrics.columns]
            
            # Display user engagement metrics
            st.markdown("#### Top Users by Engagement")
            
            # Sort by first sum column if available, otherwise by count
            sort_cols = [col for col in user_metrics.columns if col.endswith('_sum')]
            if not sort_cols:
                sort_cols = [col for col in user_metrics.columns if col.endswith('_count')]
            
            if sort_cols:
                sorted_users = user_metrics.sort_values(sort_cols[0], ascending=False).head(20)
                st.dataframe(sorted_users)
                
                # Visualize distribution of user engagement
                st.markdown("#### Distribution of User Engagement")
                
                for metric in selected_metrics:
                    sum_col = f"{metric}_sum"
                    if sum_col in user_metrics.columns:
                        fig = px.histogram(user_metrics, x=sum_col,
                                          title=f'Distribution of Total {metric} per User',
                                          labels={sum_col: f'Total {metric}'})
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No suitable metrics found for sorting users by engagement.")
        else:
            st.warning("Please select at least one behavioral metric.")
    else:
        st.warning("Need both user ID and behavioral columns for engagement analysis.")
    
    # User journey analysis (if we have event types and timestamps)
    event_type_cols = [col for col in df.columns if 'event' in col.lower() 
                       or 'action' in col.lower() or 'type' in col.lower()]
    
    if user_id_cols and event_type_cols and time_cols:
        st.markdown("### User Journey Analysis")
        
        # Select columns for journey analysis
        user_col = st.selectbox("Select user ID column for journey:", user_id_cols, key="journey_user")
        event_col = st.selectbox("Select event type column:", event_type_cols)
        time_col_journey = st.selectbox("Select timestamp column for journey:", time_cols, key="journey_time")
        
        # Convert timestamp if needed
        if not pd.api.types.is_datetime64_dtype(df[time_col_journey]):
            try:
                journey_df = df.copy()
                journey_df[time_col_journey] = pd.to_datetime(journey_df[time_col_journey])
            except:
                st.error(f"Could not convert {time_col_journey} to datetime.")
                return
        else:
            journey_df = df
        
        # Sort by user and timestamp
        journey_df = journey_df.sort_values([user_col, time_col_journey])
        
        # Get top event sequences
        st.markdown("#### Common Event Sequences")
        
        # Group by user and aggregate event types in order
        user_journeys = journey_df.groupby(user_col).apply(
            lambda x: '→'.join(x[event_col].astype(str))
        ).reset_index()
        user_journeys.columns = [user_col, 'Journey']
        
        # Count common journeys (simplify by taking first 5 steps)
        simplified_journeys = user_journeys['Journey'].apply(
            lambda x: '→'.join(x.split('→')[:5]) + ('→...' if len(x.split('→')) > 5 else '')
        )
        
        journey_counts = simplified_journeys.value_counts().reset_index()
        journey_counts.columns = ['Journey', 'Count']
        
        # Display top journeys
        st.dataframe(journey_counts.head(10))
    else:
        st.info("User journey analysis requires user ID, event type, and timestamp columns.")
