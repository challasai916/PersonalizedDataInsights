import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from models.segmentation import perform_segmentation, describe_segments
from utils.data_processor import prepare_for_segmentation
from utils.visualization import plot_segmentation_results

def show():
    """Display the user segmentation page."""
    st.title("User Segmentation")
    
    # Check if data exists in session state
    if st.session_state.data is None:
        st.warning("Please upload data first on the 'Data Upload' page.")
        return
    
    df = st.session_state.data
    
    st.markdown("""
    ## Segment Your Users
    
    User segmentation groups similar users together based on their characteristics and behaviors.
    This can help you understand different user types and create targeted personalization strategies.
    """)
    
    # Check if we already have segments
    has_segments = st.session_state.segmentation_model is not None
    
    # Tabs for segmentation workflow
    tabs = st.tabs(["Data Preparation", "Create Segments", "Segment Analysis", "Segmentation Application"])
    
    # Data Preparation tab
    with tabs[0]:
        st.markdown("### Prepare Data for Segmentation")
        
        st.write("""
        For effective user segmentation, we need user attributes and behavior metrics:
        - Demographic information
        - Activity metrics
        - Preferences
        - Behavioral indicators
        """)
        
        # Prepare the data for segmentation
        with st.spinner("Preparing data for segmentation..."):
            try:
                seg_df, feature_cols = prepare_for_segmentation(df)
                has_user_ids = isinstance(feature_cols, tuple) and len(feature_cols) > 2
                
                if has_user_ids:
                    seg_df, feature_cols, user_ids = feature_cols
            except:
                seg_df, feature_cols = None, None
                has_user_ids = False
        
        if seg_df is not None and feature_cols is not None:
            st.success("Data successfully prepared for segmentation.")
            
            # Display selected features
            st.markdown("#### Selected Features for Segmentation")
            
            # Allow users to modify features
            selected_features = st.multiselect(
                "Select features to use for segmentation:",
                feature_cols,
                default=feature_cols
            )
            
            if selected_features:
                # Filter data to only include selected features
                seg_df_filtered = seg_df[selected_features]
                
                # Display data preview
                st.markdown("#### Data Preview")
                st.dataframe(seg_df_filtered.head())
                
                # Store the prepared data in session state
                st.session_state.seg_df = seg_df_filtered
                st.session_state.seg_features = selected_features
                if has_user_ids:
                    st.session_state.seg_user_ids = user_ids
            else:
                st.warning("Please select at least one feature for segmentation.")
        else:
            st.error("Could not prepare data for segmentation. Please ensure your data includes user attributes or behaviors.")
    
    # Create Segments tab
    with tabs[1]:
        st.markdown("### Create User Segments")
        
        if not hasattr(st.session_state, 'seg_df') or st.session_state.seg_df is None:
            st.warning("Please prepare your data in the 'Data Preparation' tab first.")
        else:
            # Segmentation parameters
            st.markdown("#### Segmentation Parameters")
            
            # Allow users to specify number of clusters or find optimal
            use_optimal = st.checkbox("Automatically find optimal number of segments", value=True)
            
            if use_optimal:
                max_clusters = st.slider("Maximum number of segments to consider", min_value=2, max_value=20, value=10)
                n_clusters = None
            else:
                n_clusters = st.slider("Number of segments", min_value=2, max_value=15, value=5)
                max_clusters = 10  # Not used but required for function call
            
            # Run segmentation button
            if st.button("Create Segments"):
                with st.spinner("Creating user segments... This may take a moment."):
                    # Get the data
                    seg_df = st.session_state.seg_df
                    feature_cols = st.session_state.seg_features
                    
                    # Perform segmentation
                    labels, centers, silhouette_scores, pca_result, explained_variance = perform_segmentation(
                        seg_df, n_clusters, max_clusters
                    )
                    
                    if labels is not None:
                        # Store results in session state
                        st.session_state.segmentation_model = {
                            'labels': labels,
                            'centers': centers,
                            'silhouette_scores': silhouette_scores,
                            'pca_result': pca_result,
                            'explained_variance': explained_variance,
                            'n_clusters': len(np.unique(labels))
                        }
                        
                        # Display segmentation results
                        st.success(f"Successfully created {len(np.unique(labels))} user segments!")
                        
                        # Display segment sizes
                        segment_counts = pd.Series(labels).value_counts().sort_index()
                        
                        st.markdown("#### Segment Sizes")
                        segment_data = pd.DataFrame({
                            'Segment': segment_counts.index,
                            'Number of Users': segment_counts.values,
                            'Percentage': (segment_counts.values / len(labels) * 100).round(2)
                        })
                        
                        st.dataframe(segment_data)
                        
                        # Plot segment distribution
                        fig = px.pie(segment_data, values='Number of Users', names='Segment',
                                    title='Distribution of Users Across Segments')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("Segmentation failed. Please try different parameters.")
            
            # Display existing segmentation if available
            elif has_segments:
                st.success(f"Users are already segmented into {st.session_state.segmentation_model['n_clusters']} segments.")
                
                # Show segment distribution
                labels = st.session_state.segmentation_model['labels']
                segment_counts = pd.Series(labels).value_counts().sort_index()
                
                st.markdown("#### Segment Sizes")
                segment_data = pd.DataFrame({
                    'Segment': segment_counts.index,
                    'Number of Users': segment_counts.values,
                    'Percentage': (segment_counts.values / len(labels) * 100).round(2)
                })
                
                st.dataframe(segment_data)
    
    # Segment Analysis tab
    with tabs[2]:
        st.markdown("### Analyze Segments")
        
        if not has_segments:
            st.warning("Please create user segments in the 'Create Segments' tab first.")
        else:
            # Get segmentation results
            model = st.session_state.segmentation_model
            seg_df = st.session_state.seg_df
            feature_cols = st.session_state.seg_features
            
            # Describe segments
            segment_description = describe_segments(seg_df, model['labels'], feature_cols)
            
            if segment_description is not None:
                # Display segment profiles
                st.markdown("#### Segment Profiles")
                st.dataframe(segment_description['profiles'])
                
                # Display relative comparison to overall average
                st.markdown("#### Segment Comparison (relative to average)")
                comparison_df = segment_description['comparisons']
                
                # Format as percentages and highlight differences
                formatted_df = comparison_df.copy()
                
                for col in formatted_df.columns:
                    formatted_df[col] = formatted_df[col].apply(lambda x: f"{x*100:.1f}%")
                
                st.dataframe(formatted_df)
                
                # Visualize segments
                st.markdown("#### Segment Visualization")
                plot_segmentation_results(seg_df, model['labels'], feature_cols)
                
                # PCA visualization if available
                if model['pca_result'] is not None and model['pca_result'].shape[1] >= 2:
                    st.markdown("#### 2D Visualization of Segments")
                    
                    # Create dataframe for plotting
                    pca_df = pd.DataFrame(
                        model['pca_result'],
                        columns=[f'PC{i+1}' for i in range(model['pca_result'].shape[1])]
                    )
                    pca_df['Segment'] = model['labels']
                    
                    # Calculate explained variance
                    explained_var = model['explained_variance']
                    var_labels = [f'PC{i+1} ({var*100:.1f}%)' for i, var in enumerate(explained_var)]
                    
                    # Create scatter plot
                    fig = px.scatter(
                        pca_df, x='PC1', y='PC2', color='Segment',
                        title='Segment Visualization using PCA',
                        labels={'PC1': var_labels[0], 'PC2': var_labels[1]}
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    # Segmentation Application tab
    with tabs[3]:
        st.markdown("### Apply Segmentation")
        
        if not has_segments:
            st.warning("Please create user segments in the 'Create Segments' tab first.")
        else:
            st.markdown("""
            #### Using Segments for Personalization
            
            Now that you've segmented your users, you can use these segments to:
            
            1. **Create targeted content or recommendations** for each segment
            2. **Customize user experiences** based on segment characteristics
            3. **Develop marketing strategies** tailored to each segment
            4. **Prioritize features** that are important to specific segments
            """)
            
            # Get segmentation results
            model = st.session_state.segmentation_model
            segment_description = describe_segments(st.session_state.seg_df, model['labels'], st.session_state.seg_features)
            
            # Show segment characteristics
            if segment_description is not None:
                st.markdown("#### Segment Characteristics and Recommendations")
                
                # Create tabs for each segment
                segment_tabs = st.tabs([f"Segment {i}" for i in range(model['n_clusters'])])
                
                for i, tab in enumerate(segment_tabs):
                    with tab:
                        # Segment size
                        segment_size = segment_description['counts'][i]
                        segment_pct = segment_size / sum(segment_description['counts']) * 100
                        
                        st.markdown(f"**Segment Size:** {segment_size} users ({segment_pct:.1f}% of total)")
                        
                        # Key characteristics
                        st.markdown("**Key Characteristics:**")
                        
                        # Get profile for this segment
                        profile = segment_description['profiles'].loc[i]
                        comparison = segment_description['comparisons'].loc[i]
                        
                        # Find distinguishing features (those that deviate most from average)
                        distinguishing_features = [(col, val, comparison[col]) 
                                                   for col, val in profile.items()]
                        
                        # Sort by absolute deviation from average
                        distinguishing_features.sort(key=lambda x: abs(x[2]-1), reverse=True)
                        
                        for feature, value, comp in distinguishing_features[:5]:
                            direction = "higher" if comp > 1 else "lower"
                            deviation = abs(comp - 1) * 100
                            st.write(f"- {feature}: {value:.2f} ({deviation:.1f}% {direction} than average)")
                        
                        # Personalization recommendations
                        st.markdown("**Personalization Recommendations:**")
                        
                        # Generate recommendations based on segment characteristics
                        recommendations = []
                        
                        for feature, value, comp in distinguishing_features[:3]:
                            if comp > 1.5:  # Much higher than average
                                recommendations.append(f"Emphasize content related to {feature}")
                            elif comp < 0.5:  # Much lower than average
                                recommendations.append(f"Minimize focus on {feature}-related content")
                            elif comp > 1.1:  # Slightly higher
                                recommendations.append(f"Include more {feature}-focused elements")
                            elif comp < 0.9:  # Slightly lower
                                recommendations.append(f"Reduce emphasis on {feature}")
                        
                        # Add general recommendations
                        if i == np.argmax(segment_description['counts']):
                            recommendations.append("This is your largest segment - prioritize their needs")
                        
                        # Display recommendations
                        for j, rec in enumerate(recommendations):
                            st.write(f"{j+1}. {rec}")
                
                # Export segment data
                st.markdown("#### Export Segment Data")
                
                if st.button("Prepare User Segment Export"):
                    # Combine original data with segment labels
                    if hasattr(st.session_state, 'seg_user_ids'):
                        # If we have user IDs
                        export_df = pd.DataFrame({
                            'User ID': st.session_state.seg_user_ids,
                            'Segment': model['labels']
                        })
                    else:
                        # Otherwise just use row indices
                        export_df = pd.DataFrame({
                            'Row Index': range(len(model['labels'])),
                            'Segment': model['labels']
                        })
                    
                    # Display export data
                    st.dataframe(export_df.head(100))
                    
                    # Provide export functionality
                    csv = export_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download Segment Data as CSV",
                        csv,
                        "user_segments.csv",
                        "text/csv",
                        key='download-csv'
                    )
