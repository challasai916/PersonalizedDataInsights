import streamlit as st
import pandas as pd
import numpy as np
from utils.visualization import plot_data_overview, plot_correlation_matrix

def show():
    """Display the exploratory data analysis page."""
    st.title("Exploratory Data Analysis")
    
    # Check if data exists in session state
    if st.session_state.data is None:
        st.warning("Please upload data first on the 'Data Upload' page.")
        return
    
    df = st.session_state.data
    
    st.markdown("""
    ## Explore Your Data
    
    Exploratory Data Analysis (EDA) helps you understand your data, identify patterns, 
    detect outliers, and discover relationships between variables.
    """)
    
    # Tabs for different EDA sections
    tabs = st.tabs(["Data Overview", "Summary Statistics", "Correlation Analysis", "Column Analysis"])
    
    # Data Overview tab
    with tabs[0]:
        st.markdown("### Data Overview")
        st.write(f"Dataset: {st.session_state.filename}")
        st.write(f"Number of rows: {df.shape[0]}")
        st.write(f"Number of columns: {df.shape[1]}")
        
        # Data types and distribution
        st.markdown("#### Data Types")
        
        # Create a DataFrame with column types
        dtype_df = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.values,
            'Non-Null Count': df.count().values,
            'Null Count': df.isna().sum().values,
            'Null %': (df.isna().sum().values / len(df) * 100).round(2)
        })
        
        st.dataframe(dtype_df)
        
        # Data distribution visualizations
        st.markdown("#### Data Distributions")
        plot_data_overview(df)
    
    # Summary Statistics tab
    with tabs[1]:
        st.markdown("### Summary Statistics")
        
        # Numeric statistics
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            st.markdown("#### Numeric Columns")
            st.dataframe(df[numeric_cols].describe())
        
        # Categorical statistics
        cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        if cat_cols:
            st.markdown("#### Categorical Columns")
            
            for col in cat_cols:
                st.markdown(f"**{col}**")
                value_counts = df[col].value_counts().reset_index()
                value_counts.columns = [col, 'Count']
                value_counts['Percentage'] = (value_counts['Count'] / len(df) * 100).round(2)
                
                # Display top 10 categories
                st.dataframe(value_counts.head(10))
    
    # Correlation Analysis tab
    with tabs[2]:
        st.markdown("### Correlation Analysis")
        
        if len(df.select_dtypes(include=['number']).columns) < 2:
            st.warning("Need at least 2 numeric columns for correlation analysis.")
        else:
            # Plot correlation matrix
            plot_correlation_matrix(df)
            
            # Display correlation table
            st.markdown("#### Correlation Matrix")
            corr = df.select_dtypes(include=['number']).corr()
            st.dataframe(corr.style.background_gradient(cmap='coolwarm'))
    
    # Column Analysis tab
    with tabs[3]:
        st.markdown("### Column Analysis")
        
        # Column selector
        selected_column = st.selectbox("Select a column to analyze", df.columns)
        
        # Display column information
        st.markdown(f"#### Analysis of '{selected_column}'")
        
        # Basic stats
        st.write(f"Data type: {df[selected_column].dtype}")
        st.write(f"Number of unique values: {df[selected_column].nunique()}")
        st.write(f"Number of missing values: {df[selected_column].isna().sum()} ({df[selected_column].isna().sum() / len(df) * 100:.2f}%)")
        
        # For numeric columns
        if pd.api.types.is_numeric_dtype(df[selected_column]):
            st.markdown("##### Numeric Statistics")
            st.write(f"Min: {df[selected_column].min()}")
            st.write(f"Max: {df[selected_column].max()}")
            st.write(f"Mean: {df[selected_column].mean():.4f}")
            st.write(f"Median: {df[selected_column].median()}")
            st.write(f"Standard Deviation: {df[selected_column].std():.4f}")
            
            # Histogram
            st.markdown("##### Distribution")
            hist_values = np.histogram(df[selected_column].dropna(), bins=20)[0]
            st.bar_chart(hist_values)
        
        # For categorical/text columns
        else:
            st.markdown("##### Value Counts")
            value_counts = df[selected_column].value_counts().reset_index()
            value_counts.columns = [selected_column, 'Count']
            value_counts['Percentage'] = (value_counts['Count'] / len(df) * 100).round(2)
            
            # Display top 20 categories
            st.dataframe(value_counts.head(20))
            
            # Pie chart for top 10 categories
            if df[selected_column].nunique() <= 20:
                st.markdown("##### Distribution")
                st.bar_chart(value_counts.set_index(selected_column)['Count'].head(10))
