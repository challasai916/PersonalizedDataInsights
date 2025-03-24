import streamlit as st
import pandas as pd
from utils.data_loader import load_file
from utils.data_processor import preprocess_user_data

def show():
    """Display the data upload page."""
    st.title("Data Upload and Processing")
    
    st.markdown("""
    ## Upload your data
    
    Upload your user behavior data in CSV or JSON format. The platform will automatically process
    and prepare it for analysis.
    
    Supported file formats:
    - CSV (Comma-Separated Values)
    - JSON (JavaScript Object Notation)
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "json"])
    
    if uploaded_file is not None:
        # Display file details
        st.write(f"File name: {uploaded_file.name}")
        st.write(f"File size: {uploaded_file.size} bytes")
        
        # Load the data
        with st.spinner("Loading data..."):
            df = load_file(uploaded_file)
        
        if df is not None:
            # Store the original filename
            st.session_state.filename = uploaded_file.name
            
            # Display data preview
            st.markdown("### Data Preview")
            st.dataframe(df.head())
            
            # Display basic statistics
            st.markdown("### Basic Statistics")
            st.write(f"Number of rows: {df.shape[0]}")
            st.write(f"Number of columns: {df.shape[1]}")
            
            # Process the data
            with st.spinner("Processing data..."):
                processed_df, summary = preprocess_user_data(df)
            
            if processed_df is not None:
                # Store the processed data in session state
                st.session_state.data = processed_df
                
                # Display processing summary
                st.markdown("### Processing Summary")
                
                # Create two columns for the summary
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"Original rows: {summary['row_count']}")
                    st.write(f"Columns: {summary['column_count']}")
                    st.write(f"Missing values: {summary['missing_values']}")
                    st.write(f"Duplicates: {summary['duplicates']}")
                
                with col2:
                    if 'after_dedup_count' in summary:
                        st.write(f"Rows after deduplication: {summary['after_dedup_count']}")
                    
                    if 'potential_user_id_columns' in summary:
                        st.write("Potential user ID columns: ")
                        st.write(", ".join(summary['potential_user_id_columns']))
                    
                    if 'behavioral_columns' in summary:
                        st.write("Detected behavioral columns: ")
                        st.write(", ".join(summary['behavioral_columns'][:5]) + 
                                 (", ..." if len(summary['behavioral_columns']) > 5 else ""))
                
                # Display processed data
                st.markdown("### Processed Data Preview")
                st.dataframe(processed_df.head())
                
                # Success message
                st.success("Data has been successfully loaded and processed. You can now proceed to the analysis pages.")
