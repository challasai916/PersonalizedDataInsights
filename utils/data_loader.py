import streamlit as st
import pandas as pd
import json
import io

def load_csv(uploaded_file):
    """Load and validate a CSV file."""
    try:
        # Try to read with default settings
        df = pd.read_csv(uploaded_file)
        
        # Basic validation
        if df.empty:
            st.error("The uploaded file is empty.")
            return None
        
        return df
    except Exception as e:
        st.error(f"Error loading CSV file: {str(e)}")
        return None

def load_json(uploaded_file):
    """Load and validate a JSON file."""
    try:
        # Read the file content
        content = uploaded_file.read()
        
        # Parse JSON
        data = json.loads(content)
        
        # Convert to DataFrame
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            # Handle nested JSON structures
            if any(isinstance(v, (dict, list)) for v in data.values()):
                # Normalize semi-structured data
                df = pd.json_normalize(data)
            else:
                # Simple dictionary
                df = pd.DataFrame([data])
        else:
            st.error("Unsupported JSON structure.")
            return None
        
        # Basic validation
        if df.empty:
            st.error("The uploaded file contains no valid data.")
            return None
        
        return df
    except Exception as e:
        st.error(f"Error loading JSON file: {str(e)}")
        return None

def load_file(uploaded_file):
    """Load data from different file formats."""
    if uploaded_file is None:
        return None
    
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    if file_type == 'csv':
        return load_csv(uploaded_file)
    elif file_type == 'json':
        return load_json(uploaded_file)
    else:
        st.error(f"Unsupported file format: {file_type}. Please upload a CSV or JSON file.")
        return None
