# app.py
import streamlit as st
import os
import tempfile
import pandas as pd
from term_extractor_logic import run_extraction_process, setup_logging

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Term Extractor",
    page_icon="🤖",
    layout="wide"
)

# --- Main UI ---
st.title("🤖 AI-Powered Term Extractor")
st.markdown("Upload your `.mqxliff` or `.sdlxliff` files to extract domain-specific terminology using AI.")

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("Configuration")
    
    # Securely get API key
    api_key = st.text_input("Enter your Claude API key", type="password")
    
    st.markdown("---")
    st.info("Your API key is handled securely and is not stored.")

# --- File Uploader ---
uploaded_files = st.file_uploader(
    "Upload Files",
    type=['mqxliff', 'sdlxliff'],
    accept_multiple_files=True
)

# --- Process Button and Logic ---
if st.button("Extract Terms"):
    if not api_key:
        st.error("Please enter your Claude API key in the sidebar.")
    elif not uploaded_files:
        st.warning("Please upload at least one file.")
    else:
        # Use a temporary directory to store uploaded files
        with tempfile.TemporaryDirectory() as temp_dir:
            file_paths = []
            for uploaded_file in uploaded_files:
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                file_paths.append(temp_path)
            
            # Setup logger to display status in the app
            logger, _ = setup_logging()
            
            with st.spinner("AI is analyzing your files and extracting terms... This may take a moment."):
                try:
                    # Run the core extraction logic
                    json_result_file, excel_result_file = run_extraction_process(api_key, file_paths, logger)

                    if excel_result_file and os.path.exists(excel_result_file):
                        st.success("Term extraction complete!")
                        
                        # Display results in a table
                        st.subheader("Extracted Terms Overview")
                        df = pd.read_excel(excel_result_file, sheet_name='All AI Terms (V7)')
                        st.dataframe(df)
                        
                        # Provide download buttons
                        st.subheader("Download Full Results")
                        col1, col2 = st.columns(2)
                        
                        with open(excel_result_file, "rb") as f:
                            col1.download_button(
                                label="📥 Download Excel Report",
                                data=f,
                                file_name=os.path.basename(excel_result_file),
                                mime="application/vnd.ms-excel"
                            )
                        
                        with open(json_result_file, "rb") as f:
                            col2.download_button(
                                label="📥 Download JSON Data",
                                data=f,
                                file_name=os.path.basename(json_result_file),
                                mime="application/json"
                            )
                    else:
                        st.error("Term extraction failed. Check the logs for details.")
                        
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")