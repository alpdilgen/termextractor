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
        # --- UI Elements for Progress Tracking ---
        status_text = st.empty()
        
        st.markdown("##### Phase 1: File Analysis")
        progress_bar_1 = st.progress(0)
        
        st.markdown("##### Phase 2: AI Term Extraction")
        progress_bar_2 = st.progress(0)
        
        # --- Run Extraction ---
        try:
            # The logic function now accepts the Streamlit elements to update them
            json_result_file, excel_result_file, detected_source, detected_target = run_extraction_process(
                api_key=api_key,
                uploaded_files=uploaded_files,
                st_status=status_text,
                st_progress1=progress_bar_1,
                st_progress2=progress_bar_2
            )

            if excel_result_file and os.path.exists(excel_result_file):
                status_text.success("Term extraction complete!")
                
                # Display detected languages
                st.subheader("Extraction Summary")
                col1, col2 = st.columns(2)
                col1.metric("Source Language", detected_source.upper())
                col2.metric("Target Language", detected_target.upper())

                # Display results in a table
                st.subheader("Extracted Terms Overview")
                df = pd.read_excel(excel_result_file) # Read the first sheet by default
                st.dataframe(df)
                
                # Provide download buttons
                st.subheader("Download Full Results")
                d_col1, d_col2 = st.columns(2)
                
                with open(excel_result_file, "rb") as f:
                    d_col1.download_button(
                        label="📥 Download Excel Report",
                        data=f,
                        file_name=os.path.basename(excel_result_file),
                        mime="application/vnd.ms-excel"
                    )
                
                with open(json_result_file, "rb") as f:
                    d_col2.download_button(
                        label="📥 Download JSON Data",
                        data=f,
                        file_name=os.path.basename(json_result_file),
                        mime="application/json"
                    )
            else:
                status_text.error("Term extraction failed. No results were generated.")
                
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            st.error(traceback.format_exc())

