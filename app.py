# app.py
import streamlit as st
import os
import tempfile
import pandas as pd
import traceback
from term_extractor_logic import run_extraction_process, setup_logging

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Term Extractor v8",
    page_icon="🤖",
    layout="wide"
)

# --- Main UI ---
st.title("🤖 AI-Powered Term Extractor (V8 Logic)")
st.markdown("Upload your `.mqxliff` or `.sdlxliff` files to extract domain-specific terminology using the full V8 AI-driven pipeline.")

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter your Claude API key", type="password", help="Your key is used for the API call and is not stored.")
    st.markdown("---")
    st.info("This app uses the most advanced extraction logic (V8). The process may take a few minutes per file.")

# --- File Uploader ---
uploaded_files = st.file_uploader(
    "Upload Your XLIFF Files",
    type=['mqxliff', 'sdlxliff'],
    accept_multiple_files=True
)

# --- Process Button and Logic ---
if st.button("🚀 Extract Terms"):
    if not api_key:
        st.error("Please enter your Claude API key in the sidebar.")
    elif not uploaded_files:
        st.warning("Please upload at least one file.")
    else:
        # --- UI Elements for Progress Tracking ---
        st.markdown("---")
        st.subheader("📊 Live Progress")
        
        phase1_status = st.empty()
        st.markdown("**Phase 1: File Analysis & AI Domain Detection**")
        progress_bar_1 = st.progress(0)
        
        phase2_status = st.empty()
        st.markdown("**Phase 2: AI Term Extraction & Categorization**")
        progress_bar_2 = st.progress(0)
        
        final_status = st.empty()

        # --- Run Extraction ---
        try:
            # The logic function now accepts the Streamlit elements to update them
            json_result_file, excel_result_file, detected_source, detected_target = run_extraction_process(
                api_key=api_key,
                uploaded_files=uploaded_files,
                st_ui_elements={
                    "phase1_status": phase1_status,
                    "progress_bar_1": progress_bar_1,
                    "phase2_status": phase2_status,
                    "progress_bar_2": progress_bar_2,
                    "final_status": final_status
                }
            )

            if excel_result_file and os.path.exists(excel_result_file):
                final_status.success("✅ Term extraction complete!")
                
                # Display detected languages
                st.subheader("Extraction Summary")
                col1, col2 = st.columns(2)
                col1.metric("Source Language", detected_source.upper())
                col2.metric("Target Language", detected_target.upper())

                # Display results in a table (from the main sheet)
                st.subheader("Extracted Terms Overview")
                df = pd.read_excel(excel_result_file, sheet_name='All AI Terms (V7)')
                st.dataframe(df)
                
                # Provide download buttons
                st.subheader("Download Full Results")
                d_col1, d_col2 = st.columns(2)
                
                with open(excel_result_file, "rb") as f:
                    d_col1.download_button(
                        label="📥 Download Full Excel Report",
                        data=f,
                        file_name=os.path.basename(excel_result_file),
                        mime="application/vnd.ms-excel"
                    )
                
                with open(json_result_file, "rb") as f:
                    d_col2.download_button(
                        label="📥 Download Full JSON Data",
                        data=f,
                        file_name=os.path.basename(json_result_file),
                        mime="application/json"
                    )
            else:
                final_status.error("Term extraction failed. No results were generated.")
                
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            st.error(traceback.format_exc())
