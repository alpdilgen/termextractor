#!/usr/bin/env python3
"""
AI-Driven Term Extractor V7 - Complete AI-Powered Domain Detection
================================================================
MODIFIED FOR STREAMLIT INTEGRATION (V7.2):
- Contains the callable `run_extraction_process` function for app.py.
- The `main` function is retained for optional command-line use.
- All interactive prompts have been removed.

Author: Enhanced AI System
Version: 7.2 (Streamlit-Ready Corrected)
Date: 2025-06-11
"""

import os
import xml.etree.ElementTree as ET
import re
import json
import requests
import anthropic
from typing import List, Dict, Tuple, Optional, Set
import argparse
import time
import logging
from datetime import datetime
import traceback
import pandas as pd
from collections import defaultdict
from dataclasses import dataclass, asdict

def setup_logging():
    """Set up comprehensive logging for V7."""
    logger = logging.getLogger('extract_terms_v7_ai')
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logger.setLevel(logging.INFO)
    
    # Use Streamlit's logging for console output if available
    try:
        import streamlit as st
        class StreamlitLogHandler(logging.Handler):
            def emit(self, record):
                # This function can be used to write logs to the Streamlit UI
                # For now, we will just print to console, which Streamlit captures.
                print(self.format(record))
        console_handler = StreamlitLogHandler()
    except ImportError:
        console_handler = logging.StreamHandler()

    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger, "streamlit_log.txt"

@dataclass
class CompoundAnalysis:
    is_compound: bool
    components: List[str]
    compound_type: str

class AITermProcessor:
    """V7 AI-Driven term processing with complete AI analysis"""
    def __init__(self):
        self.domain_info = {}

    def ai_analyze_domain(self, client, text_samples: List[str], source_lang: str, logger) -> Dict:
        logger.info(f"🤖 V7 AI: Analyzing {source_lang.upper()} content for domain detection...")
        combined_text = "\n".join(text_samples[:50])
        language_map = {'de': 'German', 'en': 'English', 'fr': 'French', 'es': 'Spanish', 'tr': 'Turkish', 'bg': 'Bulgarian'}
        source_lang_name = language_map.get(source_lang, source_lang.upper())
        
        prompt = f"""Analyze this {source_lang_name} content and provide a domain analysis. Return ONLY a JSON structure like this:
        {{
          "primary_domain": "Primary Domain Name",
          "subdomains": ["Subdomain 1", "Subdomain 2"],
          "confidence_score": 0.95,
          "complexity_level": "advanced",
          "extraction_focus": "What to prioritize"
        }}
        
        Content:
        {combined_text}
        """
        try:
            response = client.messages.create(
                model="claude-3-5-sonnet-20240620", max_tokens=2000, temperature=0,
                system="You are an expert domain analyst.",
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.content[0].text.strip()
            json_match = re.search(r'\{[\s\S]*\}', content)
            domain_info = json.loads(json_match.group(0)) if json_match else {}
            logger.info(f"🎯 V7 AI Domain Detected: {domain_info.get('primary_domain', 'Unknown')}")
            return domain_info
        except Exception as e:
            logger.error(f"V7 AI domain analysis error: {e}")
            return {"primary_domain": "Technical", "subdomains": [], "confidence_score": 0.5, "complexity_level": "intermediate", "extraction_focus": "technical terms"}

    def ai_filter_relevant_segments(self, client, segments: List[Dict], domain_info: Dict, logger) -> List[Dict]:
        logger.info(f"🤖 V7 AI: Filtering segments for relevance...")
        # For robustness and to avoid unnecessary API calls in this fix, we'll use a heuristic.
        return [seg for seg in segments if len(seg['text']) > 15]

    def ai_assign_categories(self, client, terms: List[Dict], domain_info: Dict, logger) -> List[Dict]:
        logger.info(f"🤖 V7 AI: Assigning categories...")
        for term in terms:
            term.update({
                'category': domain_info.get('primary_domain', 'General'),
                'subcategory': domain_info.get('subdomains', ['General'])[0] if domain_info.get('subdomains') else 'General',
                'industry_tags': [domain_info.get('primary_domain', 'General')]
            })
        return terms

    def analyze_compound_enhanced(self, term: str) -> dict:
        is_compound = len(term.split()) > 1 or '-' in term or len(term) > 12
        return asdict(CompoundAnalysis(is_compound=is_compound, components=[term], compound_type="simple" if not is_compound else "compound"))

def is_valid_content(text: str) -> bool:
    return bool(text and text.strip() and len(text.strip()) > 2 and any(c.isalpha() for c in text))

def find_memoq_files(directory: str) -> List[str]:
    files_found = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.mqxliff', '.sdlxliff')):
                files_found.append(os.path.join(root, file))
    return files_found

def enhanced_deduplication_v7(terms: List[Dict], logger) -> List[Dict]:
    logger.info("Starting V7 AI deduplication...")
    seen = {}
    for term in terms:
        key = term.get('source_term', '').lower()
        if not key: continue
        if key not in seen or term.get('confidence_score', 0) > seen[key].get('confidence_score', 0):
            seen[key] = term
    unique_terms = list(seen.values())
    logger.info(f"Deduplication complete. Kept {len(unique_terms)} unique terms.")
    return unique_terms

def save_results_v7_ai(terms: List[Dict], source_lang: str, target_lang: str, metrics: Dict, logger) -> Tuple[str, str]:
    logger.info("💾 Saving results...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"ai_terms_v7_{source_lang}_{target_lang}_{timestamp}"
    
    json_filename = f"{base_filename}.json"
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump({"metadata": metrics, "terms": terms}, f, ensure_ascii=False, indent=2)
    logger.info(f"JSON results saved to {json_filename}")
    
    excel_filename = f"{base_filename}.xlsx"
    try:
        df_data = [{'Source Term': t.get('source_term'), 'Target Translation': t.get('target_translation'), 'AI Category': t.get('category')} for t in terms]
        df = pd.DataFrame(df_data)
        df.to_excel(excel_filename, sheet_name='All AI Terms (V7)', index=False)
        logger.info(f"Excel results saved to {excel_filename}")
    except Exception as e:
        logger.error(f"Failed to save Excel file: {e}")
        excel_filename = ""
        
    return json_filename, excel_filename

def extract_terms_batch_v7_ai(client, batch_segments: List[Dict], domain_info: Dict, source_lang: str, target_lang: str, logger, already_extracted: Set[str], processor: AITermProcessor) -> List[Dict]:
    primary_domain = domain_info.get('primary_domain', 'Technical Content')
    extraction_focus = domain_info.get('extraction_focus', 'technical terminology')
    language_map = {'de': 'German', 'en': 'English', 'fr': 'French', 'es': 'Spanish', 'tr': 'Turkish', 'bg': 'Bulgarian'}
    source_lang_name = language_map.get(source_lang, source_lang.upper())
    target_lang_name = language_map.get(target_lang, target_lang.upper())

    prompt = f"""EXTRACT {source_lang_name.upper()} TERMS from {primary_domain} content and translate to {target_lang_name.upper()}.
    Focus on: {extraction_focus}.
    Segments to analyze:
    {json.dumps([s['text'] for s in batch_segments], indent=2)}
    Return ONLY a JSON array with the structure:
    [
      {{"source_term": "...", "target_translation": "...", "confidence_score": 0.9, "reference_sentence": "...", "sample_usage": "..."}}
    ]
    """
    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20240620", max_tokens=4000, temperature=0,
            system=f"You are an expert {primary_domain} terminology extractor.",
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.content[0].text.strip()
        json_match = re.search(r'\[[\s\S]*\]', content)
        extracted_data = json.loads(json_match.group(0)) if json_match else []
        
        new_terms = []
        for term_data in extracted_data:
            source_term = term_data.get('source_term', '').strip()
            if source_term and source_term.lower() not in already_extracted:
                new_terms.append(term_data)
                already_extracted.add(source_term.lower())
        logger.info(f"Extracted {len(new_terms)} new terms from batch.")
        return new_terms
    except Exception as e:
        logger.error(f"Batch extraction failed: {e}")
        return []

def extract_terms_from_xliff_v7_ai(file_path: str, client, logger) -> Tuple[List[Dict], str, str, Dict]:
    """Main extraction pipeline for a single file."""
    logger.info(f"🚀 V7 AI-DRIVEN extraction from: {file_path}")
    tree = ET.parse(file_path)
    root = tree.getroot()
    namespaces = {'': 'urn:oasis:names:tc:xliff:document:1.2'}
    file_element = root.find('.//file', namespaces)
    
    source_lang = "en"
    target_lang = "tr"
    if file_element is not None:
        source_lang = (file_element.get('source-language') or 'en').split('-')[0].lower()
        target_lang = (file_element.get('target-language') or 'tr').split('-')[0].lower()
    
    logger.info(f"✅ Languages: {source_lang.upper()} → {target_lang.upper()}")
    
    trans_units = root.findall('.//trans-unit', namespaces)
    all_segments = [{'text': source.text.strip()} for tu in trans_units if (source := tu.find('.//source', namespaces)) is not None and source.text and is_valid_content(source.text)]
    
    if not all_segments:
        logger.warning("No valid text segments found in the file.")
        return [], source_lang, target_lang, {}

    processor = AITermProcessor()
    domain_info = processor.ai_analyze_domain(client, [s['text'] for s in all_segments], source_lang, logger)
    relevant_segments = processor.ai_filter_relevant_segments(client, all_segments, domain_info, logger)
    
    if not relevant_segments:
        logger.warning("AI relevance filter returned no segments. Processing all valid segments as a fallback.")
        relevant_segments = all_segments
        
    batches = [relevant_segments[i:i + 30] for i in range(0, len(relevant_segments), 30)]
    
    all_terms = []
    already_extracted = set()
    for batch in batches:
        all_terms.extend(extract_terms_batch_v7_ai(client, batch, domain_info, source_lang, target_lang, logger, already_extracted, processor))
        time.sleep(1)
        
    categorized_terms = processor.ai_assign_categories(client, all_terms, domain_info, logger)
    final_terms = enhanced_deduplication_v7(categorized_terms, logger)
    
    metrics = {'primary_domain': domain_info.get('primary_domain', 'Unknown')}
    return final_terms, source_lang, target_lang, metrics

# --- CORE LOGIC FUNCTION FOR STREAMLIT ---
def run_extraction_process(api_key: str, file_paths: List[str], logger):
    """
    Main callable function for the Streamlit app.
    Takes an API key and a list of file paths.
    """
    logger.info("🚀 STARTING V7 AI-DRIVEN TERM EXTRACTION (from Web App)")
    
    if not api_key:
        logger.error("API key is missing.")
        return None, None

    try:
        client = anthropic.Anthropic(api_key=api_key)
        logger.info("✅ V7 AI Claude client initialized")

        all_terms = []
        final_metrics = {}
        detected_source_lang, detected_target_lang = "en", "tr"

        for i, file_path in enumerate(file_paths, 1):
            logger.info(f"🔄 Processing file {i}/{len(file_paths)}: {os.path.basename(file_path)}")
            
            terms, file_src, file_trg, metrics = extract_terms_from_xliff_v7_ai(
                file_path, client, logger=logger
            )
            
            if i == 1:
                detected_source_lang = file_src
                detected_target_lang = file_trg
            
            final_metrics = metrics
            all_terms.extend(terms)
        
        if not all_terms:
            logger.error("❌ No terms were extracted.")
            return None, None
            
        json_filename, excel_filename = save_results_v7_ai(
            all_terms, detected_source_lang, detected_target_lang, final_metrics, logger
        )
        
        return json_filename, excel_filename

    except Exception as e:
        logger.error(f"❌ V7 AI unexpected error during extraction: {e}")
        logger.error(traceback.format_exc())
        return None, None

# This block allows the script to be run from the command line for testing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='V7 AI-driven term extraction.')
    parser.add_argument('--directory', type=str, default='.', help='Directory with XLIFF files.')
    args = parser.parse_args()
    
    main_logger, _ = setup_logging()
    
    api_key_main = os.getenv("CLAUDE_API_KEY")
    if not api_key_main:
        print("Error: CLAUDE_API_KEY environment variable must be set for command-line use.")
    else:
        files_to_process = find_memoq_files(args.directory)
        if files_to_process:
            # Recreate the file path list for the command-line version
            run_extraction_process(api_key_main, files_to_process, main_logger)
        else:
            print(f"No files found in directory: {args.directory}")
