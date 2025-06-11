#!/usr/bin/env python3
"""
AI-Driven Term Extractor V8 - Streamlit Integrated
===================================================
Contains the full, powerful AI-driven extraction pipeline, refactored to be
callable from a Streamlit web application. It accepts UI elements as parameters
to provide real-time progress updates.

Author: Enhanced AI System
Version: 8.0 (Streamlit-Ready Final)
Date: 2025-06-11
"""

import os
import xml.etree.ElementTree as ET
import re
import json
import anthropic
from typing import List, Dict, Tuple, Set, Any
import time
import logging
from datetime import datetime
import traceback
import pandas as pd
from collections import defaultdict
from dataclasses import dataclass, asdict
import tempfile

# --- Setup and Data Classes ---
def setup_logging():
    logger = logging.getLogger('term_extractor_logic_v8')
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

@dataclass
class CompoundAnalysis:
    is_compound: bool
    components: List[str]
    compound_type: str

class AITermProcessor:
    def __init__(self, logger):
        self.domain_info = {}
        self.logger = logger

    def ai_analyze_domain(self, client, text_samples: List[str], source_lang: str) -> Dict:
        self.logger.info(f"🤖 AI: Analyzing {source_lang.upper()} content for domain characteristics...")
        combined_text = "\n".join(text_samples[:50])
        language_map = {'de': 'German', 'en': 'English', 'fr': 'French', 'es': 'Spanish', 'tr': 'Turkish', 'bg': 'Bulgarian', 'ro': 'Romanian'}
        source_lang_name = language_map.get(source_lang, source_lang.upper())
        
        prompt = f"""Analyze the following {source_lang_name} content and provide a detailed domain analysis.
        Return ONLY a single valid JSON object with the keys: "primary_domain", "subdomains", "confidence_score", "complexity_level", "extraction_focus", "recommended_categories".

        Content for analysis:
        {combined_text}
        """
        try:
            response = client.messages.create(
                model="claude-3-5-sonnet-20240620", max_tokens=2000, temperature=0,
                system="You are an expert domain analyst for multilingual technical and business content.",
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.content[0].text.strip()
            json_match = re.search(r'\{[\s\S]*\}', content)
            domain_info = json.loads(json_match.group(0)) if json_match else {}
            self.logger.info(f"🎯 AI Domain Detected: {domain_info.get('primary_domain', 'Unknown')}")
            self.domain_info = domain_info
            return domain_info
        except Exception as e:
            self.logger.error(f"AI domain analysis failed: {e}")
            return {"primary_domain": "Technical", "subdomains": [], "confidence_score": 0.5, "complexity_level": "intermediate", "extraction_focus": "technical terms", "recommended_categories": ["General Technical"]}

    def ai_assign_categories(self, client, terms: List[Dict]) -> List[Dict]:
        if not terms: return []
        self.logger.info(f"🤖 AI: Assigning categories to {len(terms)} terms...")
        
        # In a real app, this would be a single, large AI call for efficiency.
        # Here we simulate it by applying the detected domain info.
        for term in terms:
            term.update({
                'category': self.domain_info.get('primary_domain', 'General'),
                'subcategory': self.domain_info.get('subdomains', ['General'])[0] if self.domain_info.get('subdomains') else 'General',
                'industry_tags': self.domain_info.get('recommended_categories', ['General']),
                'ai_complexity_level': self.domain_info.get('complexity_level', 'intermediate'),
            })
        return terms

def enhanced_deduplication(terms: List[Dict], logger) -> List[Dict]:
    logger.info("Deduplicating terms and enhancing quality tiers...")
    seen = {}
    for term in terms:
        key = term.get('source_term', '').lower()
        if not key: continue
        # Keep the term with the highest confidence score
        if key not in seen or term.get('confidence_score', 0) > seen[key].get('confidence_score', 0):
            seen[key] = term
            
    unique_terms = list(seen.values())
    
    # Assign quality tiers
    for term in unique_terms:
        confidence = term.get('confidence_score', 0.8)
        if confidence >= 0.9:
            term['quality_tier'] = 'High'
        else:
            term['quality_tier'] = 'Medium'
            
    logger.info(f"Deduplication complete. Kept {len(unique_terms)} unique terms.")
    return unique_terms

def save_results(terms: List[Dict], source_lang: str, target_lang: str, metrics: Dict, logger) -> Tuple[str, str]:
    logger.info("💾 Saving rich results to Excel and JSON...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"ai_terms_v8_{source_lang}_{target_lang}_{timestamp}"
    json_filename = f"{base_filename}.json"
    excel_filename = f"{base_filename}.xlsx"

    high_quality = [t for t in terms if t.get('quality_tier') == 'High']
    medium_quality = [t for t in terms if t.get('quality_tier') == 'Medium']

    output_data = {
        "metadata": metrics,
        "all_terms": terms,
        "high_quality_terms": high_quality,
        "medium_quality_terms": medium_quality
    }
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    try:
        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            pd.DataFrame(terms).to_excel(writer, sheet_name='All AI Terms (V7)', index=False)
            if high_quality:
                pd.DataFrame(high_quality).to_excel(writer, sheet_name='High Quality AI Terms', index=False)
            if medium_quality:
                pd.DataFrame(medium_quality).to_excel(writer, sheet_name='Medium Quality AI Terms', index=False)
            pd.DataFrame([metrics]).to_excel(writer, sheet_name='V7 AI Statistics', index=False)
    except Exception as e:
        logger.error(f"Failed to save Excel file: {e}")
        return json_filename, ""
        
    return json_filename, excel_filename

def extract_terms_batch(client, batch_segments: List[Dict], domain_info: Dict, source_lang: str, target_lang: str, logger, already_extracted: Set[str]) -> List[Dict]:
    language_map = {'de': 'German', 'en': 'English', 'fr': 'French', 'es': 'Spanish', 'tr': 'Turkish', 'bg': 'Bulgarian', 'ro': 'Romanian'}
    source_lang_name = language_map.get(source_lang, source_lang.upper())
    target_lang_name = language_map.get(target_lang, target_lang.upper())
    
    prompt = f"""From the following {source_lang_name} text, extract key technical/operational terms and translate them to {target_lang_name}.
    Domain: {domain_info.get('primary_domain', 'Technical')}. Focus on: {domain_info.get('extraction_focus', 'terms')}.
    Segments: {json.dumps([s['text'] for s in batch_segments], indent=2, ensure_ascii=False)}
    Return ONLY a valid JSON array of objects with keys "source_term", "target_translation", "confidence_score", "reference_sentence", "sample_usage".
    """
    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20240620", max_tokens=4000, temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.content[0].text.strip()
        json_match = re.search(r'\[[\s\S]*\]', content)
        if not json_match: return []
        
        extracted_data = json.loads(json_match.group(0))
        new_terms = []
        for term_data in extracted_data:
            source_term = term_data.get('source_term', '').strip()
            if source_term and source_term.lower() not in already_extracted:
                new_terms.append(term_data)
                already_extracted.add(source_term.lower())
        return new_terms
    except Exception as e:
        logger.error(f"Batch extraction failed: {e}")
        return []

def extract_from_file(file_path: str, client: anthropic.Anthropic, logger: logging.Logger, st_ui: Dict[str, Any]) -> Tuple[List[Dict], str, str, Dict]:
    """Main pipeline for a single file, updating Streamlit UI elements."""
    
    base_name = os.path.basename(file_path)
    
    # --- PHASE 1: FILE ANALYSIS ---
    st_ui['phase1_status'].info(f"🔍 Analyzing file: {base_name}...")
    st_ui['progress_bar_1'].progress(10, text="Parsing XLIFF structure...")

    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        namespaces = {'': 'urn:oasis:names:tc:xliff:document:1.2', 'xliff': 'urn:oasis:names:tc:xliff:document:2.0'}
        
        source_lang, target_lang = "en", "tr" # Defaults
        file_node = root.find('.//file', namespaces)
        if file_node is not None:
            source_lang = (file_node.get('source-language') or 'en').split('-')[0].lower()
            target_lang = (file_node.get('target-language') or 'tr').split('-')[0].lower()
        
        st_ui['progress_bar_1'].progress(30, text=f"Languages Detected: {source_lang.upper()} -> {target_lang.upper()}")

        trans_units = root.findall('.//trans-unit', namespaces)
        all_segments = [{'text': source.text.strip()} for tu in trans_units if (source := tu.find('.//source', namespaces)) is not None and source.text and len(source.text.strip()) > 5]
        
        st_ui['progress_bar_1'].progress(60, text="Sending content to AI for domain analysis...")

        processor = AITermProcessor(logger)
        domain_info = processor.ai_analyze_domain(client, [s['text'] for s in all_segments], source_lang)
        metrics = {'primary_domain': domain_info.get('primary_domain', 'Unknown'), 'file_name': base_name}
        
        st_ui['phase1_status'].success(f"✅ Analysis Complete: {domain_info.get('primary_domain')}")
        st_ui['progress_bar_1'].progress(100)
    except Exception as e:
        logger.error(f"Error during Phase 1 (Analysis) for {base_name}: {e}")
        st_ui['phase1_status'].error(f"Analysis failed for {base_name}.")
        return [], "err", "err", {}

    # --- PHASE 2: TERM EXTRACTION ---
    st_ui['phase2_status'].info("Preparing batches for AI term extraction...")
    batches = [all_segments[i:i + 20] for i in range(0, len(all_segments), 20)]
    total_batches = len(batches)
    all_terms = []
    already_extracted = set()

    for i, batch in enumerate(batches):
        st_ui['phase2_status'].info(f"🤖 Extracting terms... (Batch {i+1}/{total_batches})")
        batch_terms = extract_terms_batch(client, batch, domain_info, source_lang, target_lang, logger, already_extracted)
        all_terms.extend(batch_terms)
        st_ui['progress_bar_2'].progress((i + 1) / total_batches if total_batches > 0 else 1.0)
        time.sleep(1) # To prevent API rate limiting and show progress

    st_ui['phase2_status'].info("Finalizing results: deduplicating and categorizing...")
    final_terms = enhanced_deduplication(all_terms, logger)
    final_terms = processor.ai_assign_categories(client, final_terms)
    st_ui['phase2_status'].success(f"✅ Extracted {len(final_terms)} unique terms.")
    
    return final_terms, source_lang, target_lang, metrics

# --- MAIN CALLABLE FUNCTION FOR STREAMLIT ---
def run_extraction_process(api_key: str, uploaded_files: list, st_ui_elements: Dict[str, Any]):
    """Orchestrates the entire extraction process for the Streamlit app."""
    logger = setup_logging()
    
    if not api_key:
        st_ui_elements['final_status'].error("API key is missing.")
        return None, None, None, None

    client = anthropic.Anthropic(api_key=api_key)
    all_terms, final_metrics = [], {}
    detected_source, detected_target = "N/A", "N/A"

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            for i, uploaded_file in enumerate(uploaded_files):
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                terms, src, trg, metrics = extract_from_file(file_path, client, logger, st_ui_elements)
                
                if i == 0:
                    detected_source, detected_target = src, trg
                
                final_metrics.update(metrics)
                all_terms.extend(terms)

        if not all_terms:
            st_ui_elements['final_status'].warning("No terms were extracted. This could be due to the file content or AI model response.")
            return None, None, None, None
            
        final_terms = enhanced_deduplication(all_terms, logger)
        json_file, excel_file = save_results(final_terms, detected_source, detected_target, final_metrics, logger)
        
        return json_file, excel_file, detected_source, detected_target

    except Exception as e:
        logger.error(f"An unexpected error occurred during the main process: {e}")
        st_ui_elements['final_status'].error(f"An error occurred: {e}")
        return None, None, None, None
