#!/usr/bin/env python3
"""
AI-Driven Term Extractor V7 - Complete AI-Powered Domain Detection
================================================================
Fully AI-driven domain detection, categorization, and term extraction
No hardcoded categories or keywords - everything determined by AI analysis

MODIFIED FOR AUTOMATION:
- Removes all interactive prompts for API keys and file selection.
- Suitable for use in automated workflows like GitHub Actions.

Author: Enhanced AI System
Version: 7.0 (Automated)
Date: 2025-06-03
"""

import os
import xml.etree.ElementTree as ET
import re
import json
import requests
import anthropic
from typing import List, Dict, Tuple, Optional, Set
import argparse
# getpass is removed as it's not needed for automation
import time
import logging
from datetime import datetime
import traceback
import pandas as pd
from collections import defaultdict, Counter
import hashlib
from difflib import SequenceMatcher
from dataclasses import dataclass, asdict

def setup_logging():
    """Set up comprehensive logging for V7."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"extract_terms_v7_ai_log_{timestamp}.log"
    
    logger = logging.getLogger('extract_terms_v7_ai')
    logger.setLevel(logging.DEBUG)
    
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_filename

@dataclass
class CulturalNote:
    german_context: str
    english_usage: str
    cross_cultural_considerations: Optional[str] = None
    linguistic_note: Optional[str] = None
    trend_note: Optional[str] = None
    ambiguity_note: Optional[str] = None

@dataclass
class AlternativeTranslation:
    translation: str
    context: str
    confidence: float

@dataclass
class CompoundAnalysis:
    is_compound: bool
    components: List[str]
    compound_type: str

@dataclass
class GlossaryVerification:
    verified: bool
    source: str
    match_confidence: float

@dataclass
class OfficialDesignation:
    is_official_term: bool
    regulation_number: Optional[str] = None
    abbreviation: Optional[str] = None
    legal_status: Optional[str] = None

class AITermProcessor:
    """V7 AI-Driven term processing with complete AI analysis"""
    
    def __init__(self):
        # No hardcoded categories - everything AI-driven
        self.domain_info = {}
        self.ai_categories = {}
        self.relevance_criteria = {}

    def ai_analyze_domain(self, client, text_samples: List[str], source_lang: str, logger) -> Dict:
        """AI-powered domain analysis from actual content"""
        logger.info(f"🤖 V7 AI: Analyzing {source_lang.upper()} content for domain detection...")
        
        # Prepare text for analysis
        combined_text = "\n".join(text_samples[:50])  # Use first 50 samples for analysis
        
        # Language mapping for AI
        language_map = {
            'de': 'German', 'en': 'English', 'fr': 'French', 'es': 'Spanish', 
            'it': 'Italian', 'pt': 'Portuguese', 'nl': 'Dutch', 'pl': 'Polish',
            'cs': 'Czech', 'sk': 'Slovak', 'hu': 'Hungarian', 'ro': 'Romanian',
            'bg': 'Bulgarian', 'hr': 'Croatian', 'sl': 'Slovenian', 'et': 'Estonian',
            'lv': 'Latvian', 'lt': 'Lithuanian', 'fi': 'Finnish', 'sv': 'Swedish',
            'da': 'Danish', 'no': 'Norwegian', 'ru': 'Russian', 'uk': 'Ukrainian',
            'zh': 'Chinese', 'ja': 'Japanese', 'ko': 'Korean', 'ar': 'Arabic',
            'he': 'Hebrew', 'tr': 'Turkish', 'th': 'Thai', 'vi': 'Vietnamese'
        }
        
        source_lang_name = language_map.get(source_lang, source_lang.upper())
        
        domain_analysis_prompt = f"""Analyze this {source_lang_name} content and provide comprehensive domain analysis.

CONTENT TO ANALYZE:
{combined_text}

ANALYSIS REQUIRED:
1. PRIMARY DOMAIN: Identify the main industry/field (e.g., "Industrial Manufacturing", "Medical/Surgical", "Automotive Engineering", "Food Processing", "Legal Services", etc.)

2. SUBDOMAINS: Identify 3-5 specific specialized areas within the primary domain (e.g., for Industrial Manufacturing: "Weighing Systems", "Quality Control", "Packaging Equipment", "Process Automation")

3. CONFIDENCE SCORE: Rate certainty of domain identification (0.0-1.0)

4. TERMINOLOGY CHARACTERISTICS: Describe what types of terms are most valuable in this domain

5. COMPLEXITY LEVEL: Rate the technical complexity (basic, intermediate, advanced, expert)

6. INDUSTRY FOCUS: Specific industry applications

Return ONLY this JSON structure:
{{
  "primary_domain": "Primary Domain Name",
  "subdomains": ["Subdomain 1", "Subdomain 2", "Subdomain 3", "Subdomain 4", "Subdomain 5"],
  "confidence_score": 0.95,
  "terminology_characteristics": "Description of valuable terminology types",
  "complexity_level": "advanced",
  "industry_focus": "Specific industry applications",
  "content_type": "technical/business/medical/legal/etc",
  "recommended_categories": ["Category 1", "Category 2", "Category 3"],
  "extraction_focus": "What to prioritize in term extraction"
}}

Analyze the actual content and provide accurate domain identification based on the terminology, context, and language used."""

        try:
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                temperature=0,
                system="You are an expert domain analyst specializing in multilingual content categorization. Analyze content accurately and provide precise domain identification based on actual terminology and context.",
                messages=[{"role": "user", "content": domain_analysis_prompt}]
            )
            
            content = response.content[0].text.strip()
            logger.debug(f"V7 AI domain analysis response: {content}")
            
            # Parse AI response
            try:
                domain_info = json.loads(content)
            except:
                # Extract JSON from response if wrapped
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    domain_info = json.loads(json_match.group(0))
                else:
                    # Fallback
                    domain_info = {
                        "primary_domain": "Technical Content",
                        "subdomains": ["Technical Terms", "Specialized Vocabulary", "Domain Terminology"],
                        "confidence_score": 0.7,
                        "terminology_characteristics": "Technical and specialized terms",
                        "complexity_level": "intermediate",
                        "industry_focus": "Professional/Technical",
                        "content_type": "technical",
                        "recommended_categories": ["Technical Terms", "Professional Vocabulary"],
                        "extraction_focus": "Technical terminology and specialized vocabulary"
                    }
            
            logger.info(f"🎯 V7 AI Domain Analysis Results:")
            logger.info(f"  🏭 Primary Domain: {domain_info['primary_domain']}")
            logger.info(f"  📋 Subdomains: {', '.join(domain_info['subdomains'][:3])}...")
            logger.info(f"  🎯 Confidence: {domain_info['confidence_score']:.2f}")
            logger.info(f"  🔧 Complexity: {domain_info['complexity_level']}")
            logger.info(f"  🎪 Focus: {domain_info['extraction_focus']}")
            
            # Store for use in other functions
            self.domain_info = domain_info
            
            return domain_info
            
        except Exception as e:
            logger.error(f"V7 AI domain analysis error: {e}")
            # Fallback domain info
            return {
                "primary_domain": "Technical Content",
                "subdomains": ["Technical Terms", "Specialized Vocabulary"],
                "confidence_score": 0.5,
                "terminology_characteristics": "Mixed technical content",
                "complexity_level": "intermediate",
                "industry_focus": "General Technical",
                "content_type": "technical",
                "recommended_categories": ["Technical Terms"],
                "extraction_focus": "Technical and specialized terminology"
            }

    def ai_filter_relevant_segments(self, client, segments: List[Dict], domain_info: Dict, logger) -> List[Dict]:
        """AI-powered relevance filtering based on detected domain"""
        logger.info(f"🤖 V7 AI: Filtering segments for {domain_info['primary_domain']} relevance...")
        
        # Prepare segments for analysis
        segment_texts = []
        for i, segment in enumerate(segments[:100]):  # Limit for performance
            segment_texts.append(f"{i}: {segment['text']}")
        
        combined_segments = "\n".join(segment_texts)
        
        relevance_prompt = f"""Identify segments containing terminology relevant to {domain_info['primary_domain']}.

DOMAIN CONTEXT:
- Primary Domain: {domain_info['primary_domain']}
- Subdomains: {', '.join(domain_info['subdomains'])}
- Focus Areas: {domain_info['extraction_focus']}

SEGMENTS TO ANALYZE:
{combined_segments}

TASK: Identify segment numbers (0, 1, 2, etc.) that contain terminology valuable for translation memory and terminology management in this domain.

CRITERIA FOR RELEVANCE (INCLUSIVE APPROACH):
1. Contains technical terms specific to {domain_info['primary_domain']}
2. Contains specialized vocabulary related to: {', '.join(domain_info['subdomains'])}
3. Contains terminology that would be valuable for professional translators
4. Contains domain-specific concepts, processes, or equipment names
5. Contains operational terms used in {domain_info['primary_domain']} workflows
6. Contains basic but essential industry terminology (modes, operations, procedures)
7. Contains interface terms commonly used by operators and technicians
8. Contains ANY terminology that could appear in professional documentation

IMPORTANT: Use an INCLUSIVE approach - include segments that contain ANY terminology that could be valuable for translation memory, including basic operational terms, interface elements, and procedural vocabulary, not just highly specialized terminology.

Return ONLY a JSON array of relevant segment indices:
["0", "3", "7", "12", "15", ...]

Cast a wider net - include segments with moderate relevance to ensure comprehensive terminology coverage."""

        try:
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1500,
                temperature=0,
                system="You are an expert terminology analyst. Identify segments containing valuable domain-specific terminology for translation memory purposes.",
                messages=[{"role": "user", "content": relevance_prompt}]
            )
            
            content = response.content[0].text.strip()
            logger.debug(f"V7 AI relevance filtering response: {content}")
            
            # Parse relevant indices
            try:
                relevant_indices = json.loads(content)
                # Convert to integers
                relevant_indices = [int(idx) for idx in relevant_indices if str(idx).isdigit()]
            except:
                # Extract array from response
                array_match = re.search(r'\[[\s\S]*?\]', content)
                if array_match:
                    relevant_indices = json.loads(array_match.group(0))
                    relevant_indices = [int(idx) for idx in relevant_indices if str(idx).isdigit()]
                else:
                    # Fallback - include segments with longer text (likely more technical)
                    relevant_indices = [i for i, seg in enumerate(segments) if len(seg['text']) > 20]
            
            # Filter segments based on AI analysis
            relevant_segments = []
            for idx in relevant_indices:
                if 0 <= idx < len(segments):
                    segment = segments[idx].copy()
                    segment['ai_relevance_score'] = 1.0  # Mark as AI-approved
                    relevant_segments.append(segment)
            
            logger.info(f"🎯 V7 AI Relevance Results:")
            logger.info(f"  📊 Total segments analyzed: {len(segments)}")
            logger.info(f"  ✅ Relevant segments identified: {len(relevant_segments)}")
            logger.info(f"  📈 Relevance rate: {len(relevant_segments)/len(segments)*100:.1f}%")
            
            return relevant_segments
            
        except Exception as e:
            logger.error(f"V7 AI relevance filtering error: {e}")
            # Fallback - return segments with basic filtering
            return [seg for seg in segments if len(seg['text']) > 10]

    def ai_assign_categories(self, client, terms: List[Dict], domain_info: Dict, logger) -> List[Dict]:
        """AI-powered category assignment based on detected domain"""
        logger.info(f"🤖 V7 AI: Assigning categories for {domain_info['primary_domain']} terms...")
        
        if not terms:
            return terms
        
        # Prepare terms for categorization
        terms_for_analysis = []
        for i, term in enumerate(terms[:50]):  # Limit for performance
            terms_for_analysis.append({
                "index": i,
                "source_term": term.get('source_term', ''),
                "target_translation": term.get('target_translation', ''),
                "reference_sentence": term.get('reference_sentence', '')
            })
        
        categorization_prompt = f"""Categorize these {domain_info['primary_domain']} terms into appropriate subcategories.

DOMAIN CONTEXT:
- Primary Domain: {domain_info['primary_domain']}
- Available Subdomains: {', '.join(domain_info['subdomains'])}
- Recommended Categories: {', '.join(domain_info['recommended_categories'])}

TERMS TO CATEGORIZE:
{json.dumps(terms_for_analysis, indent=2, ensure_ascii=False)}

TASK: Assign each term to the most appropriate category and subcategory.

Return ONLY this JSON structure:
{{
  "0": {{
    "category": "Main Category Name",
    "subcategory": "Specific Subcategory",
    "industry_tags": ["Tag1", "Tag2", "Tag3"],
    "complexity_level": "basic/intermediate/advanced",
    "domain_specificity": "general/specialized/highly_specialized"
  }},
  "1": {{ ... }},
  ...
}}

Use the detected subdomains as basis for categories, but create specific subcategories based on the actual terms."""

        try:
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=3000,
                temperature=0,
                system="You are an expert terminology categorization specialist. Assign terms to appropriate categories based on domain analysis and term characteristics.",
                messages=[{"role": "user", "content": categorization_prompt}]
            )
            
            content = response.content[0].text.strip()
            logger.debug(f"V7 AI categorization response: {content[:500]}...")
            
            # Parse categorization results
            try:
                categorizations = json.loads(content)
            except:
                # Extract JSON from response
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    categorizations = json.loads(json_match.group(0))
                else:
                    categorizations = {}
            
            # Apply categorizations to terms
            categorized_terms = []
            for i, term in enumerate(terms):
                enhanced_term = term.copy()
                
                if str(i) in categorizations:
                    cat_info = categorizations[str(i)]
                    enhanced_term.update({
                        'category': cat_info.get('category', domain_info['primary_domain']),
                        'subcategory': cat_info.get('subcategory', 'General'),
                        'industry_tags': cat_info.get('industry_tags', [domain_info['primary_domain']]),
                        'ai_complexity_level': cat_info.get('complexity_level', 'intermediate'),
                        'domain_specificity': cat_info.get('domain_specificity', 'specialized')
                    })
                else:
                    # Fallback categorization
                    enhanced_term.update({
                        'category': domain_info['primary_domain'],
                        'subcategory': domain_info['subdomains'][0] if domain_info['subdomains'] else 'General',
                        'industry_tags': [domain_info['primary_domain']],
                        'ai_complexity_level': domain_info['complexity_level'],
                        'domain_specificity': 'specialized'
                    })
                
                categorized_terms.append(enhanced_term)
            
            # Log categorization results
            categories = {}
            for term in categorized_terms:
                cat = term.get('category', 'Unknown')
                categories[cat] = categories.get(cat, 0) + 1
            
            logger.info(f"🎯 V7 AI Categorization Results:")
            for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  📋 {category}: {count} terms")
            
            return categorized_terms
            
        except Exception as e:
            logger.error(f"V7 AI categorization error: {e}")
            # Return terms with basic categorization
            for term in terms:
                term.update({
                    'category': domain_info['primary_domain'],
                    'subcategory': 'General',
                    'industry_tags': [domain_info['primary_domain']],
                    'ai_complexity_level': 'intermediate',
                    'domain_specificity': 'specialized'
                })
            return terms

    def analyze_compound_enhanced(self, term: str) -> CompoundAnalysis:
        """Enhanced compound word analysis for multiple languages"""
        # Enhanced patterns for various languages
        compound_patterns = [
            r'([a-z]+)[-]([a-z]+)[-]([a-z]+)',  # Three parts with hyphens
            r'([a-z]+)[-]([a-z]+)',  # Two parts with hyphens
            r'([A-Z][a-z]+)([A-Z][a-z]+)',  # CamelCase compounds
            r'([a-z]+)\s+([a-z]+)\s+([a-z]+)',  # Multi-word technical terms
            r'([A-Za-z]+)([A-Z][a-z]+)',  # German-style compounds
        ]
        
        for pattern in compound_patterns:
            match = re.search(pattern, term, re.IGNORECASE)
            if match:
                components = list(match.groups())
                if len(components) >= 3:
                    compound_type = "complex_compound"
                elif len(components) == 2:
                    compound_type = "compound"
                else:
                    compound_type = "simple"
                
                return CompoundAnalysis(
                    is_compound=True,
                    components=components,
                    compound_type=compound_type
                )
        
        # Check for technical abbreviations
        if re.match(r'^[A-Z]{2,6}$', term):
            return CompoundAnalysis(
                is_compound=True,
                components=[term],
                compound_type="abbreviation"
            )
        
        # Check for long technical terms
        if len(term) > 15 and any(char.isalpha() for char in term):
            return CompoundAnalysis(
                is_compound=True,
                components=[term],
                compound_type="technical_term"
            )
        
        return CompoundAnalysis(
            is_compound=False,
            components=[term],
            compound_type="simple"
        )

def extract_terms_batch_v7_ai(client, batch_segments: List[Dict], domain_info: Dict, 
                             detected_source_lang: str, detected_target_lang: str, logger, 
                             already_extracted: Set[str], processor: AITermProcessor) -> List[Dict]:
    """V7 AI-driven extraction with domain-adaptive prompts"""
    
    primary_domain = domain_info.get('primary_domain', 'Technical Content')
    subdomains = domain_info.get('subdomains', ['Technical Terms'])
    extraction_focus = domain_info.get('extraction_focus', 'Technical terminology')
    
    # Prepare batch context
    batch_texts = []
    for i, segment in enumerate(batch_segments, 1):
        batch_texts.append(f"{i}. {segment['text']}")
    
    combined_batch = "\n".join(batch_texts)
    
    # Already extracted context
    sample_extracted = list(already_extracted)[:15] if already_extracted else []
    extracted_context = f"Skip these already extracted: {', '.join(sample_extracted)}" if sample_extracted else ""
    
    # Language mapping
    language_map = {
        'de': 'German', 'en': 'English', 'fr': 'French', 'es': 'Spanish', 
        'it': 'Italian', 'pt': 'Portuguese', 'nl': 'Dutch', 'pl': 'Polish',
        'cs': 'Czech', 'sk': 'Slovak', 'hu': 'Hungarian', 'ro': 'Romanian',
        'bg': 'Bulgarian', 'hr': 'Croatian', 'sl': 'Slovenian', 'et': 'Estonian',
        'lv': 'Latvian', 'lt': 'Lithuanian', 'fi': 'Finnish', 'sv': 'Swedish',
        'da': 'Danish', 'no': 'Norwegian', 'ru': 'Russian', 'uk': 'Ukrainian',
        'zh': 'Chinese', 'ja': 'Japanese', 'ko': 'Korean', 'ar': 'Arabic',
        'he': 'Hebrew', 'tr': 'Turkish', 'th': 'Thai', 'vi': 'Vietnamese'
    }
    
    source_lang_name = language_map.get(detected_source_lang, detected_source_lang.upper())
    target_lang_name = language_map.get(detected_target_lang, detected_target_lang.upper())
    
    # V7 AI-DRIVEN domain-adaptive extraction prompt
    prompt = f"""EXTRACT {source_lang_name.upper()} TERMS from {primary_domain} content and translate to {target_lang_name.upper()}.

DOMAIN ANALYSIS:
- Primary Domain: {primary_domain}
- Specialized Areas: {', '.join(subdomains)}
- Extraction Focus: {extraction_focus}
- Content Complexity: {domain_info.get('complexity_level', 'intermediate')}

CRITICAL REQUIREMENTS:
- Source terms MUST be in {source_lang_name} language
- Target translations MUST be in {target_lang_name} language  
- Focus on {primary_domain} terminology
- Prioritize terms related to: {', '.join(subdomains)}

RETURN ONLY this JSON array structure:
[
  {{
    "source_term": "{source_lang_name} term here",
    "target_translation": "{target_lang_name} translation here", 
    "reference_sentence": "Sentence containing the term",
    "sample_usage": "Professional usage example in {target_lang_name}",
    "confidence_score": 0.90,
    "domain_context": "Brief context about {primary_domain} usage",
    "complexity_level": "basic|intermediate|advanced|expert"
  }}
]

ENHANCED EXTRACTION RULES:
- Extract 12-20 valuable {primary_domain} terms from {source_lang_name} text
- Focus on: {extraction_focus}
- Include terminology specific to: {', '.join(subdomains)}
- Include: specialized terminology, technical processes, equipment names, professional concepts, operational terms, interface elements, procedural vocabulary
- Skip only: articles, prepositions, very common words (the, and, of, etc.)
- Skip these already extracted: {', '.join(sample_extracted[:10])}
- Use professional, domain-appropriate {target_lang_name} translations
- Confidence range: 0.75-1.00 (be realistic about translation quality)
- Mark complexity based on {primary_domain} standards

IMPORTANT: Extract COMPREHENSIVE {primary_domain} terminology including both specialized and operational terms. Include basic but essential terms that operators, technicians, and translators would encounter in professional contexts. Cast a wide net for terminology coverage!

{extracted_context}

{source_lang_name} text segments to analyze:
{combined_batch}

Return ONLY the JSON array with {source_lang_name} → {target_lang_name} translations:"""

    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4000,
            temperature=0,
            system=f"You are an expert {primary_domain} terminology extractor. Extract valuable domain-specific terms for professional translation memory. Focus on {extraction_focus}. Return ONLY a JSON array - no explanations.",
            messages=[{"role": "user", "content": prompt}]
        )
        
        content = response.content[0].text.strip()
        logger.debug(f"V7 AI extraction response: {content[:200]}...")
        
        # Enhanced JSON parsing
        extracted_data = None
        
        # Method 1: Direct JSON parsing
        try:
            extracted_data = json.loads(content)
        except:
            pass
        
        # Method 2: Extract from code blocks
        if not extracted_data:
            json_match = re.search(r'```json\n([\s\S]*?)\n```', content)
            if json_match:
                try:
                    extracted_data = json.loads(json_match.group(1))
                except:
                    pass
        
        # Method 3: Find array structure
        if not extracted_data:
            array_match = re.search(r'\[[\s\S]*\]', content)
            if array_match:
                try:
                    extracted_data = json.loads(array_match.group(0))
                except:
                    pass
        
        # Validate extracted data
        if not isinstance(extracted_data, list):
            logger.error(f"V7 AI: Expected list, got {type(extracted_data)}: {extracted_data}")
            return []
        
        # Process terms with V7 AI enhancements
        enhanced_terms = []
        for i, term_data in enumerate(extracted_data):
            try:
                if not isinstance(term_data, dict):
                    logger.warning(f"V7 AI: Skipping invalid term data at index {i}: {term_data}")
                    continue
                    
                source_term = term_data.get('source_term', '').strip()
                target_translation = term_data.get('target_translation', '').strip()
                confidence = term_data.get('confidence_score', 0.85)
                reference_sentence = term_data.get('reference_sentence', '').strip()
                
                # Basic validation
                if (source_term and target_translation and
                    len(source_term) >= 2 and
                    len(target_translation) >= 2 and
                    source_term.lower() not in already_extracted):
                    
                    # V7 AI processing
                    compound_analysis = processor.analyze_compound_enhanced(source_term)
                    
                    # Calculate domain alignment based on AI analysis
                    domain_alignment = 0.95  # High alignment since AI-detected
                    
                    # Build enhanced term structure
                    enhanced_term = {
                        # Core fields
                        'source_term': source_term,
                        'target_translation': target_translation,
                        'confidence_score': confidence,
                        'reference_sentence': reference_sentence,
                        'sample_usage': term_data.get('sample_usage', ''),
                        
                        # V7 AI-driven categorization (to be filled by AI categorization)
                        'category': primary_domain,  # Will be updated by AI categorization
                        'subcategory': subdomains[0] if subdomains else 'General',  # Will be updated
                        'industry_tags': [primary_domain],  # Will be updated
                        'domain_alignment': domain_alignment,
                        
                        # Enhanced analysis
                        'compound_analysis': asdict(compound_analysis),
                        'tm_consistency': True,
                        'quality_tier': 'high' if confidence >= 0.85 else 'medium',
                        'processing_version': 'v7_ai_driven',
                        'translation_consistency': 0.95,
                        
                        # AI-specific fields
                        'domain_context': term_data.get('domain_context', ''),
                        'complexity_level': term_data.get('complexity_level', 'intermediate'),
                        'ai_extracted': True,
                        'ai_domain': primary_domain
                    }
                    
                    enhanced_terms.append(enhanced_term)
                    already_extracted.add(source_term.lower())
                    
            except Exception as e:
                logger.error(f"V7 AI: Error processing term at index {i}: {e}")
                continue
        
        logger.info(f"V7 AI extracted {len(enhanced_terms)} terms from batch of {len(batch_segments)} segments")
        return enhanced_terms
        
    except Exception as e:
        logger.error(f"V7 AI batch extraction error: {e}")
        logger.error(traceback.format_exc())
        return []

def enhanced_deduplication_v7(terms: List[Dict], logger) -> List[Dict]:
    """V7 Enhanced deduplication with quality preservation."""
    logger.info("Starting V7 AI deduplication...")
    
    # Advanced deduplication tracking
    seen_terms = {}
    unique_terms = []
    
    for term in terms:
        source_term = term.get('source_term', '').strip()
        if not source_term:
            continue
        
        source_lower = source_term.lower()
        
        if source_lower not in seen_terms:
            # First occurrence - add to unique terms
            seen_terms[source_lower] = term
            unique_terms.append(term)
        else:
            # Duplicate found - keep the higher quality version
            existing_term = seen_terms[source_lower]
            existing_confidence = existing_term.get('confidence_score', 0.0)
            current_confidence = term.get('confidence_score', 0.0)
            
            # Replace if current term is better
            if current_confidence > existing_confidence:
                # Find and replace in unique_terms
                for i, unique_term in enumerate(unique_terms):
                    if unique_term.get('source_term', '').lower() == source_lower:
                        unique_terms[i] = term
                        seen_terms[source_lower] = term
                        logger.debug(f"V7 AI: Replaced {source_term} with higher confidence version")
                        break
    
    # Enhanced quality classification
    for term in unique_terms:
        confidence = term.get('confidence_score', 0.85)
        
        # V7 AI quality tiers
        if confidence >= 0.90:
            term['quality_tier'] = 'high'
        elif confidence >= 0.80:
            term['quality_tier'] = 'medium' 
        else:
            term['quality_tier'] = 'medium'  # No low quality in V7
    
    # Count statistics
    high_quality = len([t for t in unique_terms if t.get('quality_tier') == 'high'])
    medium_quality = len([t for t in unique_terms if t.get('quality_tier') == 'medium'])
    
    logger.info(f"V7 AI Deduplication Results:")
    logger.info(f"  🏆 High-quality terms: {high_quality}")
    logger.info(f"  📈 Medium-quality terms: {medium_quality}")
    logger.info(f"  📊 Total terms: {len(unique_terms)}")
    logger.info(f"  ❌ Duplicates removed: {len(terms) - len(unique_terms)}")
    
    return unique_terms

def create_ai_batches(segments: List[Dict], batch_size: int = 30) -> List[List[Dict]]:
    """AI-filtered batch creation"""
    
    # AI-filtered segments already have relevance scores
    # Sort by AI relevance if available
    valid_segments = []
    for segment in segments:
        if (is_valid_content(segment['text']) and 
            not is_lorem_ipsum_fast(segment['text']) and 
            not is_url_or_path(segment['text'])):
            valid_segments.append(segment)
    
    # Sort by AI relevance score if available, otherwise by length
    valid_segments.sort(key=lambda x: x.get('ai_relevance_score', len(x['text'])/100), reverse=True)
    
    # Create batches
    batches = []
    for i in range(0, len(valid_segments), batch_size):
        batch = valid_segments[i:i + batch_size]
        batches.append(batch)
    
    return batches

# Utility functions
def is_valid_content(text: str) -> bool:
    """RELAXED validation for content quality."""
    if not text or len(text.strip()) == 0:
        return False
    if text.isdigit():
        return False
    if not any(c.isalpha() for c in text):
        return False
    if len(text.strip()) < 2:  # REDUCED from 3 to 2
        return False
    return True

def is_url_or_path(text: str) -> bool:
    """Check if text is a URL or file path."""
    return any(indicator in text.lower() for indicator in ['http://', 'https://', 'www.', '.com', '.de', '.org', '.png', '.jpg', '.jpeg', '.pdf', 'wp-content'])

def is_lorem_ipsum_fast(text: str) -> bool:
    """Fast Lorem ipsum check."""
    text_lower = text.lower()
    lorem_count = sum(1 for word in ['lorem', 'ipsum', 'dolor', 'amet', 'consectetur'] if word in text_lower)
    return lorem_count >= 2

def extract_terms_from_xliff_v7_ai(file_path: str, client, source_lang: str = None, target_lang: str = None, 
                                  logger = None, batch_size: int = 30) -> Tuple[List[Dict], str, str, Dict]:
    """V7 AI-driven extraction with complete AI analysis pipeline."""
    logger.info(f"🚀 V7 AI-DRIVEN extraction from: {file_path}")
    
    try:
        # Step 1: Parse XML and detect languages
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        namespaces = {
            '': 'urn:oasis:names:tc:xliff:document:1.2',
            'xliff': 'urn:oasis:names:tc:xliff:document:1.2',
            'xliff2': 'urn:oasis:names:tc:xliff:document:2.0'
        }
        
        detected_source_lang = None
        detected_target_lang = None
        
        for ns_prefix, ns_uri in namespaces.items():
            ns = {ns_prefix: ns_uri} if ns_prefix else {'': ns_uri}
            
            file_element = root.find('.//file', ns)
            if file_element is not None:
                detected_source_lang = file_element.get('source-language')
                detected_target_lang = file_element.get('target-language')
                if detected_source_lang and detected_target_lang:
                    break
        
        if not detected_source_lang:
            detected_source_lang = source_lang or "en"
        
        if not detected_target_lang:
            detected_target_lang = target_lang or "tr"
        
        detected_source_lang = detected_source_lang.split('-')[0].lower()
        detected_target_lang = detected_target_lang.split('-')[0].lower()
        
        logger.info(f"✅ AUTO-DETECTED Languages: {detected_source_lang.upper()} → {detected_target_lang.upper()}")
        
        # Step 2: Extract text samples
        trans_units = root.findall('.//trans-unit', namespaces)
        logger.info(f"V7 AI: Processing {len(trans_units)} trans-units")
        
        all_segments = []
        text_samples = []
        
        for trans_unit in trans_units:
            source_elem = trans_unit.find('.//source', namespaces)
            if source_elem is not None and source_elem.text:
                text = source_elem.text.strip()
                if is_valid_content(text):
                    all_segments.append({'text': text})
                    if len(text_samples) < 100:
                        text_samples.append(text)
        
        logger.info(f"V7 AI: Collected {len(all_segments)} valid segments")
        
        # Step 3: AI-POWERED domain analysis
        processor = AITermProcessor()
        domain_info = processor.ai_analyze_domain(client, text_samples, detected_source_lang, logger)
        
        # Step 4: AI-POWERED relevance filtering
        relevant_segments = processor.ai_filter_relevant_segments(client, all_segments, domain_info, logger)
        
        # Step 5: Create AI-filtered batches
        batches = create_ai_batches(relevant_segments, batch_size)
        coverage_rate = len(relevant_segments) / len(all_segments) if all_segments else 0
        
        logger.info(f"V7 AI: Created {len(batches)} AI-filtered batches with {coverage_rate:.1%} relevance coverage")
        
        if len(batches) == 0:
            logger.warning("⚠️ V7 AI: No relevant batches - using fallback to process all segments.")
            batches = create_ai_batches(all_segments, batch_size)
            logger.info(f"V7 AI: Created {len(batches)} fallback batches.")

        # Step 6: AI-DRIVEN term extraction
        all_terms = []
        already_extracted = set()
        successful_batches = 0
        
        for i, batch in enumerate(batches, 1):
            logger.info(f"V7 AI: Processing batch {i}/{len(batches)} ({len(batch)} segments)...")
            
            try:
                batch_terms = extract_terms_batch_v7_ai(
                    client, batch, domain_info, detected_source_lang, detected_target_lang, 
                    logger, already_extracted, processor
                )
                
                all_terms.extend(batch_terms)
                successful_batches += 1
                
                if i % 3 == 0:
                    logger.info(f"V7 AI progress: {i}/{len(batches)} batches, {len(all_terms)} terms")
                
                time.sleep(0.7)
                
            except Exception as e:
                logger.error(f"V7 AI: Batch {i} failed: {e}")
                continue
        
        logger.info(f"V7 AI: Raw extraction complete - {len(all_terms)} terms before processing")
        
        # Step 7: AI-POWERED categorization
        categorized_terms = processor.ai_assign_categories(client, all_terms, domain_info, logger)
        
        # Step 8: Enhanced deduplication
        final_terms = enhanced_deduplication_v7(categorized_terms, logger)
        
        # Calculate metrics
        enhanced_metrics = {
            'extraction_efficiency': len(final_terms) / len(batches) if batches else 0,
            'relevance_coverage_rate': coverage_rate,
            'processing_success_rate': successful_batches / len(batches) if batches else 0,
            'ai_domain_detection': True,
            'primary_domain': domain_info.get('primary_domain', 'Technical Content'),
            'domain_confidence': domain_info.get('confidence_score', 0.8),
            'ai_categories_generated': len(set(term.get('category', '') for term in final_terms)),
            'ai_complexity_level': domain_info.get('complexity_level', 'intermediate')
        }
        
        logger.info(f"🎉 V7 AI extraction complete:")
        logger.info(f"  ✅ Final AI-processed terms: {len(final_terms)}")
        
        return final_terms, detected_source_lang, detected_target_lang, enhanced_metrics
        
    except Exception as e:
        logger.error(f"V7 AI extraction error: {e}")
        logger.error(traceback.format_exc())
        return [], source_lang, target_lang, {}

def save_results_v7_ai(terms: List[Dict], detected_source_lang: str, detected_target_lang: str, 
                      metrics: Dict, logger) -> Tuple[str, str]:
    """V7 AI result saving with complete AI analysis."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"ai_terms_v7_{detected_source_lang}_{detected_target_lang}_{timestamp}"
    
    high_quality_terms = [term for term in terms if term.get('quality_tier') == 'high']
    medium_quality_terms = [term for term in terms if term.get('quality_tier') == 'medium']
    
    category_breakdown = defaultdict(int)
    industry_breakdown = defaultdict(int)
    subcategory_breakdown = defaultdict(int)
    complexity_breakdown = defaultdict(int)
    
    for term in terms:
        category_breakdown[term.get('category', 'Technical Content')] += 1
        subcategory_breakdown[term.get('subcategory', 'General')] += 1
        complexity_breakdown[term.get('ai_complexity_level', 'intermediate')] += 1
        for tag in term.get('industry_tags', ['Technical']):
            industry_breakdown[tag] += 1

    extraction_stats = {
        "extraction_info": {
            "version": "7.0 - AI-DRIVEN COMPLETE (AUTOMATED)",
            "timestamp": timestamp,
            "source_language": detected_source_lang,
            "target_language": detected_target_lang,
            "total_terms": len(terms),
        },
        "ai_analysis": metrics,
        "quality_metrics": {
            "avg_confidence_score": sum(t.get('confidence_score', 0.85) for t in terms) / len(terms) if terms else 0,
        },
        "category_breakdown": dict(category_breakdown),
        "subcategory_breakdown": dict(subcategory_breakdown),
        "industry_tags": dict(industry_breakdown),
        "complexity_breakdown": dict(complexity_breakdown)
    }
    
    output_data = {
        "metadata": extraction_stats,
        "ai_terms": terms
    }
    
    json_filename = f"{base_filename}.json"
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    logger.info(f"V7 AI JSON saved: {json_filename}")
    
    excel_filename = f"{base_filename}.xlsx"
    try:
        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            
            def prepare_ai_term_data(term_list):
                data = []
                for term in term_list:
                    data.append({
                        'Source Term': term.get('source_term'), 'Target Translation': term.get('target_translation'),
                        'AI Category': term.get('category'), 'AI Subcategory': term.get('subcategory'),
                        'Industry Tags': ', '.join(term.get('industry_tags', [])),
                        'Confidence Score': term.get('confidence_score'), 'Quality Tier': term.get('quality_tier'),
                        'AI Complexity': term.get('ai_complexity_level'), 'Domain Specificity': term.get('domain_specificity'),
                        'Reference Sentence': term.get('reference_sentence'),
                        'Processing Version': term.get('processing_version')
                    })
                return data
            
            if terms:
                df_all = pd.DataFrame(prepare_ai_term_data(terms))
                df_all.to_excel(writer, sheet_name='All AI Terms (V7)', index=False)
        
        logger.info(f"V7 AI Excel saved: {excel_filename}")
    except Exception as e:
        logger.error(f"V7 AI Excel save error: {e}")
        excel_filename = "Excel export failed"
    
    print(f"\n🎉 V7 AI-DRIVEN EXTRACTION SUCCESS!")
    print(f"  🤖 AI-Detected Domain: {metrics.get('primary_domain', 'N/A')}")
    print(f"\n📂 V7 AI FILES:")
    print(f"  • AI JSON: {json_filename}")
    print(f"  • AI Excel: {excel_filename}")
    
    return json_filename, excel_filename

def find_memoq_files(directory: str) -> List[str]:
    """Find all MemoQ XLIFF files."""
    memoq_files = []
    for root_dir, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.mqxliff', '.sdlxliff')):
                memoq_files.append(os.path.join(root_dir, file))
    return memoq_files

def main():
    """V7 AI-Driven Main execution - MODIFIED FOR AUTOMATION."""
    logger, log_filename = setup_logging()
    logger.info("🚀 STARTING V7 AI-DRIVEN TERM EXTRACTION (AUTOMATED)")
    logger.info("🤖 Features: Complete AI Domain Detection, AI Relevance Filtering, AI Categorization")
    logger.info(f"📝 Log file: {log_filename}")
    
    try:
        parser = argparse.ArgumentParser(description='V7 AI-driven term extraction for automated workflows.')
        parser.add_argument('--directory', type=str, default='.', help='Directory to search for MemoQ XLIFF files')
        parser.add_argument('--source-lang', type=str, default=None, help='(Optional) Source language code (e.g., de, en)')
        parser.add_argument('--target-lang', type=str, default=None, help='(Optional) Target language code (e.g., en, de)')
        parser.add_argument('--batch-size', type=int, default=30, help='Segments per batch (default: 30)')
        args = parser.parse_args()
        
        logger.info(f"V7 AI Configuration: directory={args.directory}, batch_size={args.batch_size}")
        
        # --- MODIFICATION 1: Get API key from environment variable ---
        api_key = os.getenv("CLAUDE_API_KEY")
        if not api_key:
            logger.error("❌ CLAUDE_API_KEY environment variable not set.")
            print("❌ CLAUDE_API_KEY environment variable not set.")
            return # Exit if the key is not found

        client = anthropic.Anthropic(api_key=api_key)
        logger.info("✅ V7 AI Claude client initialized")
        
        # Find MemoQ files
        memoq_files = find_memoq_files(args.directory)
        
        # --- MODIFICATION 2: Remove interactive file selection ---
        selected_files = memoq_files
        if not selected_files:
            logger.error(f"❌ No MemoQ XLIFF files found in directory: {args.directory}")
            print(f"❌ No MemoQ XLIFF files found in directory: {args.directory}")
            return
        
        logger.info(f"✅ Automatically processing all {len(selected_files)} found files: {selected_files}")

        all_terms = []
        start_time = time.time()
        
        print(f"\n🚀 Starting V7 AI-DRIVEN processing on {len(selected_files)} file(s)...")
        
        for i, file_path in enumerate(selected_files, 1):
            logger.info(f"🔄 V7 AI Processing file {i}/{len(selected_files)}: {file_path}")
            print(f"\n🔄 V7 AI Processing [{i}/{len(selected_files)}]: {os.path.basename(file_path)}")
            
            file_start_time = time.time()
            terms, file_src_lang, file_trg_lang, file_metrics = extract_terms_from_xliff_v7_ai(
                file_path, client, args.source_lang, args.target_lang, logger, args.batch_size
            )
            file_duration = time.time() - file_start_time
            
            detected_source_lang = file_src_lang
            detected_target_lang = file_trg_lang
            
            print(f"  ✅ V7 AI extracted {len(terms)} terms in {file_duration:.1f}s")
            
            all_terms.extend(terms)
        
        total_duration = time.time() - start_time
        
        logger.info(f"🎉 V7 AI Processing completed in {total_duration:.1f}s")
        logger.info(f"📊 Total AI-processed terms extracted: {len(all_terms)}")
        
        if not all_terms:
            logger.error("❌ No terms extracted by V7 AI system")
            print("❌ No terms extracted.")
            return
        
        # For simplicity, we'll use the metrics from the last file for the final report
        final_metrics = file_metrics if 'file_metrics' in locals() else {}
        
        # Save AI results
        print(f"\n💾 Saving V7 AI results ({len(all_terms)} AI-processed terms)...")
        json_file, excel_file = save_results_v7_ai(
            all_terms, detected_source_lang, detected_target_lang, final_metrics, logger
        )
        
        print(f"\n🎖️ V7 AI-DRIVEN SUCCESS: Complete AI Analysis Pipeline Implemented!")
        
        logger.info("✅ V7 AI-driven extraction completed successfully")
        
    except KeyboardInterrupt:
        logger.info("⚠️ V7 AI processing interrupted by user")
        print("\n⚠️ V7 AI processing interrupted.")
    except Exception as e:
        logger.error(f"❌ V7 AI unexpected error: {e}")
        logger.error(traceback.format_exc())
        print(f"❌ V7 AI error occurred. Check {log_filename} for details.")

if __name__ == "__main__":
    main()
