#!/usr/bin/env python3
"""
EEG Named Entity Recognition (NER) Demonstration
Shows comprehensive extraction of EEG terminology from research text
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eeg_rag.nlp.ner_eeg import EEGNER, EntityType


def print_section(title):
    """Print section header"""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def demo_basic_extraction():
    """Demonstrate basic entity extraction"""
    print_section("BASIC ENTITY EXTRACTION")
    
    ner = EEGNER()
    
    text = """
    We recorded EEG from electrodes Fp1, Fz, Cz, and O1. Analysis revealed 
    increased theta and alpha power in the frontal cortex of patients with 
    epilepsy compared to healthy controls.
    """
    
    print("\nInput Text:")
    print(text.strip())
    
    result = ner.extract_entities(text)
    
    print(f"\n✓ Found {len(result.entities)} entities in {result.processing_time:.4f}s")
    print(f"✓ Entity types: {len(result.entity_counts)}")
    
    print("\nExtracted Entities:")
    for entity in result.entities:
        print(f"  • {entity.text:20s} [{entity.entity_type.value}] (conf: {entity.confidence:.2f})")


def demo_entity_types():
    """Demonstrate extraction of different entity types"""
    print_section("ENTITY TYPE BREAKDOWN")
    
    ner = EEGNER()
    
    examples = {
        "Frequency Bands": "Delta, theta, alpha, beta, and gamma oscillations were analyzed.",
        "Brain Regions": "Activity in the frontal cortex, hippocampus, and amygdala was measured.",
        "Electrodes": "Electrodes Fp1, Fp2, Fz, Cz, Pz, O1, and O2 were used.",
        "Clinical Conditions": "Patients with epilepsy, depression, and Alzheimer's disease participated.",
        "Biomarkers": "We measured P300 amplitude, alpha asymmetry, and theta-beta ratio.",
        "Processing Methods": "ICA, bandpass filter, and FFT were applied for artifact removal.",
    }
    
    for category, text in examples.items():
        result = ner.extract_entities(text)
        print(f"\n{category}:")
        print(f"  Text: \"{text}\"")
        print(f"  Entities found: {[e.text for e in result.entities]}")


def demo_research_abstract():
    """Demonstrate NER on a realistic research abstract"""
    print_section("RESEARCH ABSTRACT ANALYSIS")
    
    ner = EEGNER()
    
    abstract = """
    Background: This study investigates quantitative EEG biomarkers in patients with 
    epilepsy. We hypothesized that theta-beta ratio and alpha asymmetry would differ 
    between patients and controls.
    
    Methods: EEG was recorded from 64 electrodes (10-20 system) including Fp1, Fp2, 
    F3, F4, Fz, C3, C4, Cz, P3, P4, Pz, O1, and O2 during resting state with eyes 
    closed. Independent component analysis (ICA) removed artifacts. Power spectral 
    density (PSD) was computed using fast Fourier transform (FFT) for delta (0.5-4 Hz), 
    theta (4-8 Hz), alpha (8-13 Hz), beta (13-30 Hz), and gamma (30-100 Hz) bands.
    
    Results: Patients with epilepsy showed increased theta power in frontal cortex 
    (F3, F4, Fz) compared to controls (p < 0.001). Alpha asymmetry in parietal regions 
    correlated with seizure frequency (r = 0.68, p < 0.01). P300 amplitude during an 
    oddball task was reduced in patients (t = -3.45, p < 0.001).
    
    Conclusion: Quantitative EEG measures including theta power, alpha asymmetry, and 
    P300 amplitude provide sensitive biomarkers for epilepsy diagnosis and monitoring.
    """
    
    print("\nAnalyzing research abstract...")
    result = ner.extract_entities(abstract)
    
    print(f"\n✓ Total entities found: {len(result.entities)}")
    print(f"✓ Processing time: {result.processing_time:.4f}s")
    print(f"✓ Unique entities: {len(set(e.text.lower() for e in result.entities))}")
    
    # Group by entity type
    print("\nEntities by Type:")
    for entity_type in EntityType:
        entities = [e for e in result.entities if e.entity_type == entity_type]
        if entities:
            unique_texts = list(set(e.text for e in entities))
            print(f"  {entity_type.value:20s}: {len(entities):2d} ({', '.join(unique_texts[:5])}{'...' if len(unique_texts) > 5 else ''})")
    
    # Get summary
    summary = ner.get_entity_summary(result)
    print(f"\nSummary Statistics:")
    print(f"  Most common type: {summary['most_common_type']}")
    print(f"  Average confidence: {summary['avg_confidence']:.2f}")


def demo_frequency_band_metadata():
    """Demonstrate frequency band metadata extraction"""
    print_section("FREQUENCY BAND METADATA")
    
    ner = EEGNER()
    
    text = "We analyzed delta, theta, alpha, beta, and gamma oscillations."
    result = ner.extract_entities(text)
    
    print("\nFrequency Band Details:")
    for entity in result.entities:
        if entity.entity_type == EntityType.FREQUENCY_BAND:
            print(f"\n  {entity.text}:")
            if 'frequency_range' in entity.metadata:
                low, high = entity.metadata['frequency_range']
                print(f"    Range: {low}-{high} Hz")
            if 'description' in entity.metadata:
                print(f"    Description: {entity.metadata['description']}")


def demo_context_extraction():
    """Demonstrate context extraction around entities"""
    print_section("CONTEXT EXTRACTION")
    
    ner = EEGNER()
    
    text = """
    The study examined alpha oscillations in the posterior brain regions during 
    resting state with eyes closed. We found that alpha power was significantly 
    elevated in the occipital cortex.
    """
    
    result = ner.extract_entities(text, context_window=40)
    
    print("\nEntities with Context:")
    for entity in result.entities[:5]:  # Show first 5
        print(f"\n  Entity: {entity.text} [{entity.entity_type.value}]")
        print(f"  Context: ...{entity.context}...")


def demo_batch_processing():
    """Demonstrate batch processing of multiple texts"""
    print_section("BATCH PROCESSING")
    
    ner = EEGNER()
    
    texts = [
        "Alpha waves at 10 Hz were recorded from electrode Cz.",
        "Patients with epilepsy showed abnormal theta activity.",
        "The P300 amplitude was measured during an oddball task.",
        "Independent component analysis removed eye movement artifacts.",
        "Beta desynchronization occurred in the motor cortex during hand movement."
    ]
    
    print(f"\nProcessing {len(texts)} texts...")
    results = ner.extract_batch(texts)
    
    print("\nResults:")
    for i, (text, result) in enumerate(zip(texts, results), 1):
        entities_str = ', '.join([e.text for e in result.entities])
        print(f"  Text {i}: {len(result.entities)} entities ({entities_str})")
    
    # Overall statistics
    stats = ner.get_statistics()
    print(f"\nOverall Statistics:")
    print(f"  Documents processed: {stats['documents_processed']}")
    print(f"  Total entities: {stats['total_entities_found']}")
    print(f"  Avg entities/doc: {stats['avg_entities_per_doc']:.1f}")
    print(f"  Terminology size: {stats['terminology_size']} terms")


def demo_confidence_filtering():
    """Demonstrate confidence-based filtering"""
    print_section("CONFIDENCE FILTERING")
    
    ner = EEGNER()
    
    text = """
    EEG analysis of alpha, beta, and gamma oscillations in the frontal and 
    parietal cortex during a cognitive task.
    """
    
    print("\nSame text with different confidence thresholds:")
    
    for min_conf in [0.0, 0.85, 0.9, 0.95]:
        result = ner.extract_entities(text, min_confidence=min_conf)
        print(f"\n  Min confidence {min_conf:.2f}: {len(result.entities)} entities")
        if result.entities:
            print(f"    Entities: {[e.text for e in result.entities]}")
            print(f"    Confidences: {[f'{e.confidence:.2f}' for e in result.entities]}")


def demo_real_world_methods():
    """Demonstrate NER on a realistic methods section"""
    print_section("METHODS SECTION ANALYSIS")
    
    ner = EEGNER()
    
    methods = """
    EEG Recording and Preprocessing: Continuous EEG was recorded using a 64-channel 
    BioSemi ActiveTwo system with electrodes positioned according to the extended 
    10-20 international system. Electrode impedances were kept below 5 kΩ. Signals 
    were sampled at 512 Hz and referenced online to the CMS/DRL ground. Offline, 
    data were re-referenced to the average reference and bandpass filtered between 
    0.5 and 100 Hz using a zero-phase Butterworth filter. A notch filter at 60 Hz 
    removed power line noise. Independent component analysis (ICA) using the extended 
    Infomax algorithm identified and removed ocular, muscular, and cardiac artifacts. 
    Data were then segmented into 2-second epochs and visually inspected for remaining 
    artifacts.
    
    Spectral Analysis: Power spectral density (PSD) was computed for each electrode 
    using Welch's method with a Hamming window (50% overlap, 1-second segments). 
    Absolute power was calculated for delta (0.5-4 Hz), theta (4-8 Hz), alpha 
    (8-13 Hz), beta (13-30 Hz), and gamma (30-100 Hz) frequency bands. Relative 
    power was computed as the ratio of band power to total power (0.5-100 Hz).
    
    Statistical Analysis: Between-group differences in EEG power were assessed using 
    independent samples t-tests with FDR correction for multiple comparisons. Effect 
    sizes were calculated using Cohen's d. Correlations between EEG measures and 
    clinical variables were evaluated using Pearson's r.
    """
    
    print("\nAnalyzing methods section...")
    result = ner.extract_entities(methods)
    
    print(f"\n✓ Found {len(result.entities)} entities")
    
    # Categorize findings
    categories = {
        'Hardware': EntityType.HARDWARE,
        'Electrodes': EntityType.ELECTRODE,
        'Frequency Bands': EntityType.FREQUENCY_BAND,
        'Processing Methods': EntityType.PROCESSING_METHOD,
        'Measurement Units': EntityType.MEASUREMENT_UNIT,
        'Signal Features': EntityType.SIGNAL_FEATURE,
    }
    
    print("\nKey Findings by Category:")
    for category, entity_type in categories.items():
        entities = [e for e in result.entities if e.entity_type == entity_type]
        if entities:
            unique = list(set(e.text for e in entities))
            print(f"\n  {category} ({len(unique)}):")
            for item in unique[:8]:  # Show first 8
                print(f"    • {item}")
            if len(unique) > 8:
                print(f"    ... and {len(unique) - 8} more")


def demo_export_json():
    """Demonstrate JSON export functionality"""
    print_section("JSON EXPORT")
    
    ner = EEGNER()
    
    text = """
    EEG biomarkers including P300 amplitude, alpha asymmetry, and theta-beta 
    ratio were analyzed in patients with epilepsy.
    """
    
    result = ner.extract_entities(text)
    
    # Export to JSON
    output_path = "data/ner_results_demo.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    ner.export_entities_to_json(result, output_path)
    
    print(f"\n✓ Exported entities to: {output_path}")
    print(f"\nSample Entity (JSON format):")
    if result.entities:
        import json
        print(json.dumps(result.entities[0].to_dict(), indent=2))


def main():
    """Run all demonstrations"""
    print("\n" + "=" * 80)
    print(" EEG NAMED ENTITY RECOGNITION (NER) - COMPREHENSIVE DEMONSTRATION")
    print("=" * 80)
    
    print("\nThis system extracts and categorizes EEG terminology from research papers.")
    print("It recognizes 12 entity types across 400+ EEG-specific terms.")
    
    # Run all demos
    demo_basic_extraction()
    demo_entity_types()
    demo_research_abstract()
    demo_frequency_band_metadata()
    demo_context_extraction()
    demo_batch_processing()
    demo_confidence_filtering()
    demo_real_world_methods()
    demo_export_json()
    
    print("\n" + "=" * 80)
    print(" ✓ ALL DEMONSTRATIONS COMPLETE")
    print("=" * 80)
    print("\nThe EEG NER system is ready for:")
    print("  • Research paper analysis")
    print("  • Automated metadata extraction")
    print("  • Terminology indexing")
    print("  • Knowledge graph construction")
    print("  • Literature review assistance")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
