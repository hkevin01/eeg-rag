"""
Systematic Review Demo - Full Workflow

Demonstrates automated extraction, comparison, and reproducibility scoring
for systematic reviews of deep learning EEG papers.

Example workflow:
1. Load YAML extraction schema
2. Extract structured data from papers
3. Score reproducibility
4. Compare against baseline (Roy et al. 2019)
5. Generate reports and export results
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eeg_rag.review.extractor import SystematicReviewExtractor
from eeg_rag.review.comparator import SystematicReviewComparator, ReproducibilityScorer
import pandas as pd


def create_example_papers():
    """Create example papers for demonstration."""
    papers = [
        {
            "paper_id": "example_2020_1",
            "title": "Deep Convolutional Networks for Automated Seizure Detection in EEG",
            "authors": "Smith et al.",
            "year": 2020,
            "abstract": """
            We propose a deep convolutional neural network (CNN) for automated seizure 
            detection using the CHB-MIT scalp EEG database. Our end-to-end learning approach 
            achieves 98.5% accuracy without manual feature engineering. The model uses 5 
            convolutional layers followed by 2 fully connected layers. We validated using 
            5-fold cross-validation on 23 patients. Code is publicly available on GitHub 
            at https://github.com/example/seizure-cnn.
            """
        },
        {
            "paper_id": "example_2021_1",
            "title": "Transformer-Based Motor Imagery Classification for Brain-Computer Interfaces",
            "authors": "Johnson et al.",
            "year": 2021,
            "abstract": """
            This paper presents a Transformer architecture for motor imagery classification 
            in BCI systems using the BCI Competition IV 2a dataset. We achieve 87.3% accuracy 
            (kappa=0.83) with attention mechanisms capturing temporal dependencies. The model 
            was evaluated using leave-one-subject-out cross-validation on 9 subjects. 
            Saliency maps reveal which EEG channels contribute most to classification. 
            Source code will be made available upon publication.
            """
        },
        {
            "paper_id": "example_2022_1",
            "title": "Hybrid CNN-LSTM for Sleep Stage Classification",
            "authors": "Zhang et al.",
            "year": 2022,
            "abstract": """
            We develop a hybrid CNN-LSTM architecture for automatic sleep staging using 
            single-channel EEG from the Sleep-EDF database. The CNN extracts spatial features 
            while LSTM captures temporal transitions between sleep stages. We achieve 89.2% 
            accuracy with F1-score of 0.87 across 5 sleep stages. Feature extraction includes 
            bandpass filtering (0.5-30 Hz) and normalization. Hold-out validation on 153 
            subjects from PhysioNet. Dataset is publicly available but code is available 
            upon request from authors.
            """
        },
        {
            "paper_id": "example_2023_1",
            "title": "Self-Supervised Learning for EEG-Based Emotion Recognition",
            "authors": "Lee et al.",
            "year": 2023,
            "abstract": """
            We propose a self-supervised pre-training approach using contrastive learning 
            for emotion recognition from EEG signals. Using the DEAP dataset with 32 subjects, 
            we achieve 92.7% accuracy for valence-arousal classification. Transfer learning 
            from a large unlabeled EEG corpus improves performance by 8.3% over training from 
            scratch. The model uses a ResNet-style CNN backbone with attention pooling. 
            Time-series split validation ensures temporal generalization. Code and pre-trained 
            models available at https://github.com/example/emotion-ssl with BSD-3 license.
            """
        }
    ]
    return papers


def main():
    """Run systematic review demo."""
    print("="*80)
    print("SYSTEMATIC REVIEW EXTRACTION DEMO")
    print("="*80)
    
    # Step 1: Initialize extractor with schema
    print("\n[1/6] Loading extraction schema...")
    schema_path = Path(__file__).parent.parent / "schemas" / "dl_eeg_review_2019_schema.yaml"
    
    extractor = SystematicReviewExtractor(protocol=str(schema_path))
    print(f"   ✓ Loaded schema with {len(extractor.fields)} extraction fields")
    
    # Step 2: Create example papers
    print("\n[2/6] Loading papers for extraction...")
    papers = create_example_papers()
    print(f"   ✓ Loaded {len(papers)} papers")
    
    # Step 3: Extract structured data
    print("\n[3/6] Extracting structured data (using rule-based extraction)...")
    df = extractor.run(papers)
    print(f"   ✓ Extracted {len(df)} papers")
    print(f"   ✓ Fields extracted: {len([f for f in extractor.fields if f.name in df.columns])}")
    
    # Show sample extraction
    print("\n   Sample extraction:")
    sample_row = df.iloc[0]
    print(f"   Title: {sample_row['title'][:60]}...")
    print(f"   Architecture: {sample_row.get('architecture_type', 'N/A')}")
    print(f"   Task: {sample_row.get('task_type', 'N/A')}")
    print(f"   Dataset: {sample_row.get('dataset_name', 'N/A')}")
    print(f"   Code Available: {sample_row.get('code_available', 'N/A')}")
    
    # Step 4: Score reproducibility
    print("\n[4/6] Scoring reproducibility...")
    scorer = ReproducibilityScorer()
    scored_df = scorer.score_dataset(df)
    print(f"   ✓ Scored {len(scored_df)} papers")
    print(f"   ✓ Mean reproducibility score: {scored_df['reproducibility_score'].mean():.2f}/18")
    
    print("\n   Reproducibility breakdown:")
    repro_counts = scored_df['reproducibility_category'].value_counts()
    for category, count in repro_counts.items():
        print(f"     {category}: {count} ({count/len(scored_df)*100:.1f}%)")
    
    # Step 5: Export results
    print("\n[5/6] Exporting results...")
    output_dir = Path(__file__).parent.parent / "data" / "systematic_review"
    output_dir.mkdir(exist_ok=True)
    
    csv_path = output_dir / "extracted_papers.csv"
    extractor.export(str(csv_path), format="csv")
    print(f"   ✓ Saved CSV: {csv_path}")
    
    json_path = output_dir / "extracted_papers.json"
    extractor.export(str(json_path), format="json")
    print(f"   ✓ Saved JSON: {json_path}")
    
    # Step 6: Comparison (simulated - would need real Roy et al. 2019 data)
    print("\n[6/6] Comparison analysis...")
    print("   NOTE: Comparison requires baseline data from Roy et al. 2019")
    print("   To enable comparison:")
    print("     1. Create CSV with Roy et al. 2019 extracted data")
    print("     2. Save as 'data/systematic_review/roy_2019_baseline.csv'")
    print("     3. Run:")
    print("        comparator = SystematicReviewComparator('roy_2019_baseline.csv')")
    print("        comparison = comparator.compare(scored_df)")
    print("        print(comparison.summary())")
    
    # Show what fields would be compared
    print("\n   Fields configured for comparison:")
    print("     - Year distribution")
    print("     - Architecture shifts")
    print("     - Performance improvements")
    print("     - Reproducibility trends")
    print("     - Dataset usage patterns")
    print("     - Task distribution")
    
    # Low confidence extractions
    print("\n" + "="*80)
    print("LOW CONFIDENCE EXTRACTIONS (need manual review)")
    print("="*80)
    low_conf = extractor.get_low_confidence_extractions(threshold=0.7)
    if not low_conf.empty:
        print(f"\nFound {len(low_conf)} papers with low-confidence extractions:")
        # Get confidence columns
        conf_cols = [col for col in low_conf.columns if col.endswith('_confidence')]
        for idx in low_conf.index[:3]:  # Show first 3
            row = low_conf.loc[idx]
            print(f"\n  Paper: {row['title'][:60]}...")
            for col in conf_cols:
                conf = row[col]
                if conf < 0.7:
                    field_name = col.replace('_confidence', '')
                    print(f"    {field_name}: {conf:.2f} confidence")
    else:
        print("\n✓ All extractions have high confidence (>0.70)")
    
    # Summary statistics
    print("\n" + "="*80)
    print("EXTRACTION SUMMARY")
    print("="*80)
    print(f"\nTotal Papers Processed: {len(df)}")
    print(f"Year Range: {df['year'].min()}-{df['year'].max()}")
    print(f"\nArchitecture Distribution:")
    for arch, count in df['architecture_type'].value_counts().items():
        print(f"  {arch}: {count} ({count/len(df)*100:.1f}%)")
    print(f"\nTask Distribution:")
    for task, count in df['task_type'].value_counts().items():
        print(f"  {task}: {count} ({count/len(df)*100:.1f}%)")
    print(f"\nDatasets Used:")
    for dataset, count in df['dataset_name'].value_counts().items():
        print(f"  {dataset}: {count}")
    
    print(f"\n{scorer.generate_report(df)}")
    
    print("\n" + "="*80)
    print("Demo completed! Check output files in data/systematic_review/")
    print("="*80)


if __name__ == "__main__":
    main()
