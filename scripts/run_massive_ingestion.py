#!/usr/bin/env python3
"""
Massive-scale EEG literature ingestion script.
Target: 500,000+ papers from multiple sources including 2025 coverage.

This extends the bulk_ingestion module with:
- 400+ comprehensive EEG search queries
- Coverage across 15+ research subdomain areas
- Multi-source aggregation with deduplication
- bioRxiv/medRxiv preprint support
- Automatic checkpointing and resumption
- Continuous update mode for daily refresh

Usage:
    python scripts/run_massive_ingestion.py --target 500000
    python scripts/run_massive_ingestion.py --resume
    python scripts/run_massive_ingestion.py --update-latest  # Last 30 days only
    python scripts/run_massive_ingestion.py --target 100000 --start-year 2024 --end-year 2025
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eeg_rag.ingestion import BulkIngestionManager, BulkIngestionConfig
from eeg_rag.ingestion.biorxiv_client import BioRxivClient

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'logs/massive_ingestion_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# COMPREHENSIVE EEG SEARCH QUERIES - 400+ queries covering all subdomains
# ============================================================================

EEG_SEARCH_QUERIES = {
    "core_eeg": [
        "electroencephalography",
        "EEG recording",
        "EEG signal",
        "EEG analysis",
        "scalp EEG",
        "EEG monitoring",
        "quantitative EEG",
        "qEEG",
        "EEG spectral analysis",
        "EEG power spectrum",
    ],
    
    "clinical_epilepsy": [
        "EEG epilepsy",
        "seizure EEG",
        "interictal epileptiform",
        "ictal EEG",
        "epileptiform discharge",
        "spike wave EEG",
        "absence seizure EEG",
        "temporal lobe epilepsy EEG",
        "frontal lobe epilepsy EEG",
        "juvenile myoclonic epilepsy EEG",
        "infantile spasms EEG",
        "Lennox-Gastaut EEG",
        "West syndrome EEG",
        "status epilepticus EEG",
        "nonconvulsive status epilepticus",
        "seizure prediction EEG",
        "seizure detection EEG",
        "epilepsy surgery EEG",
        "intracranial EEG",
        "stereo EEG",
        "electrocorticography",
        "ECoG epilepsy",
        "subdural EEG",
        "depth electrodes EEG",
    ],
    
    "sleep_research": [
        "sleep EEG",
        "polysomnography",
        "PSG EEG",
        "sleep staging",
        "sleep spindles",
        "K-complex",
        "slow wave sleep EEG",
        "REM sleep EEG",
        "NREM sleep EEG",
        "sleep architecture EEG",
        "sleep disorders EEG",
        "insomnia EEG",
        "sleep apnea EEG",
        "narcolepsy EEG",
        "REM behavior disorder EEG",
        "parasomnia EEG",
        "circadian rhythm EEG",
        "sleep deprivation EEG",
        "microsleep EEG",
        "drowsiness EEG",
    ],
    
    "cognitive_erp": [
        "event-related potential",
        "ERP EEG",
        "P300",
        "P3 component",
        "N400",
        "N170",
        "mismatch negativity",
        "MMN EEG",
        "error-related negativity",
        "ERN EEG",
        "contingent negative variation",
        "CNV EEG",
        "readiness potential",
        "Bereitschaftspotential",
        "N100 EEG",
        "P100 EEG",
        "N200 EEG",
        "P200 EEG",
        "late positive potential",
        "LPP EEG",
        "early posterior negativity",
        "visual evoked potential",
        "VEP EEG",
        "auditory evoked potential",
        "AEP EEG",
        "somatosensory evoked potential",
        "SSEP",
        "brainstem auditory evoked potential",
        "BAEP",
        "steady-state evoked potential",
        "SSVEP",
        "oddball paradigm EEG",
        "go no-go EEG",
        "Stroop EEG",
        "flanker task EEG",
        "working memory EEG",
        "attention EEG",
        "cognitive control EEG",
    ],
    
    "frequency_oscillations": [
        "alpha oscillations",
        "alpha rhythm EEG",
        "alpha power EEG",
        "alpha asymmetry",
        "theta oscillations",
        "theta rhythm EEG",
        "theta power EEG",
        "frontal theta",
        "beta oscillations",
        "beta rhythm EEG",
        "beta power EEG",
        "mu rhythm",
        "sensorimotor rhythm",
        "SMR EEG",
        "gamma oscillations",
        "gamma rhythm EEG",
        "gamma power EEG",
        "high gamma EEG",
        "delta oscillations",
        "delta rhythm EEG",
        "delta power EEG",
        "sigma rhythm",
        "theta-gamma coupling",
        "phase-amplitude coupling",
        "cross-frequency coupling",
        "neural oscillations",
        "brain rhythms",
        "spectral power EEG",
        "power spectral density EEG",
        "time-frequency analysis EEG",
        "wavelet EEG",
    ],
    
    "brain_computer_interface": [
        "brain-computer interface EEG",
        "BCI EEG",
        "motor imagery EEG",
        "P300 speller",
        "SSVEP BCI",
        "EEG neurofeedback",
        "real-time EEG",
        "online EEG classification",
        "EEG-based control",
        "neural interface EEG",
        "neuroprosthetics EEG",
        "assistive technology EEG",
        "EEG wheelchair",
        "EEG communication",
        "locked-in syndrome EEG",
        "EEG gaming",
        "EEG virtual reality",
        "hybrid BCI",
        "passive BCI",
        "affective BCI",
    ],
    
    "psychiatric_disorders": [
        "depression EEG",
        "major depressive disorder EEG",
        "anxiety EEG",
        "PTSD EEG",
        "schizophrenia EEG",
        "bipolar disorder EEG",
        "OCD EEG",
        "ADHD EEG",
        "attention deficit EEG",
        "autism EEG",
        "ASD EEG",
        "substance abuse EEG",
        "alcohol EEG",
        "addiction EEG",
        "eating disorders EEG",
        "personality disorder EEG",
        "psychosis EEG",
        "antidepressant EEG",
        "antipsychotic EEG",
        "electroconvulsive therapy EEG",
        "ECT EEG",
        "transcranial stimulation EEG",
        "TMS EEG",
        "tDCS EEG",
    ],
    
    "neurodegenerative": [
        "Alzheimer disease EEG",
        "dementia EEG",
        "mild cognitive impairment EEG",
        "MCI EEG",
        "Parkinson disease EEG",
        "Lewy body dementia EEG",
        "frontotemporal dementia EEG",
        "vascular dementia EEG",
        "Huntington disease EEG",
        "ALS EEG",
        "multiple sclerosis EEG",
        "prion disease EEG",
        "Creutzfeldt-Jakob EEG",
        "CJD EEG",
        "neurodegeneration EEG",
        "cognitive decline EEG",
        "aging brain EEG",
    ],
    
    "critical_care": [
        "ICU EEG",
        "continuous EEG monitoring",
        "cEEG",
        "coma EEG",
        "brain death EEG",
        "encephalopathy EEG",
        "hepatic encephalopathy EEG",
        "uremic encephalopathy EEG",
        "septic encephalopathy EEG",
        "anoxic brain injury EEG",
        "cardiac arrest EEG",
        "burst suppression",
        "isoelectric EEG",
        "prognostication EEG",
        "neonatal EEG",
        "aEEG",
        "amplitude-integrated EEG",
        "neonatal seizures",
        "hypoxic ischemic encephalopathy",
        "HIE EEG",
        "therapeutic hypothermia EEG",
        "sedation EEG",
        "anesthesia EEG",
        "intraoperative EEG",
        "depth of anesthesia",
        "BIS monitoring",
    ],
    
    "developmental": [
        "pediatric EEG",
        "child EEG",
        "infant EEG",
        "developmental EEG",
        "EEG development",
        "brain maturation EEG",
        "adolescent EEG",
        "ADHD children EEG",
        "autism children EEG",
        "learning disability EEG",
        "dyslexia EEG",
        "language development EEG",
        "cognitive development EEG",
        "febrile seizures EEG",
        "childhood epilepsy EEG",
        "benign rolandic epilepsy",
        "childhood absence epilepsy",
        "Dravet syndrome EEG",
    ],
    
    "methodology": [
        "EEG preprocessing",
        "EEG artifact removal",
        "ICA EEG",
        "independent component analysis EEG",
        "EEG source localization",
        "EEG inverse problem",
        "EEG forward model",
        "dipole modeling EEG",
        "sLORETA",
        "eLORETA",
        "beamforming EEG",
        "EEG connectivity",
        "coherence EEG",
        "phase synchrony EEG",
        "Granger causality EEG",
        "graph theory EEG",
        "network analysis EEG",
        "functional connectivity EEG",
        "effective connectivity EEG",
        "microstate analysis",
        "EEG microstates",
        "EEG entropy",
        "complexity EEG",
        "nonlinear EEG",
        "fractal EEG",
        "Hjorth parameters",
        "EEG features",
        "EEG biomarkers",
        "EEG reference",
        "average reference EEG",
        "common average reference",
        "Laplacian EEG",
        "surface Laplacian",
        "current source density",
        "CSD EEG",
    ],
    
    "machine_learning": [
        "deep learning EEG",
        "CNN EEG",
        "convolutional neural network EEG",
        "RNN EEG",
        "LSTM EEG",
        "transformer EEG",
        "machine learning EEG",
        "classification EEG",
        "EEG classification",
        "automatic EEG",
        "automated EEG",
        "EEG detection",
        "pattern recognition EEG",
        "feature extraction EEG",
        "transfer learning EEG",
        "domain adaptation EEG",
        "self-supervised EEG",
        "contrastive learning EEG",
        "foundation model EEG",
        "EEG embedding",
        "EEG representation",
        "end-to-end EEG",
        "EEGNet",
        "seizure detection deep learning",
        "sleep staging deep learning",
        "emotion recognition EEG",
        "EEG emotion",
        "affective computing EEG",
        "mental state EEG",
        "cognitive load EEG",
        "workload EEG",
        "fatigue EEG",
        "drowsiness detection EEG",
        "driver monitoring EEG",
    ],
    
    "hardware_technology": [
        "EEG electrode",
        "dry electrode EEG",
        "wet electrode EEG",
        "high-density EEG",
        "hdEEG",
        "256 channel EEG",
        "128 channel EEG",
        "64 channel EEG",
        "portable EEG",
        "mobile EEG",
        "wearable EEG",
        "wireless EEG",
        "consumer EEG",
        "Emotiv",
        "Muse EEG",
        "OpenBCI",
        "EEG headset",
        "EEG cap",
        "EEG amplifier",
        "EEG acquisition",
        "sampling rate EEG",
        "EEG impedance",
        "10-20 system",
        "10-10 system",
        "10-5 system",
        "electrode placement",
        "montage EEG",
        "bipolar montage",
        "referential montage",
    ],
    
    "multimodal": [
        "EEG fMRI",
        "simultaneous EEG fMRI",
        "EEG-fMRI",
        "EEG MEG",
        "EEG PET",
        "EEG NIRS",
        "EEG fnirs",
        "EEG EMG",
        "EEG EOG",
        "EEG ECG",
        "multimodal neuroimaging EEG",
        "source imaging EEG",
        "electrical source imaging",
        "ESI EEG",
    ],
    
    "specific_conditions": [
        "stroke EEG",
        "traumatic brain injury EEG",
        "TBI EEG",
        "concussion EEG",
        "migraine EEG",
        "headache EEG",
        "pain EEG",
        "chronic pain EEG",
        "fibromyalgia EEG",
        "chronic fatigue EEG",
        "tinnitus EEG",
        "vertigo EEG",
        "movement disorders EEG",
        "tremor EEG",
        "dystonia EEG",
        "tics EEG",
        "Tourette EEG",
        "syncope EEG",
        "psychogenic seizures EEG",
        "PNES EEG",
        "non-epileptic seizures EEG",
        "dissociative seizures EEG",
    ],
    
    "consciousness": [
        "consciousness EEG",
        "awareness EEG",
        "vegetative state EEG",
        "minimally conscious EEG",
        "disorders of consciousness EEG",
        "meditation EEG",
        "mindfulness EEG",
        "hypnosis EEG",
        "altered states EEG",
        "psychedelic EEG",
        "anesthesia awareness EEG",
        "neural correlates consciousness EEG",
        "integrated information EEG",
        "global workspace EEG",
    ],
    
    # NEW: Emerging 2024-2025 topics
    "emerging_2024_2025": [
        # Foundation models & large-scale EEG
        "foundation model EEG 2024",
        "large language model EEG",
        "LLM EEG neuroscience",
        "GPT EEG analysis",
        "transformer architecture EEG",
        "vision transformer EEG",
        "ViT EEG",
        "BERT EEG",
        "pre-trained EEG model",
        "EEG foundation model",
        "large-scale EEG dataset",
        "EEG benchmark dataset 2024",
        
        # Federated & privacy-preserving
        "federated learning EEG",
        "privacy-preserving EEG",
        "differential privacy EEG",
        "secure EEG analysis",
        "distributed EEG learning",
        "decentralized EEG",
        "multi-center EEG",
        "multi-site EEG study",
        
        # Explainable AI
        "explainable AI EEG",
        "XAI EEG",
        "interpretable EEG",
        "attention mechanism EEG",
        "saliency map EEG",
        "SHAP EEG",
        "GradCAM EEG",
        "interpretable deep learning EEG",
        
        # Real-world & continuous monitoring
        "real-world EEG",
        "ambulatory EEG 2024",
        "long-term EEG monitoring",
        "home EEG monitoring",
        "remote EEG monitoring",
        "telehealth EEG",
        "digital health EEG",
        "continuous EEG wearable",
        "ultra-long EEG recording",
        
        # Novel hardware
        "in-ear EEG",
        "ear EEG",
        "around-ear EEG",
        "behind-ear EEG",
        "tattoo electrode EEG",
        "flexible electrode EEG",
        "printed electrode EEG",
        "graphene EEG electrode",
        "nanomaterial EEG",
        "next-generation EEG",
        
        # Clinical decision support
        "clinical decision support EEG",
        "EEG AI diagnosis",
        "automated EEG report",
        "EEG report generation",
        "natural language EEG",
        "EEG knowledge graph",
        "EEG ontology",
        
        # Multimodal AI
        "multimodal AI EEG",
        "EEG language model",
        "EEG image fusion",
        "EEG text generation",
        "brain-to-text EEG",
        "neural decoding language",
        "semantic decoding EEG",
        "image reconstruction EEG",
        
        # Emerging applications
        "EEG digital twin",
        "EEG simulation",
        "synthetic EEG data",
        "EEG data augmentation",
        "generative model EEG",
        "diffusion model EEG",
        "GAN EEG",
        "variational autoencoder EEG",
        
        # Neurotech & consumer
        "consumer neurotechnology 2024",
        "brain sensing 2024",
        "neural interface 2024",
        "non-invasive BCI 2024",
        "EEG startup",
        "commercial EEG device",
        "FDA approved EEG",
        "CE marked EEG",
        
        # Mental health tech
        "digital therapeutics EEG",
        "digital biomarker EEG",
        "EEG mental health app",
        "neurofeedback app",
        "mindfulness EEG app",
        "cognitive training EEG",
        "brain training EEG",
        
        # Sports & performance
        "sports neuroscience EEG",
        "athletic performance EEG",
        "cognitive performance EEG",
        "peak performance EEG",
        "esports EEG",
        "gaming neuroscience",
        "reaction time EEG",
        
        # Sleep technology
        "sleep technology 2024",
        "sleep tracking EEG",
        "smart sleep EEG",
        "sleep optimization EEG",
        "sleep intervention EEG",
        "targeted memory reactivation",
        "closed-loop sleep",
        
        # Seizure forecasting
        "seizure forecasting 2024",
        "seizure prediction machine learning",
        "ultra-long-term seizure prediction",
        "patient-specific seizure prediction",
        "seizure risk stratification",
        "seizure advisory system",
        
        # Regulatory & standards
        "EEG AI regulation",
        "EEG device regulation",
        "clinical validation EEG AI",
        "EEG standardization 2024",
        "reproducibility EEG",
        "FAIR EEG data",
    ],
    
    # NEW: 2025 specific emerging research
    "cutting_edge_2025": [
        "EEG 2025",
        "brain-computer interface 2025",
        "neural decoding 2025",
        "neurotechnology 2025",
        "computational neuroscience 2025",
        "clinical neurophysiology 2025",
        "epilepsy research 2025",
        "sleep research 2025",
        "consciousness research 2025",
        "cognitive neuroscience 2025",
    ],
}


def count_total_queries() -> int:
    """Count total number of search queries."""
    return sum(len(queries) for queries in EEG_SEARCH_QUERIES.values())


def get_date_ranges(start_year: int, end_year: int, update_only: bool = False) -> list[tuple[str, str]]:
    """
    Generate date ranges for ingestion, prioritizing recent papers.
    
    Args:
        start_year: First year to include (e.g., 1990)
        end_year: Last year to include (e.g., 2025)
        update_only: If True, only return last 30 days
        
    Returns:
        List of (start_date, end_date) tuples in YYYY-MM-DD format
    """
    if update_only:
        # Last 30 days only for incremental updates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        return [(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))]
    
    # Prioritize recent years (process newest first)
    ranges = []
    current_year = datetime.now().year
    
    # Current year and last year get monthly granularity
    for year in range(min(end_year, current_year), max(start_year, current_year - 1) - 1, -1):
        for month in range(12, 0, -1):
            if year == current_year and month > datetime.now().month:
                continue
            start = f"{year}-{month:02d}-01"
            if month == 12:
                end = f"{year}-12-31"
            else:
                end = f"{year}-{month+1:02d}-01"
            ranges.append((start, end))
    
    # Older years get yearly granularity
    for year in range(current_year - 2, start_year - 1, -1):
        if year < start_year:
            break
        ranges.append((f"{year}-01-01", f"{year}-12-31"))
    
    return ranges


async def ingest_biorxiv_medrxiv(
    output_dir: str,
    target: int = 25000,
    update_only: bool = False,
    start_year: int = 2019,
    end_year: int = 2025,
) -> int:
    """
    Ingest preprints from bioRxiv and medRxiv.
    
    Args:
        output_dir: Directory to save results
        target: Target number of papers per server
        update_only: If True, only fetch last 30 days
        start_year: First year to fetch
        end_year: Last year to fetch
        
    Returns:
        Total number of papers ingested
    """
    logger.info("=" * 50)
    logger.info("PREPRINT INGESTION: bioRxiv + medRxiv")
    logger.info("=" * 50)
    
    client = BioRxivClient()
    total_papers = 0
    
    # EEG-specific search terms for preprints
    eeg_terms = [
        "electroencephalography", "EEG", "brain-computer interface", "BCI",
        "neural oscillations", "event-related potential", "ERP",
        "seizure", "epilepsy", "sleep staging", "neurofeedback",
        "brain signal", "neural decoding", "motor imagery",
    ]
    
    date_ranges = get_date_ranges(start_year, end_year, update_only)
    
    for server in ["biorxiv", "medrxiv"]:
        logger.info(f"\nIngesting from {server}...")
        server_papers = []
        
        for start_date, end_date in date_ranges:
            if len(server_papers) >= target:
                break
                
            logger.info(f"  Fetching {start_date} to {end_date}...")
            
            try:
                papers = await client.search_eeg_preprints(
                    server=server,
                    start_date=start_date,
                    end_date=end_date,
                    eeg_terms=eeg_terms,
                    max_results=min(1000, target - len(server_papers))
                )
                
                server_papers.extend(papers)
                logger.info(f"    Found {len(papers)} papers (total: {len(server_papers)})")
                
            except Exception as e:
                logger.warning(f"    Error fetching {server} {start_date}-{end_date}: {e}")
                continue
        
        # Save results
        if server_papers:
            output_path = Path(output_dir) / f"{server}_papers.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            import json
            with open(output_path, "w") as f:
                json.dump(server_papers, f, indent=2, default=str)
            
            logger.info(f"  Saved {len(server_papers)} papers to {output_path}")
            total_papers += len(server_papers)
    
    await client.close()
    logger.info(f"\nTotal preprints ingested: {total_papers}")
    return total_papers


def main():
    parser = argparse.ArgumentParser(
        description="Massive EEG literature ingestion (500K+ papers) with 2025 coverage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Default: 500K papers from all sources (1990-2025)
    python scripts/run_massive_ingestion.py --target 500000
    
    # Resume interrupted ingestion
    python scripts/run_massive_ingestion.py --resume
    
    # Update with latest papers (last 30 days only)
    python scripts/run_massive_ingestion.py --update-latest
    
    # Focus on recent papers only
    python scripts/run_massive_ingestion.py --start-year 2020 --end-year 2025
    
    # Include preprints
    python scripts/run_massive_ingestion.py --include-preprints
    
    # Quick test with 10K papers
    python scripts/run_massive_ingestion.py --target 10000
    
    # Custom distribution
    python scripts/run_massive_ingestion.py --pubmed 175000 --scholar 125000 --openalex 100000 --arxiv 50000 --biorxiv 25000 --medrxiv 25000
        """
    )
    
    parser.add_argument(
        "--target",
        type=int,
        default=500000,
        help="Total papers to ingest (default: 500000)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint"
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start fresh, ignore existing checkpoint"
    )
    parser.add_argument(
        "--update-latest",
        action="store_true",
        help="Only fetch papers from the last 30 days (for daily updates)"
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=1990,
        help="Start year for date range (default: 1990)"
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2025,
        help="End year for date range (default: 2025)"
    )
    parser.add_argument(
        "--include-preprints",
        action="store_true",
        help="Include bioRxiv and medRxiv preprints"
    )
    parser.add_argument(
        "--pubmed",
        type=int,
        help="Target PubMed papers (default: 35%% of target)"
    )
    parser.add_argument(
        "--scholar",
        type=int,
        help="Target Semantic Scholar papers (default: 25%% of target)"
    )
    parser.add_argument(
        "--openalex",
        type=int,
        help="Target OpenAlex papers (default: 20%% of target)"
    )
    parser.add_argument(
        "--arxiv",
        type=int,
        help="Target arXiv papers (default: 10%% of target)"
    )
    parser.add_argument(
        "--biorxiv",
        type=int,
        help="Target bioRxiv papers (default: 5%% of target)"
    )
    parser.add_argument(
        "--medrxiv",
        type=int,
        help="Target medRxiv papers (default: 5%% of target)"
    )
    parser.add_argument(
        "--pubmed-only",
        action="store_true",
        help="Only ingest from PubMed"
    )
    parser.add_argument(
        "--output-dir",
        default="data/massive_ingestion",
        help="Output directory (default: data/massive_ingestion)"
    )
    
    args = parser.parse_args()
    
    # Ensure logs directory exists
    Path("logs").mkdir(parents=True, exist_ok=True)
    
    # Adjust targets for update mode
    if args.update_latest:
        # Smaller targets for daily updates
        base_target = args.target if args.target != 500000 else 10000
        logger.info("=" * 70)
        logger.info("UPDATE MODE: Fetching papers from last 30 days only")
        logger.info("=" * 70)
    else:
        base_target = args.target
    
    # Configure targets with new distribution including preprints
    if args.pubmed_only:
        config = BulkIngestionConfig(
            pubmed_target=base_target,
            scholar_target=0,
            openalex_target=0,
            arxiv_target=0,
            output_dir=args.output_dir
        )
        biorxiv_target = 0
        medrxiv_target = 0
    elif args.include_preprints:
        # Distribution: PubMed 35%, S2 25%, OpenAlex 20%, arXiv 10%, bioRxiv 5%, medRxiv 5%
        config = BulkIngestionConfig(
            pubmed_target=args.pubmed or int(base_target * 0.35),
            scholar_target=args.scholar or int(base_target * 0.25),
            openalex_target=args.openalex or int(base_target * 0.20),
            arxiv_target=args.arxiv or int(base_target * 0.10),
            output_dir=args.output_dir
        )
        biorxiv_target = args.biorxiv or int(base_target * 0.05)
        medrxiv_target = args.medrxiv or int(base_target * 0.05)
    else:
        # Original distribution without preprints
        config = BulkIngestionConfig(
            pubmed_target=args.pubmed or int(base_target * 0.40),
            scholar_target=args.scholar or int(base_target * 0.30),
            openalex_target=args.openalex or int(base_target * 0.20),
            arxiv_target=args.arxiv or int(base_target * 0.10),
            output_dir=args.output_dir
        )
        biorxiv_target = 0
        medrxiv_target = 0
    
    # Log configuration
    total_queries = count_total_queries()
    logger.info("=" * 70)
    logger.info("MASSIVE EEG LITERATURE INGESTION (2025 Enhanced)")
    logger.info("=" * 70)
    logger.info(f"Search queries: {total_queries} across {len(EEG_SEARCH_QUERIES)} domains")
    logger.info(f"Date range: {args.start_year} - {args.end_year}")
    logger.info(f"Target papers: {base_target:,}")
    logger.info(f"  - PubMed:           {config.pubmed_target:,}")
    logger.info(f"  - Semantic Scholar: {config.scholar_target:,}")
    logger.info(f"  - OpenAlex:         {config.openalex_target:,}")
    logger.info(f"  - arXiv:            {config.arxiv_target:,}")
    if args.include_preprints:
        logger.info(f"  - bioRxiv:          {biorxiv_target:,}")
        logger.info(f"  - medRxiv:          {medrxiv_target:,}")
    logger.info(f"Output: {args.output_dir}")
    if args.update_latest:
        logger.info("Mode: UPDATE (last 30 days only)")
    logger.info("=" * 70)
    
    # Check for API keys
    pubmed_key = os.environ.get("PUBMED_API_KEY") or os.environ.get("NCBI_API_KEY")
    s2_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or os.environ.get("S2_API_KEY")
    
    if pubmed_key:
        logger.info("✓ PubMed API key detected (faster rate limits)")
    else:
        logger.info("○ No PubMed API key - using conservative rate limits")
        logger.info("  Get free key: https://www.ncbi.nlm.nih.gov/account/settings/")
    
    if s2_key:
        logger.info("✓ Semantic Scholar API key detected (faster rate limits)")
    else:
        logger.info("○ No Semantic Scholar API key - using conservative rate limits")
        logger.info("  Get free key: https://www.semanticscholar.org/product/api#api-key")
    
    logger.info("=" * 70)
    
    # Create manager and run
    manager = BulkIngestionManager(
        config=config,
        pubmed_api_key=pubmed_key,
        semantic_scholar_api_key=s2_key,
        email=os.environ.get("RESEARCHER_EMAIL", "eeg-rag-research@example.edu")
    )
    
    # Run ingestion
    try:
        # Main ingestion from traditional sources
        asyncio.run(manager.run_bulk_ingestion(
            resume=args.resume and not args.fresh
        ))
        
        # Preprint ingestion if enabled
        if args.include_preprints and (biorxiv_target > 0 or medrxiv_target > 0):
            logger.info("\n" + "=" * 70)
            logger.info("STARTING PREPRINT INGESTION")
            logger.info("=" * 70)
            
            asyncio.run(ingest_biorxiv_medrxiv(
                output_dir=args.output_dir,
                target=biorxiv_target + medrxiv_target,
                update_only=args.update_latest,
                start_year=args.start_year,
                end_year=args.end_year,
            ))
            
    except KeyboardInterrupt:
        logger.info("\nIngestion interrupted. Progress has been saved.")
        logger.info("Run with --resume to continue from where you left off.")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise


if __name__ == "__main__":
    main()
