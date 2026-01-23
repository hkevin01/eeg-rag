#!/usr/bin/env python3
"""
EEG-RAG Production Setup Script

This script helps new users set up the production paper database with 500K+ papers.
It provides interactive options and progress tracking.

Usage:
    python scripts/setup_production.py              # Interactive setup
    python scripts/setup_production.py --full       # Full 500K ingestion
    python scripts/setup_production.py --quick      # Quick 10K test
    python scripts/setup_production.py --status     # Check current status
"""

import asyncio
import argparse
import logging
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure logs directory exists
Path("logs").mkdir(exist_ok=True)


def print_banner():
    """Print welcome banner."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║     ███████╗███████╗ ██████╗       ██████╗  █████╗  ██████╗     ║
║     ██╔════╝██╔════╝██╔════╝       ██╔══██╗██╔══██╗██╔════╝     ║
║     █████╗  █████╗  ██║  ███╗█████╗██████╔╝███████║██║  ███╗    ║
║     ██╔══╝  ██╔══╝  ██║   ██║╚════╝██╔══██╗██╔══██║██║   ██║    ║
║     ███████╗███████╗╚██████╔╝      ██║  ██║██║  ██║╚██████╔╝    ║
║     ╚══════╝╚══════╝ ╚═════╝       ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝     ║
║                                                                  ║
║              Production Setup - 500K+ Paper Database             ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)


def get_current_status():
    """Get current database status."""
    try:
        from eeg_rag.db.paper_store import get_paper_store
        store = get_paper_store()
        stats = store.get_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return None


def print_status(stats):
    """Print formatted status."""
    if not stats:
        print("\n❌ Database not initialized or error occurred\n")
        return
    
    total = stats.get('total_papers', 0)
    by_source = stats.get('by_source', {})
    year_range = stats.get('year_range', {})
    db_size = stats.get('db_size_mb', 0)
    
    target = 500000
    progress = min(100, (total / target) * 100)
    bar_filled = int(progress / 2)
    bar_empty = 50 - bar_filled
    
    print(f"""
┌─────────────────────────────────────────────────────────────────┐
│                    📊 Database Status                           │
├─────────────────────────────────────────────────────────────────┤
│  Total Papers: {total:>10,} / {target:,} ({progress:.1f}%)            │
│  Database Size: {db_size:>8.1f} MB                                    │
│                                                                 │
│  Progress: [{'█' * bar_filled}{'░' * bar_empty}]    │
│                                                                 │
│  Papers by Source:                                              │""")
    
    for source, count in by_source.items():
        print(f"│    • {source:<20}: {count:>10,}                       │")
    
    if year_range.get('min') and year_range.get('max'):
        print(f"""│                                                                 │
│  Year Range: {year_range.get('min', 'N/A')} - {year_range.get('max', 'N/A')}                                   │
│  PMID Coverage: {stats.get('pmid_coverage', 0):.1f}%                                       │
│  DOI Coverage: {stats.get('doi_coverage', 0):.1f}%                                        │""")
    
    print("└─────────────────────────────────────────────────────────────────┘")


def estimate_time(target: int, rate_per_minute: int = 1000) -> str:
    """Estimate ingestion time."""
    minutes = target / rate_per_minute
    if minutes < 60:
        return f"{int(minutes)} minutes"
    elif minutes < 1440:
        return f"{minutes / 60:.1f} hours"
    else:
        return f"{minutes / 1440:.1f} days"


async def run_ingestion(target: int, resume: bool = True):
    """Run the ingestion process."""
    from eeg_rag.db.paper_store import get_paper_store, Paper
    from eeg_rag.ingestion.openalex_client import OpenAlexClient
    
    paper_store = get_paper_store()
    current_count = paper_store.get_total_count()
    
    if current_count >= target:
        print(f"\n✅ Already have {current_count:,} papers (target: {target:,})")
        return
    
    remaining = target - current_count
    print(f"\n📥 Starting ingestion: {remaining:,} papers to collect")
    print(f"   Current: {current_count:,} | Target: {target:,}")
    print(f"   Estimated time: {estimate_time(remaining)}")
    print("\n   Press Ctrl+C to pause (progress is saved automatically)\n")
    
    client = OpenAlexClient(email="eeg-rag-production@research.edu")
    
    # EEG search queries for comprehensive coverage
    search_queries = [
        "electroencephalography",
        "EEG brain",
        "brain-computer interface EEG",
        "event-related potential ERP",
        "epilepsy EEG seizure",
        "sleep EEG polysomnography",
        "P300 cognitive",
        "theta oscillation brain",
        "alpha rhythm EEG",
        "gamma oscillation neural",
        "motor imagery BCI",
        "steady-state evoked potential SSVEP",
        "attention EEG neural",
        "working memory EEG",
        "emotion recognition EEG",
    ]
    
    papers_collected = []
    batch_size = 1000
    total_added = 0
    start_time = datetime.now()
    
    per_query = (remaining // len(search_queries)) + 1000
    
    try:
        for query_idx, query in enumerate(search_queries):
            if total_added >= remaining:
                break
            
            print(f"\n🔍 Query {query_idx + 1}/{len(search_queries)}: '{query}'")
            
            query_count = 0
            try:
                async for work in client.search_works(
                    query=query,
                    from_year=2000,
                    max_results=per_query
                ):
                    if total_added >= remaining:
                        break
                    
                    # Extract data
                    pub_year = None
                    if work.publication_date:
                        try:
                            pub_year = work.publication_date.year
                        except:
                            pass
                    
                    author_names = []
                    for a in (work.authors or []):
                        name = a.get('name', '')
                        if name:
                            author_names.append(name)
                    
                    concept_names = []
                    for c in (work.concepts or []):
                        if isinstance(c, dict):
                            concept_names.append(c.get('name', ''))
                        elif isinstance(c, str):
                            concept_names.append(c)
                    
                    paper = Paper(
                        paper_id=work.openalex_id or f"openalex_{total_added}",
                        title=work.title or "",
                        abstract=work.abstract or "",
                        authors=author_names[:20],  # Limit authors
                        year=pub_year,
                        source="openalex",
                        doi=work.doi,
                        pmid=work.pmid,
                        openalex_id=work.openalex_id,
                        url=work.pdf_url,
                        journal=work.journal or "",
                        citation_count=work.citation_count or 0,
                        keywords=concept_names[:10]
                    )
                    papers_collected.append(paper)
                    query_count += 1
                    
                    # Batch insert
                    if len(papers_collected) >= batch_size:
                        added, updated, skipped = paper_store.add_papers_batch(
                            papers_collected, update_if_exists=False
                        )
                        total_added += added
                        papers_collected = []
                        
                        # Progress update
                        elapsed = (datetime.now() - start_time).total_seconds()
                        rate = total_added / (elapsed / 60) if elapsed > 0 else 0
                        eta = estimate_time(remaining - total_added, int(rate)) if rate > 0 else "calculating..."
                        
                        current_total = current_count + total_added
                        progress = (current_total / target) * 100
                        
                        print(f"\r   Progress: {current_total:,}/{target:,} ({progress:.1f}%) | "
                              f"Rate: {rate:.0f}/min | ETA: {eta}     ", end="", flush=True)
                
            except Exception as e:
                logger.warning(f"Error with query '{query}': {e}")
                continue
            
            print(f"\n   ✓ Collected {query_count:,} from '{query}'")
        
        # Final batch
        if papers_collected:
            added, updated, skipped = paper_store.add_papers_batch(
                papers_collected, update_if_exists=False
            )
            total_added += added
        
        # Summary
        elapsed = (datetime.now() - start_time).total_seconds()
        final_stats = paper_store.get_statistics()
        
        print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                    ✅ Ingestion Complete!                        ║
╠══════════════════════════════════════════════════════════════════╣
║  Papers Added:     {total_added:>10,}                                  ║
║  Total in Database: {final_stats['total_papers']:>9,}                                  ║
║  Time Elapsed:     {elapsed/60:>10.1f} minutes                          ║
║  Database Size:    {final_stats['db_size_mb']:>10.1f} MB                              ║
╚══════════════════════════════════════════════════════════════════╝
        """)
        
    except KeyboardInterrupt:
        # Save remaining papers
        if papers_collected:
            added, _, _ = paper_store.add_papers_batch(papers_collected, update_if_exists=False)
            total_added += added
        
        final_stats = paper_store.get_statistics()
        print(f"""
        
⏸️  Ingestion paused. Progress saved!
   Papers added this session: {total_added:,}
   Total in database: {final_stats['total_papers']:,}
   
   Run again to continue from where you left off.
        """)


def interactive_setup():
    """Run interactive setup."""
    print_banner()
    
    # Check current status
    stats = get_current_status()
    print_status(stats)
    
    current = stats.get('total_papers', 0) if stats else 0
    target = 500000
    
    if current >= target:
        print("\n✅ Production database is ready! You have 500K+ papers.\n")
        return
    
    print("""
┌─────────────────────────────────────────────────────────────────┐
│                    📦 Ingestion Options                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [1] Quick Start   - 10,000 papers  (~10 minutes)              │
│  [2] Standard      - 100,000 papers (~2 hours)                 │
│  [3] Full Production - 500,000 papers (~10 hours)              │
│  [4] Custom        - Specify your own target                   │
│  [5] Exit                                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
    """)
    
    try:
        choice = input("Select option [1-5]: ").strip()
        
        if choice == "1":
            target = 10000
        elif choice == "2":
            target = 100000
        elif choice == "3":
            target = 500000
        elif choice == "4":
            try:
                target = int(input("Enter target number of papers: ").strip())
            except ValueError:
                print("Invalid number. Using 10,000.")
                target = 10000
        elif choice == "5":
            print("\nExiting. Run again when ready!\n")
            return
        else:
            print("Invalid choice. Using Quick Start (10,000 papers).")
            target = 10000
        
        print(f"\n🚀 Starting ingestion to {target:,} papers...")
        print("   (Press Ctrl+C anytime to pause - progress is saved)\n")
        
        asyncio.run(run_ingestion(target))
        
    except KeyboardInterrupt:
        print("\n\nSetup cancelled. Run again when ready!\n")


def main():
    parser = argparse.ArgumentParser(
        description="EEG-RAG Production Setup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/setup_production.py              # Interactive setup
  python scripts/setup_production.py --full       # Full 500K ingestion
  python scripts/setup_production.py --quick      # Quick 10K test
  python scripts/setup_production.py --status     # Check current status
  python scripts/setup_production.py --target 50000  # Custom target
        """
    )
    parser.add_argument("--status", action="store_true", help="Show current database status")
    parser.add_argument("--quick", action="store_true", help="Quick ingestion (10K papers)")
    parser.add_argument("--standard", action="store_true", help="Standard ingestion (100K papers)")
    parser.add_argument("--full", action="store_true", help="Full production ingestion (500K papers)")
    parser.add_argument("--target", type=int, help="Custom target paper count")
    
    args = parser.parse_args()
    
    if args.status:
        print_banner()
        stats = get_current_status()
        print_status(stats)
        return
    
    if args.quick:
        print_banner()
        asyncio.run(run_ingestion(10000))
    elif args.standard:
        print_banner()
        asyncio.run(run_ingestion(100000))
    elif args.full:
        print_banner()
        asyncio.run(run_ingestion(500000))
    elif args.target:
        print_banner()
        asyncio.run(run_ingestion(args.target))
    else:
        # Interactive mode
        interactive_setup()


if __name__ == "__main__":
    main()
