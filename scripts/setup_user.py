#!/usr/bin/env python3
# scripts/setup_user.py
"""
First-run setup for new EEG-RAG users.

This script:
1. Extracts the compressed metadata index (if needed)
2. Initializes the local cache directory
3. Tests API connectivity
4. Shows usage instructions

Users run this once after cloning the repository.

Usage:
    python scripts/setup_user.py
    python scripts/setup_user.py --status  # Check setup status
"""

import argparse
import gzip
import shutil
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def print_banner():
    """Print welcome banner."""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   🧠 EEG-RAG: EEG Research Assistant                          ║
║                                                               ║
║   Retrieval-Augmented Generation for                         ║
║   Electroencephalography Research                             ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
    """)


def check_metadata_index() -> tuple[bool, Path]:
    """Check if metadata index exists."""
    data_dir = Path(__file__).parent.parent / "data" / "metadata"
    index_db = data_dir / "index.db"
    index_gz = data_dir / "index.db.gz"
    
    if index_db.exists():
        return True, index_db
    elif index_gz.exists():
        return False, index_gz
    else:
        return False, data_dir


def extract_metadata_index(gz_path: Path) -> Path:
    """Extract compressed metadata index."""
    db_path = gz_path.with_suffix('')
    
    print(f"📦 Extracting metadata index...")
    print(f"   From: {gz_path}")
    print(f"   To: {db_path}")
    
    with gzip.open(gz_path, 'rb') as f_in:
        with open(db_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    size_mb = db_path.stat().st_size / 1024 / 1024
    print(f"   ✓ Extracted {size_mb:.1f} MB")
    
    return db_path


def init_cache_dir() -> Path:
    """Initialize user cache directory."""
    cache_dir = Path.home() / ".eeg_rag" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Cache directory initialized: {cache_dir}")
    
    return cache_dir


def test_pubmed_connection() -> bool:
    """Test connection to PubMed API."""
    import urllib.request
    
    print("🌐 Testing PubMed connection...")
    
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/einfo.fcgi?db=pubmed"
    
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            if resp.status == 200:
                print("   ✓ PubMed API accessible")
                return True
    except Exception as e:
        print(f"   ✗ PubMed API error: {e}")
    
    return False


def test_openalex_connection() -> bool:
    """Test connection to OpenAlex API."""
    import urllib.request
    import json
    
    print("🌐 Testing OpenAlex connection...")
    
    url = "https://api.openalex.org/works?sample=1&per_page=1"
    
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode('utf-8'))
                if data.get("results"):
                    print("   ✓ OpenAlex API accessible")
                    return True
    except Exception as e:
        print(f"   ✗ OpenAlex API error: {e}")
    
    return False


def show_status():
    """Show current setup status."""
    print_banner()
    print("📊 Setup Status\n")
    
    # Check metadata index
    exists, path = check_metadata_index()
    if exists:
        size_mb = path.stat().st_size / 1024 / 1024
        print(f"✓ Metadata index: {path}")
        print(f"  Size: {size_mb:.1f} MB")
        
        # Get stats
        try:
            from eeg_rag.db.metadata_index import MetadataIndex
            idx = MetadataIndex(db_path=path)
            stats = idx.get_stats()
            print(f"  Papers: {stats['total_papers']:,}")
            print(f"  With PMID: {stats['with_pmid']:,}")
            print(f"  Year range: {stats['year_range']['min']} - {stats['year_range']['max']}")
        except Exception as e:
            print(f"  (Could not read stats: {e})")
    elif path.suffix == '.gz':
        print(f"○ Metadata index: compressed (run setup to extract)")
        print(f"  {path}")
    else:
        print("✗ Metadata index: not found")
        print("  Run 'python scripts/build_metadata_index.py --quick' to create")
    
    print()
    
    # Check cache
    cache_dir = Path.home() / ".eeg_rag" / "cache"
    cache_db = cache_dir / "papers.db"
    
    if cache_db.exists():
        size_mb = cache_db.stat().st_size / 1024 / 1024
        
        import sqlite3
        conn = sqlite3.connect(str(cache_db))
        cursor = conn.execute("SELECT COUNT(*) FROM papers")
        count = cursor.fetchone()[0]
        conn.close()
        
        print(f"✓ Local cache: {cache_db}")
        print(f"  Cached papers: {count:,}")
        print(f"  Size: {size_mb:.2f} MB")
    else:
        print(f"○ Local cache: not initialized")
        print(f"  Will be created at: {cache_dir}")
    
    print()
    
    # Test connections
    pubmed_ok = test_pubmed_connection()
    openalex_ok = test_openalex_connection()
    
    print()
    if pubmed_ok and openalex_ok:
        print("✅ All systems ready!")
    else:
        print("⚠️  Some connections failed (may work with retry)")


def setup():
    """Run first-time setup."""
    print_banner()
    print("🚀 First-Time Setup\n")
    
    # Step 1: Check/extract metadata index
    exists, path = check_metadata_index()
    
    if exists:
        size_mb = path.stat().st_size / 1024 / 1024
        print(f"✓ Metadata index already exists: {size_mb:.1f} MB")
    elif path.suffix == '.gz':
        extract_metadata_index(path)
    else:
        print("⚠️  No metadata index found!")
        print("   The repository should include data/metadata/index.db.gz")
        print("   You can build one with: python scripts/build_metadata_index.py --quick")
        print()
    
    # Step 2: Initialize cache
    print()
    cache_dir = init_cache_dir()
    
    # Step 3: Test connections
    print()
    pubmed_ok = test_pubmed_connection()
    openalex_ok = test_openalex_connection()
    
    # Done
    print()
    print("=" * 60)
    print()
    print("✅ Setup Complete!")
    print()
    print("How it works:")
    print("  • The metadata index contains 500K+ paper references")
    print("  • When you search, matching paper IDs are found instantly")
    print("  • Full abstracts are fetched on-demand from PubMed/OpenAlex")
    print("  • Fetched papers are cached locally for future use")
    print()
    print("Quick start:")
    print("  streamlit run src/eeg_rag/web_ui/app_enhanced.py")
    print()
    print("Or programmatically:")
    print("  from eeg_rag.db.metadata_index import MetadataIndex")
    print("  from eeg_rag.db.paper_resolver import PaperResolver")
    print()
    print(f"Local cache: {cache_dir}")
    print()


def main():
    parser = argparse.ArgumentParser(description="EEG-RAG User Setup")
    parser.add_argument("--status", action="store_true",
                       help="Show setup status")
    
    args = parser.parse_args()
    
    if args.status:
        show_status()
    else:
        setup()


if __name__ == "__main__":
    main()
