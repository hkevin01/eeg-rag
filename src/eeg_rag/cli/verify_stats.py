# eeg_rag/cli/verify_stats.py
"""
CLI tool to verify and diagnose statistics issues.
"""

import click
from pathlib import Path
import json

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from eeg_rag.services.stats_service import StatsService, get_stats_service

console = Console() if RICH_AVAILABLE else None


def _print(msg: str, style: str = None):
    if RICH_AVAILABLE and console:
        console.print(msg, style=style)
    else:
        print(msg)


@click.group()
def stats():
    """Statistics verification and management."""
    pass


@stats.command()
@click.option('--db-path', type=click.Path(), help='Path to papers database')
@click.option('--json', 'as_json', is_flag=True, help='Output as JSON')
def verify(db_path: str, as_json: bool):
    """Verify database counts and detect issues."""
    
    service = StatsService(
        papers_db_path=Path(db_path) if db_path else None
    )
    
    _print("[bold]Verifying database statistics...[/bold]\n")
    
    report = service.verify_counts()
    
    if as_json:
        click.echo(json.dumps(report, indent=2))
        return
    
    if RICH_AVAILABLE:
        # Display results
        console.print(Panel(
            f"[bold green]Verified Total: {report.get('verified_total', 'Unknown')}[/bold green]\n"
            f"Display Format: {report.get('display_total', 'N/A')}",
            title="Paper Count Verification"
        ))
        
        # Tables found
        if 'tables_found' in report:
            console.print(f"\n[bold]Tables found:[/bold] {', '.join(report['tables_found'])}")
        
        # Counts per table
        if report['counts']:
            table = Table(title="Row Counts by Table", box=box.ROUNDED)
            table.add_column("Table", style="cyan")
            table.add_column("Count", justify="right")
            
            for tbl, count in report['counts'].items():
                table.add_row(tbl, str(count))
            
            console.print(table)
        
        # Inconsistencies
        if report['inconsistencies']:
            console.print("\n[bold red]⚠ Issues Found:[/bold red]")
            for issue in report['inconsistencies']:
                console.print(f"  • {issue}")
        else:
            console.print("\n[bold green]✓ No inconsistencies detected[/bold green]")
        
        # Recommendations
        if report['recommendations']:
            console.print("\n[bold yellow]Recommendations:[/bold yellow]")
            for rec in report['recommendations']:
                console.print(f"  → {rec}")
    else:
        print(f"\nVerified Total: {report.get('verified_total', 'Unknown')}")
        print(f"Display Format: {report.get('display_total', 'N/A')}")
        if report['inconsistencies']:
            print("\nIssues Found:")
            for issue in report['inconsistencies']:
                print(f"  - {issue}")


@stats.command()
@click.option('--no-cache', is_flag=True, help='Bypass cache for fresh data')
def show(no_cache: bool):
    """Show current statistics."""
    
    service = get_stats_service()
    
    if no_cache:
        service.invalidate_cache()
    
    stats_data = service.get_full_stats(use_cache=not no_cache)
    display = service.get_display_stats()
    
    if RICH_AVAILABLE:
        # Header stats (what would show on homepage)
        console.print(Panel(
            f"[bold cyan]{display['papers_indexed']}[/bold cyan] Papers Indexed\n"
            f"[bold cyan]{display['ai_agents']}[/bold cyan] AI Agents\n"
            f"[bold cyan]{display['citation_accuracy']}[/bold cyan] Citation Accuracy",
            title="Homepage Display Stats"
        ))
        
        # Detailed breakdown
        console.print("\n[bold]Detailed Statistics:[/bold]")
        console.print(f"  Total Papers: {stats_data.total_papers:,}")
        console.print(f"  With Abstracts: {stats_data.papers_with_abstracts:,}")
        console.print(f"  With Embeddings: {stats_data.papers_with_embeddings:,}")
        console.print(f"  Year Range: {stats_data.date_range['min_year']} - {stats_data.date_range['max_year']}")
        
        # By source
        if stats_data.papers_by_source:
            console.print("\n[bold]Papers by Source:[/bold]")
            table = Table(box=box.SIMPLE)
            table.add_column("Source")
            table.add_column("Count", justify="right")
            table.add_column("Percentage", justify="right")
            
            for source, count in sorted(stats_data.papers_by_source.items(), key=lambda x: -x[1]):
                pct = (count / stats_data.total_papers * 100) if stats_data.total_papers > 0 else 0
                table.add_row(source, f"{count:,}", f"{pct:.1f}%")
            
            console.print(table)
        
        # Health status
        health = stats_data.index_health
        status_color = "green" if health['status'] == 'healthy' else "yellow"
        console.print(f"\n[bold]Index Health:[/bold] [{status_color}]{health['status']}[/{status_color}]")
        
        if health['issues']:
            for issue in health['issues']:
                console.print(f"  ⚠ {issue}")
        
        console.print(f"\n[dim]Last updated: {display['last_updated']}[/dim]")
    else:
        print(f"\nHomepage Display Stats:")
        print(f"  Papers Indexed: {display['papers_indexed']}")
        print(f"  AI Agents: {display['ai_agents']}")
        print(f"  Citation Accuracy: {display['citation_accuracy']}")
        print(f"\nDetailed Statistics:")
        print(f"  Total Papers: {stats_data.total_papers:,}")


@stats.command()
def refresh():
    """Refresh cached statistics."""
    service = get_stats_service()
    service.invalidate_cache()
    
    _print("[bold]Refreshing statistics...[/bold]")
    stats_data = service.get_full_stats(use_cache=False)
    
    _print(f"[green]✓ Cache refreshed[/green]")
    _print(f"  Total papers: {stats_data.total_papers:,}")


@stats.command()
@click.option('--fix-duplicates', is_flag=True, help='Remove duplicate papers')
@click.option('--fix-null-sources', is_flag=True, help='Set default source for NULL entries')
@click.option('--dry-run', is_flag=True, help='Show what would be fixed without making changes')
def fix(fix_duplicates: bool, fix_null_sources: bool, dry_run: bool):
    """Fix common database issues."""
    
    service = get_stats_service()
    
    if dry_run:
        _print("[yellow]DRY RUN - No changes will be made[/yellow]\n")
    
    # First verify to see issues
    report = service.verify_counts()
    
    if not report['inconsistencies']:
        _print("[green]No issues to fix![/green]")
        return
    
    _print(f"[bold]Found {len(report['inconsistencies'])} issues:[/bold]")
    for issue in report['inconsistencies']:
        _print(f"  • {issue}")
    
    if fix_duplicates and 'duplicate_ids' in report:
        _print(f"\n[bold]Fixing duplicates...[/bold]")
        if not dry_run:
            _print("[yellow]Duplicate removal not yet implemented[/yellow]")
        else:
            _print(f"  Would remove duplicates for {len(report['duplicate_ids'])} paper_ids")
    
    if fix_null_sources:
        _print(f"\n[bold]Fixing NULL sources...[/bold]")
        if not dry_run:
            _print("[yellow]Source fix not yet implemented[/yellow]")
        else:
            _print("  Would set source='unknown' for NULL entries")


def main():
    """Main entry point."""
    stats()


if __name__ == "__main__":
    main()
