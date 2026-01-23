# src/eeg_rag/cli/history_cli.py
"""
CLI commands for search history management.
Uses Click for command handling and Rich for output formatting.
"""

import click
from datetime import datetime
from pathlib import Path
from typing import Optional
import json

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from eeg_rag.services.history_manager import HistoryManager


# Initialize console if rich is available
console = Console() if RICH_AVAILABLE else None


def _print(message: str, style: str = None):
    """Print with optional rich formatting."""
    if RICH_AVAILABLE and console:
        console.print(message, style=style)
    else:
        print(message)


def _get_manager() -> HistoryManager:
    """Get the history manager instance."""
    return HistoryManager.get_instance()


@click.group(name='history')
def history_cli():
    """Search history management commands."""
    pass


@history_cli.command('list')
@click.option('--limit', '-n', default=20, help='Number of searches to show')
@click.option('--starred', '-s', is_flag=True, help='Show only starred searches')
@click.option('--format', '-f', type=click.Choice(['table', 'json']), default='table')
def list_searches(limit: int, starred: bool, format: str):
    """List recent searches."""
    manager = _get_manager()
    searches = manager.get_recent(limit=limit, starred_only=starred, include_results=False)
    
    if format == 'json':
        data = [s.to_dict() for s in searches]
        click.echo(json.dumps(data, indent=2, default=str))
        return
    
    if not searches:
        _print("No search history found.", style="yellow")
        return
    
    if RICH_AVAILABLE:
        table = Table(title="Recent Searches", box=box.ROUNDED)
        table.add_column("ID", style="dim", width=8)
        table.add_column("Query", style="cyan", max_width=50)
        table.add_column("Type", style="green")
        table.add_column("Results", justify="right")
        table.add_column("Time", style="dim")
        table.add_column("⭐", justify="center")
        
        for s in searches:
            star = "⭐" if s.starred else ""
            query_display = s.query_text[:47] + "..." if len(s.query_text) > 50 else s.query_text
            table.add_row(
                s.id[:8],
                query_display,
                s.query_type,
                str(s.result_count),
                s.timestamp.strftime("%Y-%m-%d %H:%M"),
                star
            )
        
        console.print(table)
    else:
        for s in searches:
            star = "⭐" if s.starred else "  "
            print(f"{star} [{s.id[:8]}] {s.timestamp.strftime('%m/%d %H:%M')} - {s.query_text[:40]}... ({s.result_count} results)")


@history_cli.command('show')
@click.argument('query_id')
@click.option('--results', '-r', is_flag=True, help='Show search results')
def show_search(query_id: str, results: bool):
    """Show details of a specific search."""
    manager = _get_manager()
    
    # Try to find by partial ID
    searches = manager.get_recent(limit=100, include_results=True)
    search = None
    for s in searches:
        if s.id.startswith(query_id):
            search = s
            break
    
    if not search:
        _print(f"Search not found: {query_id}", style="red")
        return
    
    if RICH_AVAILABLE:
        # Header panel
        header = Text()
        header.append(f"Query: ", style="bold")
        header.append(f"{search.query_text}\n", style="cyan")
        header.append(f"Type: ", style="bold")
        header.append(f"{search.query_type}\n", style="green")
        header.append(f"Time: ", style="bold")
        header.append(f"{search.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n", style="dim")
        header.append(f"Results: ", style="bold")
        header.append(f"{search.result_count}\n")
        header.append(f"Execution Time: ", style="bold")
        header.append(f"{search.execution_time_ms:.2f}ms\n")
        
        if search.starred:
            header.append("⭐ Starred\n", style="yellow")
        
        if search.notes:
            header.append(f"Notes: ", style="bold")
            header.append(f"{search.notes}\n", style="italic")
        
        console.print(Panel(header, title=f"Search [{search.id[:8]}]"))
        
        if results and search.results:
            table = Table(title="Results", box=box.SIMPLE)
            table.add_column("#", width=3)
            table.add_column("Title", max_width=60)
            table.add_column("Year", width=6)
            table.add_column("Source", width=10)
            table.add_column("Score", width=6)
            
            for i, r in enumerate(search.results[:20], 1):
                title = r.title[:57] + "..." if len(r.title) > 60 else r.title
                table.add_row(
                    str(i),
                    title,
                    str(r.year or "N/A"),
                    r.source,
                    f"{r.relevance_score:.2f}"
                )
            
            console.print(table)
    else:
        print(f"\nSearch [{search.id[:8]}]")
        print(f"  Query: {search.query_text}")
        print(f"  Type: {search.query_type}")
        print(f"  Time: {search.timestamp}")
        print(f"  Results: {search.result_count}")
        if search.starred:
            print("  ⭐ Starred")


@history_cli.command('search')
@click.argument('search_text')
@click.option('--limit', '-n', default=10, help='Maximum results')
def search_history(search_text: str, limit: int):
    """Search through your search history."""
    manager = _get_manager()
    matches = manager.search_in_history(search_text, limit=limit)
    
    if not matches:
        _print(f"No matches found for: {search_text}", style="yellow")
        return
    
    _print(f"\nFound {len(matches)} matches:\n", style="green")
    
    for s in matches:
        star = "⭐" if s.starred else "  "
        _print(f"{star} [{s.id[:8]}] {s.query_text}")


@history_cli.command('star')
@click.argument('query_id')
def toggle_star(query_id: str):
    """Toggle star status for a search."""
    manager = _get_manager()
    
    # Find by partial ID
    searches = manager.get_recent(limit=100, include_results=False)
    for s in searches:
        if s.id.startswith(query_id):
            new_status = manager.toggle_star(s.id)
            status_text = "starred" if new_status else "unstarred"
            _print(f"Search {s.id[:8]} is now {status_text}", style="green")
            return
    
    _print(f"Search not found: {query_id}", style="red")


@history_cli.command('note')
@click.argument('query_id')
@click.argument('note_text')
def add_note(query_id: str, note_text: str):
    """Add a note to a search."""
    manager = _get_manager()
    
    searches = manager.get_recent(limit=100, include_results=False)
    for s in searches:
        if s.id.startswith(query_id):
            manager.add_note(s.id, note_text)
            _print(f"Note added to search {s.id[:8]}", style="green")
            return
    
    _print(f"Search not found: {query_id}", style="red")


@history_cli.command('delete')
@click.argument('query_id')
@click.option('--yes', '-y', is_flag=True, help='Skip confirmation')
def delete_search(query_id: str, yes: bool):
    """Delete a search from history."""
    manager = _get_manager()
    
    searches = manager.get_recent(limit=100, include_results=False)
    for s in searches:
        if s.id.startswith(query_id):
            if not yes:
                if not click.confirm(f"Delete search '{s.query_text[:50]}'?"):
                    _print("Cancelled.", style="yellow")
                    return
            
            manager.delete_search(s.id)
            _print(f"Deleted search {s.id[:8]}", style="green")
            return
    
    _print(f"Search not found: {query_id}", style="red")


@history_cli.command('clear')
@click.option('--days', '-d', default=30, help='Clear searches older than N days')
@click.option('--keep-starred', '-k', is_flag=True, default=True, help='Keep starred searches')
@click.option('--yes', '-y', is_flag=True, help='Skip confirmation')
def clear_history(days: int, keep_starred: bool, yes: bool):
    """Clear old search history."""
    if not yes:
        msg = f"Clear searches older than {days} days"
        if keep_starred:
            msg += " (keeping starred)"
        if not click.confirm(f"{msg}?"):
            _print("Cancelled.", style="yellow")
            return
    
    manager = _get_manager()
    count = manager.clear_old_history(days=days, keep_starred=keep_starred)
    _print(f"Cleared {count} searches", style="green")


@history_cli.command('stats')
def show_stats():
    """Show search history statistics."""
    manager = _get_manager()
    stats = manager.get_stats()
    
    if RICH_AVAILABLE:
        panel_text = Text()
        panel_text.append(f"Total Searches: ", style="bold")
        panel_text.append(f"{stats['total_searches']}\n")
        panel_text.append(f"Starred: ", style="bold")
        panel_text.append(f"{stats['starred_count']}\n")
        panel_text.append(f"Saved Papers: ", style="bold")
        panel_text.append(f"{stats['saved_papers']}\n")
        panel_text.append(f"Avg Results/Search: ", style="bold")
        panel_text.append(f"{stats['avg_results']:.1f}\n")
        
        console.print(Panel(panel_text, title="Search Statistics"))
        
        if stats.get('by_type'):
            table = Table(title="Searches by Type", box=box.SIMPLE)
            table.add_column("Type")
            table.add_column("Count", justify="right")
            
            for qtype, count in stats['by_type'].items():
                table.add_row(qtype, str(count))
            
            console.print(table)
        
        if stats.get('top_queries'):
            table = Table(title="Top Queries", box=box.SIMPLE)
            table.add_column("Query", max_width=50)
            table.add_column("Count", justify="right")
            
            for item in stats['top_queries'][:5]:
                query = item['query'][:47] + "..." if len(item['query']) > 50 else item['query']
                table.add_row(query, str(item['count']))
            
            console.print(table)
    else:
        print(f"\nSearch Statistics")
        print(f"  Total Searches: {stats['total_searches']}")
        print(f"  Starred: {stats['starred_count']}")
        print(f"  Saved Papers: {stats['saved_papers']}")


@history_cli.command('export')
@click.argument('filepath', type=click.Path())
@click.option('--format', '-f', type=click.Choice(['json', 'csv']), default='json')
def export_history(filepath: str, format: str):
    """Export search history to a file."""
    manager = _get_manager()
    output_path = Path(filepath)
    manager.export_history(output_path, format=format)
    _print(f"Exported history to: {output_path}", style="green")


@history_cli.command('papers')
@click.option('--status', '-s', type=click.Choice(['unread', 'reading', 'read', 'archived']))
@click.option('--tag', '-t', help='Filter by tag')
@click.option('--limit', '-n', default=20, help='Number of papers to show')
def list_papers(status: Optional[str], tag: Optional[str], limit: int):
    """List saved papers."""
    manager = _get_manager()
    papers = manager.get_saved_papers(read_status=status, tag=tag, limit=limit)
    
    if not papers:
        _print("No saved papers found.", style="yellow")
        return
    
    if RICH_AVAILABLE:
        table = Table(title="Saved Papers", box=box.ROUNDED)
        table.add_column("ID", style="dim", width=10)
        table.add_column("Title", max_width=50)
        table.add_column("Year", width=6)
        table.add_column("Status", width=10)
        table.add_column("Saved", width=12)
        
        for p in papers:
            title = p['title'][:47] + "..." if len(p['title']) > 50 else p['title']
            saved_date = p['saved_at'][:10] if p['saved_at'] else ""
            table.add_row(
                p['paper_id'][:10],
                title,
                str(p['year'] or "N/A"),
                p['read_status'],
                saved_date
            )
        
        console.print(table)
    else:
        for p in papers:
            print(f"[{p['paper_id'][:10]}] {p['title'][:50]}... ({p['read_status']})")


# Main entry point for standalone use
def main():
    """Main entry point."""
    history_cli()


if __name__ == '__main__':
    main()
