#!/usr/bin/env python3
"""
Additional CLI Commands for EEG-RAG

Provides extended functionality:
- Graph population and visualization
- Load testing and benchmarking
- System health monitoring
- Data export and import
"""

import asyncio
import click
import json
from pathlib import Path
from typing import Optional, List
import logging

# Import EEG-RAG components
from ..knowledge_graph.graph_populator import GraphPopulator, populate_graph_from_corpus
from ..knowledge_graph.graph_visualizer import GraphVisualizer
from ..knowledge_graph.graph_interface import Neo4jInterface
from ..knowledge_graph.web_interface import GraphWebInterface
from ..evaluation.load_testing import LoadTester, LoadTestConfig
from ..evaluation.benchmarking import EEGRAGBenchmark
from ..agents.orchestrator.orchestrator_agent import OrchestratorAgent
from ..memory.memory_manager import MemoryManager
from ..monitoring.production_monitor import ProductionMonitor

logger = logging.getLogger(__name__)


@click.group()
def graph():
    """Knowledge graph operations."""
    pass


@graph.command()
@click.option('--corpus-path', required=True, type=click.Path(exists=True),
              help='Path to corpus JSONL file')
@click.option('--neo4j-uri', default='bolt://localhost:7687',
              help='Neo4j database URI')
@click.option('--neo4j-user', default='neo4j',
              help='Neo4j username')
@click.option('--neo4j-password', prompt=True, hide_input=True,
              help='Neo4j password')
@click.option('--limit', type=int, help='Limit number of papers to process')
@click.option('--batch-size', default=100, help='Batch size for processing')
async def populate(corpus_path, neo4j_uri, neo4j_user, neo4j_password, limit, batch_size):
    """Populate knowledge graph from corpus data."""
    try:
        corpus_path = Path(corpus_path)
        
        click.echo(f"Populating graph from {corpus_path}...")
        
        stats = await populate_graph_from_corpus(
            corpus_path=corpus_path,
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            limit=limit,
            batch_size=batch_size
        )
        
        click.echo(f"\nPopulation completed:")
        click.echo(f"  Papers processed: {stats.papers_processed}")
        click.echo(f"  Nodes created: {stats.nodes_created}")
        click.echo(f"  Relationships created: {stats.relationships_created}")
        click.echo(f"  Entities extracted: {stats.entities_extracted}")
        click.echo(f"  Citations resolved: {stats.citations_resolved}")
        click.echo(f"  Processing time: {stats.processing_time_ms:.1f}ms")
        
        if stats.errors > 0:
            click.echo(f"  Errors: {stats.errors}", err=True)
        
    except Exception as e:
        click.echo(f"Graph population failed: {str(e)}", err=True)
        raise click.Abort()


@graph.command()
@click.option('--query-terms', required=True,
              help='Comma-separated list of query terms')
@click.option('--neo4j-uri', default='bolt://localhost:7687',
              help='Neo4j database URI')
@click.option('--neo4j-user', default='neo4j',
              help='Neo4j username')
@click.option('--neo4j-password', prompt=True, hide_input=True,
              help='Neo4j password')
@click.option('--max-nodes', default=100, help='Maximum nodes to include')
@click.option('--max-depth', default=2, help='Maximum relationship depth')
@click.option('--output', type=click.Path(), help='Output file for visualization data')
async def visualize(query_terms, neo4j_uri, neo4j_user, neo4j_password, 
                   max_nodes, max_depth, output):
    """Create network visualization for query terms."""
    try:
        terms = [term.strip() for term in query_terms.split(',')]
        
        neo4j_interface = Neo4jInterface(neo4j_uri, neo4j_user, neo4j_password)
        
        async with neo4j_interface:
            visualizer = GraphVisualizer(neo4j_interface)
            
            click.echo(f"Creating visualization for: {', '.join(terms)}")
            
            vis_data = await visualizer.create_research_network_visualization(
                query_terms=terms,
                max_nodes=max_nodes,
                max_depth=max_depth
            )
            
            click.echo(f"Generated visualization with {len(vis_data.nodes)} nodes and {len(vis_data.edges)} edges")
            
            if output:
                output_path = Path(output)
                visualizer.export_to_json(vis_data, output_path)
                click.echo(f"Visualization exported to {output_path}")
            else:
                # Print summary
                click.echo("\nVisualization Summary:")
                click.echo(f"  Title: {vis_data.metadata.get('title', 'N/A')}")
                click.echo(f"  Nodes: {len(vis_data.nodes)}")
                click.echo(f"  Edges: {len(vis_data.edges)}")
                
                # Show node type distribution
                node_types = {}
                for node in vis_data.nodes:
                    node_types[node.node_type] = node_types.get(node.node_type, 0) + 1
                
                click.echo("  Node types:")
                for node_type, count in sorted(node_types.items()):
                    click.echo(f"    {node_type}: {count}")
        
    except Exception as e:
        click.echo(f"Visualization failed: {str(e)}", err=True)
        raise click.Abort()


@graph.command()
@click.option('--neo4j-uri', default='bolt://localhost:7687',
              help='Neo4j database URI')
@click.option('--neo4j-user', default='neo4j',
              help='Neo4j username')
@click.option('--neo4j-password', prompt=True, hide_input=True,
              help='Neo4j password')
@click.option('--host', default='localhost', help='Web server host')
@click.option('--port', default=8080, help='Web server port')
async def web(neo4j_uri, neo4j_user, neo4j_password, host, port):
    """Start web interface for graph visualization."""
    try:
        neo4j_interface = Neo4jInterface(neo4j_uri, neo4j_user, neo4j_password)
        
        async with neo4j_interface:
            web_interface = GraphWebInterface(neo4j_interface)
            
            click.echo(f"Starting web interface at http://{host}:{port}")
            click.echo("Press Ctrl+C to stop...")
            
            await web_interface.start_server(host=host, port=port)
        
    except KeyboardInterrupt:
        click.echo("\nWeb interface stopped.")
    except Exception as e:
        click.echo(f"Web interface failed: {str(e)}", err=True)
        raise click.Abort()


@click.group()
def test():
    """Testing and evaluation commands."""
    pass


@test.command()
@click.option('--concurrent-users', default=10, help='Number of concurrent users')
@click.option('--total-requests', default=100, help='Total number of requests')
@click.option('--ramp-up-time', default=30.0, help='Ramp-up time in seconds')
@click.option('--output-dir', type=click.Path(), help='Output directory for results')
@click.option('--queries-file', type=click.Path(exists=True),
              help='File containing test queries (one per line)')
async def load_test(concurrent_users, total_requests, ramp_up_time, output_dir, queries_file):
    """Run load testing against the EEG-RAG system."""
    try:
        # Initialize orchestrator (simplified)
        memory_manager = MemoryManager()
        orchestrator = OrchestratorAgent(memory_manager=memory_manager)
        
        # Load custom queries if provided
        test_queries = None
        if queries_file:
            with open(queries_file, 'r') as f:
                test_queries = [line.strip() for line in f if line.strip()]
        
        # Create load tester
        load_tester = LoadTester(orchestrator, test_queries)
        
        # Configure test
        config = LoadTestConfig(
            concurrent_users=concurrent_users,
            total_requests=total_requests,
            ramp_up_time=ramp_up_time
        )
        
        click.echo(f"Starting load test with {concurrent_users} users, {total_requests} requests...")
        
        # Run test with progress updates
        def progress_callback(progress):
            percentage = progress * 100
            click.echo(f"Progress: {percentage:.1f}%")
        
        results = await load_tester.run_load_test(config, progress_callback)
        
        # Display results
        click.echo("\nLoad Test Results:")
        click.echo(f"  Success rate: {(1 - results.error_rate) * 100:.1f}%")
        click.echo(f"  Average response time: {results.avg_response_time_ms:.1f}ms")
        click.echo(f"  P95 response time: {results.p95_response_time_ms:.1f}ms")
        click.echo(f"  Queries per second: {results.queries_per_second:.2f}")
        click.echo(f"  Performance score: {results.performance_score:.1f}/100")
        click.echo(f"  Passed thresholds: {'✓' if results.passed_thresholds else '✗'}")
        
        # Export results if output directory provided
        if output_dir:
            output_path = Path(output_dir) / f"load_test_results_{int(results.start_time)}.json"
            output_path.parent.mkdir(exist_ok=True)
            load_tester.export_results(results, output_path)
            click.echo(f"\nResults exported to {output_path}")
        
    except Exception as e:
        click.echo(f"Load test failed: {str(e)}", err=True)
        raise click.Abort()


@test.command()
@click.option('--output-dir', type=click.Path(), help='Output directory for results')
async def benchmark(output_dir):
    """Run comprehensive benchmarking suite."""
    try:
        # Initialize components
        memory_manager = MemoryManager()
        orchestrator = OrchestratorAgent(memory_manager=memory_manager)
        
        # Create benchmark
        benchmark = EEGRAGBenchmark(orchestrator)
        
        click.echo("Starting comprehensive benchmark suite...")
        
        # Run benchmarks
        results = await benchmark.run_full_benchmark()
        
        # Display results
        click.echo("\nBenchmark Results:")
        click.echo(f"  Overall score: {results.overall_score:.1f}/100")
        click.echo(f"  Retrieval score: {results.retrieval_score:.1f}/100")
        click.echo(f"  Generation score: {results.generation_score:.1f}/100")
        click.echo(f"  Average retrieval time: {results.avg_retrieval_time_ms:.1f}ms")
        click.echo(f"  Average generation time: {results.avg_generation_time_ms:.1f}ms")
        click.echo(f"  Average citation accuracy: {results.avg_citation_accuracy:.3f}")
        click.echo(f"  Average response quality: {results.avg_response_quality:.3f}")
        
        # Export results if output directory provided
        if output_dir:
            output_path = Path(output_dir) / f"benchmark_results_{int(asyncio.get_event_loop().time())}.json"
            output_path.parent.mkdir(exist_ok=True)
            benchmark.export_benchmark_results(results, output_path)
            click.echo(f"\nResults exported to {output_path}")
        
    except Exception as e:
        click.echo(f"Benchmark failed: {str(e)}", err=True)
        raise click.Abort()


@click.group()
def monitor():
    """System monitoring commands."""
    pass


@monitor.command()
@click.option('--duration', default=60, help='Monitoring duration in seconds')
@click.option('--interval', default=5, help='Monitoring interval in seconds')
async def health(duration, interval):
    """Monitor system health and performance."""
    try:
        # Initialize production monitor
        monitor = ProductionMonitor()
        
        click.echo(f"Monitoring system health for {duration} seconds...")
        click.echo("Press Ctrl+C to stop early.\n")
        
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < duration:
            try:
                # Get current metrics
                metrics = await monitor.get_current_metrics()
                
                # Display key metrics
                click.echo(f"Time: {asyncio.get_event_loop().time() - start_time:.1f}s")
                click.echo(f"  Memory: {metrics.get('memory_usage_mb', 0):.1f} MB")
                click.echo(f"  CPU: {metrics.get('cpu_usage_percent', 0):.1f}%")
                click.echo(f"  Active queries: {metrics.get('active_queries', 0)}")
                click.echo(f"  Cache hits: {metrics.get('cache_hit_rate', 0):.3f}")
                click.echo(f"  Response time P95: {metrics.get('response_time_p95_ms', 0):.1f}ms")
                click.echo("---")
                
                await asyncio.sleep(interval)
                
            except KeyboardInterrupt:
                break
        
        click.echo("\nMonitoring completed.")
        
    except Exception as e:
        click.echo(f"Monitoring failed: {str(e)}", err=True)
        raise click.Abort()


@monitor.command()
async def status():
    """Check current system status."""
    try:
        # Initialize production monitor
        monitor = ProductionMonitor()
        
        # Get system status
        status = await monitor.get_system_health()
        
        click.echo("System Status:")
        click.echo(f"  Overall health: {'✓ Healthy' if status.get('healthy', False) else '✗ Unhealthy'}")
        
        components = status.get('components', {})
        for component, health in components.items():
            icon = '✓' if health.get('healthy', False) else '✗'
            click.echo(f"  {component}: {icon} {health.get('status', 'Unknown')}")
        
        # Show resource usage
        resources = status.get('resources', {})
        if resources:
            click.echo("\nResource Usage:")
            click.echo(f"  Memory: {resources.get('memory_usage_mb', 0):.1f} MB")
            click.echo(f"  CPU: {resources.get('cpu_usage_percent', 0):.1f}%")
            click.echo(f"  Disk: {resources.get('disk_usage_percent', 0):.1f}%")
        
        # Show recent metrics
        recent_metrics = status.get('recent_metrics', {})
        if recent_metrics:
            click.echo("\nRecent Performance:")
            click.echo(f"  Queries processed: {recent_metrics.get('queries_processed', 0)}")
            click.echo(f"  Average response time: {recent_metrics.get('avg_response_time_ms', 0):.1f}ms")
            click.echo(f"  Error rate: {recent_metrics.get('error_rate', 0):.3f}")
        
    except Exception as e:
        click.echo(f"Status check failed: {str(e)}", err=True)
        raise click.Abort()


@click.group()
def data():
    """Data import/export operations."""
    pass


@data.command()
@click.option('--corpus-path', required=True, type=click.Path(exists=True),
              help='Path to corpus file to validate')
async def validate(corpus_path):
    """Validate corpus data format and integrity."""
    try:
        corpus_path = Path(corpus_path)
        
        click.echo(f"Validating corpus: {corpus_path}")
        
        # Simple validation
        valid_papers = 0
        invalid_papers = 0
        total_size_mb = corpus_path.stat().st_size / 1024 / 1024
        
        required_fields = ['pmid', 'title', 'abstract']
        
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    
                    # Check required fields
                    missing_fields = [field for field in required_fields if field not in data]
                    
                    if missing_fields:
                        if invalid_papers < 5:  # Show first few errors
                            click.echo(f"  Line {line_num}: Missing fields: {missing_fields}")
                        invalid_papers += 1
                    else:
                        valid_papers += 1
                
                except json.JSONDecodeError as e:
                    if invalid_papers < 5:
                        click.echo(f"  Line {line_num}: JSON error: {str(e)}")
                    invalid_papers += 1
        
        total_papers = valid_papers + invalid_papers
        
        click.echo(f"\nValidation Results:")
        click.echo(f"  File size: {total_size_mb:.1f} MB")
        click.echo(f"  Total papers: {total_papers}")
        click.echo(f"  Valid papers: {valid_papers} ({valid_papers/total_papers*100:.1f}%)")
        click.echo(f"  Invalid papers: {invalid_papers} ({invalid_papers/total_papers*100:.1f}%)")
        
        if invalid_papers > 5:
            click.echo(f"  (Showing first 5 errors, {invalid_papers-5} more errors found)")
        
        if invalid_papers == 0:
            click.echo("  ✓ Corpus is valid")
        elif invalid_papers / total_papers < 0.1:
            click.echo("  ⚠ Corpus has minor issues")
        else:
            click.echo("  ✗ Corpus has significant issues")
        
    except Exception as e:
        click.echo(f"Validation failed: {str(e)}", err=True)
        raise click.Abort()


# Add commands to main CLI
def add_extended_commands(cli_app):
    """Add extended commands to main CLI application."""
    cli_app.add_command(graph)
    cli_app.add_command(test)
    cli_app.add_command(monitor)
    cli_app.add_command(data)