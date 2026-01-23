"""
Demo: Enhanced Multi-Agent Orchestrator

Shows how the orchestrator coordinates multiple agents for literature search.
"""

import asyncio
import logging
from eeg_rag.orchestrator import Orchestrator, QueryType

logging.basicConfig(level=logging.INFO)


async def demo_query_analysis():
    """Demonstrate query analysis and planning."""
    print("\n" + "="*60)
    print("QUERY ANALYSIS DEMO")
    print("="*60)
    
    orchestrator = Orchestrator()
    
    test_queries = [
        "EEG seizure detection deep learning",
        "Compare CNN vs LSTM for EEG classification",
        "Recent advances in EEG analysis 2023",
        "Papers by Yann LeCun on neural networks",
        "Influential EEG research papers",
    ]
    
    for query in test_queries:
        plan = orchestrator.planner.create_plan(query, max_results=20)
        print(f"\nQuery: {query}")
        print(f"  Type: {plan.query_type.value}")
        print(f"  Strategy: {plan.strategy.value}")
        print(f"  Agents: {[aid for aid, _ in plan.agent_tasks]}")
        print(f"  Estimated time: {plan.estimated_time_ms}ms")


async def demo_mock_search():
    """Demonstrate orchestrator search with mock agents."""
    print("\n" + "="*60)
    print("MOCK SEARCH DEMO")
    print("="*60)
    
    # Create orchestrator with minimal config
    orchestrator = Orchestrator(config={
        "email": "demo@example.com"
    })
    
    query = "deep learning EEG classification"
    print(f"\nQuery: {query}")
    
    # Progress callback
    def on_progress(stage: str, percent: float):
        bar_length = 40
        filled = int(bar_length * percent)
        bar = "█" * filled + "░" * (bar_length - filled)
        print(f"\r[{bar}] {stage:20s} {percent*100:5.1f}%", end="", flush=True)
    
    try:
        result = await orchestrator.search(
            query=query,
            max_results=10,
            sources=["pubmed"],  # Only use PubMed for demo
            synthesize=False,  # Skip synthesis for speed
            progress_callback=on_progress
        )
        
        print("\n")  # New line after progress bar
        print(f"\nResult:")
        print(f"  Success: {result.success}")
        print(f"  Papers found: {result.total_found}")
        print(f"  Sources used: {result.sources_used}")
        print(f"  Execution time: {result.execution_time_ms:.1f}ms")
        print(f"  Errors: {result.errors if result.errors else 'None'}")
        
        if result.papers:
            print(f"\n  Sample papers:")
            for i, paper in enumerate(result.papers[:3], 1):
                print(f"    {i}. {paper.get('title', 'N/A')[:80]}...")
                print(f"       Year: {paper.get('year')}, Citations: {paper.get('citation_count')}")
        
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        await orchestrator.close()


async def demo_cascading_strategy():
    """Demonstrate cascading execution strategy."""
    print("\n" + "="*60)
    print("CASCADING STRATEGY DEMO")
    print("="*60)
    
    orchestrator = Orchestrator()
    
    # Query that triggers cascading (exploratory)
    query = "EEG signal processing techniques"
    
    plan = orchestrator.planner.create_plan(query)
    print(f"\nQuery: {query}")
    print(f"Strategy: {plan.strategy.value}")
    print("\nExecution flow:")
    print("  1. Search local database first")
    print("  2. If insufficient results, expand to external sources")
    print("  3. Merge and deduplicate results")
    print("  4. Synthesize findings")
    
    await orchestrator.close()


async def demo_parallel_strategy():
    """Demonstrate parallel execution strategy."""
    print("\n" + "="*60)
    print("PARALLEL STRATEGY DEMO")
    print("="*60)
    
    orchestrator = Orchestrator()
    
    # Query that triggers parallel (comparative)
    query = "Compare transformer vs CNN for EEG analysis"
    
    plan = orchestrator.planner.create_plan(query)
    print(f"\nQuery: {query}")
    print(f"Strategy: {plan.strategy.value}")
    print("\nExecution flow:")
    print("  1. Search all sources simultaneously")
    print("  2. Wait for all agents to complete")
    print("  3. Merge results from all sources")
    print("  4. Comparative synthesis")
    
    await orchestrator.close()


async def main():
    """Run all demos."""
    demos = [
        ("Query Analysis", demo_query_analysis),
        ("Mock Search", demo_mock_search),
        ("Cascading Strategy", demo_cascading_strategy),
        ("Parallel Strategy", demo_parallel_strategy),
    ]
    
    for name, demo_func in demos:
        try:
            await demo_func()
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
        
        # Pause between demos
        await asyncio.sleep(1)
    
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
