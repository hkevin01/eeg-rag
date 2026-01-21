"""
End-to-End RAG Pipeline with Hybrid Retrieval

Demonstrates the complete RAG pipeline:
1. Query Understanding & Planning
2. Hybrid Retrieval (BM25 + Dense + RRF)
3. Context Aggregation
4. Response Generation

This shows how hybrid retrieval improves the entire RAG pipeline.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any

from eeg_rag.agents.local_agent.local_data_agent import LocalDataAgent
from eeg_rag.agents.base_agent import AgentQuery, AgentResult
from eeg_rag.planning.query_planner import QueryPlanner


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("end_to_end_demo")


class SimpleRAGPipeline:
    """Simplified RAG pipeline showcasing hybrid retrieval"""
    
    def __init__(self, use_hybrid: bool = True):
        """
        Initialize RAG pipeline
        
        Args:
            use_hybrid: Use hybrid retrieval (recommended)
        """
        self.use_hybrid = use_hybrid
        
        # Initialize query planner
        self.planner = QueryPlanner(logger=logger)
        
        # Initialize retrieval agent with hybrid mode
        config = {
            "qdrant_url": "http://localhost:6333",
            "qdrant_collection": "eeg_papers",
            "bm25_cache_dir": "data/bm25_cache",
            "top_k": 5,
            "retrieve_k": 20,
            "bm25_weight": 0.5,
            "dense_weight": 0.5,
            "rrf_k": 60,
            "use_query_expansion": True,
            "min_relevance_score": 0.01
        }
        
        self.retrieval_agent = LocalDataAgent(
            config=config,
            use_hybrid_retrieval=use_hybrid
        )
        
        logger.info(f"RAG Pipeline initialized (hybrid={use_hybrid})")
    
    async def process_query(self, query_text: str) -> Dict[str, Any]:
        """
        Process a query through the full RAG pipeline
        
        Args:
            query_text: User query
            
        Returns:
            Complete pipeline results
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing: {query_text}")
        logger.info(f"{'='*80}")
        
        # Step 1: Query Understanding & Planning
        logger.info("Step 1: Query Planning...")
        query_plan = self.planner.plan(query_text)
        
        plan_info = {
            "intent": query_plan.intent.value,
            "complexity": query_plan.complexity.value,
            "sub_queries": [sq.text for sq in query_plan.sub_queries],
            "num_actions": len(query_plan.actions),
            "estimated_latency": query_plan.estimated_latency
        }
        
        logger.info(f"  Intent: {plan_info['intent']}")
        logger.info(f"  Complexity: {plan_info['complexity']}")
        logger.info(f"  Sub-queries: {len(plan_info['sub_queries'])}")
        logger.info(f"  Estimated latency: {plan_info['estimated_latency']:.2f}s")
        
        # Step 2: Hybrid Retrieval
        logger.info("\nStep 2: Hybrid Retrieval...")
        query = AgentQuery(text=query_text, intent=plan_info['intent'])
        retrieval_result = await self.retrieval_agent.execute(query)
        
        if not retrieval_result.success:
            logger.error(f"Retrieval failed: {retrieval_result.error}")
            return {
                "success": False,
                "error": retrieval_result.error
            }
        
        results = retrieval_result.data['results']
        search_time = retrieval_result.data['search_time_ms']
        
        logger.info(f"  Found {len(results)} results in {search_time:.1f}ms")
        logger.info(f"  Source: {retrieval_result.metadata.get('source', 'unknown')}")
        
        # Step 3: Context Aggregation
        logger.info("\nStep 3: Context Aggregation...")
        aggregated_context = self._aggregate_context(results)
        
        logger.info(f"  Total context length: {len(aggregated_context['combined_text'])} chars")
        logger.info(f"  Unique documents: {aggregated_context['num_docs']}")
        logger.info(f"  Average relevance: {aggregated_context['avg_relevance']:.4f}")
        
        # Step 4: Response Generation (simulated)
        logger.info("\nStep 4: Response Generation...")
        response = self._generate_response(query_text, aggregated_context)
        
        logger.info(f"  Generated response: {len(response)} chars")
        
        return {
            "success": True,
            "query": query_text,
            "plan": plan_info,
            "retrieval": {
                "num_results": len(results),
                "search_time_ms": search_time,
                "results": results[:3]  # Top 3 for display
            },
            "context": aggregated_context,
            "response": response,
            "pipeline_stats": {
                "planning_time": 0.05,  # Estimated
                "retrieval_time": search_time / 1000,
                "aggregation_time": 0.01,  # Estimated
                "generation_time": 0.5,  # Estimated
                "total_time": 0.05 + search_time/1000 + 0.01 + 0.5
            }
        }
    
    def _aggregate_context(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate retrieved contexts"""
        if not results:
            return {
                "combined_text": "",
                "num_docs": 0,
                "avg_relevance": 0.0,
                "sources": []
            }
        
        combined_text = "\n\n".join([
            f"[{i+1}] {r['content'][:300]}..."
            for i, r in enumerate(results)
        ])
        
        avg_relevance = sum(r['relevance_score'] for r in results) / len(results)
        
        sources = []
        for r in results:
            citation = r.get('citation', {})
            sources.append({
                "doc_id": r['document_id'],
                "title": citation.get('title', 'N/A'),
                "year": citation.get('year'),
                "relevance": r['relevance_score']
            })
        
        return {
            "combined_text": combined_text,
            "num_docs": len(results),
            "avg_relevance": avg_relevance,
            "sources": sources
        }
    
    def _generate_response(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> str:
        """
        Generate response (simulated - in production would use LLM)
        
        In a real system, this would:
        1. Format context and query into a prompt
        2. Call LLM (GPT-4, Claude, local model)
        3. Parse and validate response
        4. Add citations
        """
        sources = context['sources']
        
        response = f"Based on the retrieved EEG research literature:\n\n"
        
        if sources:
            response += f"I found {len(sources)} relevant papers. "
            response += f"The most relevant is '{sources[0]['title']}' "
            response += f"with a relevance score of {sources[0]['relevance']:.4f}.\n\n"
            
            response += "Key findings:\n"
            for i, source in enumerate(sources[:3], 1):
                response += f"{i}. {source['title']} ({source['year']})\n"
            
            response += f"\n[This is a simulated response. In production, an LLM would "
            response += f"synthesize information from the {len(sources)} retrieved papers.]"
        else:
            response = "No relevant papers found for this query."
        
        return response


async def main():
    """Run end-to-end RAG demo"""
    
    print("=" * 80)
    print("End-to-End RAG Pipeline with Hybrid Retrieval Demo")
    print("=" * 80)
    print()
    
    # Initialize pipeline
    pipeline = SimpleRAGPipeline(use_hybrid=True)
    
    # Test queries
    queries = [
        "What are the best CNN architectures for epileptic seizure detection?",
        "How do motor imagery BCI systems work?",
        "What is the role of P300 in ERP-based spelling systems?"
    ]
    
    results = []
    
    for query in queries:
        result = await pipeline.process_query(query)
        results.append(result)
        
        if result['success']:
            print(f"\n{'='*80}")
            print(f"QUERY: {result['query']}")
            print(f"{'='*80}")
            
            # Show plan
            print(f"\n📋 PLAN:")
            print(f"  Intent: {result['plan']['intent']}")
            print(f"  Complexity: {result['plan']['complexity']}")
            
            # Show retrieval results
            print(f"\n🔍 RETRIEVAL:")
            print(f"  Retrieved: {result['retrieval']['num_results']} papers")
            print(f"  Search time: {result['retrieval']['search_time_ms']:.1f}ms")
            
            print(f"\n  Top Results:")
            for i, res in enumerate(result['retrieval']['results'], 1):
                print(f"    {i}. Doc {res['document_id']}: {res['relevance_score']:.4f}")
                if 'bm25_score' in res['metadata']:
                    print(f"       BM25={res['metadata']['bm25_score']:.3f}, "
                          f"Dense={res['metadata']['dense_score']:.3f}, "
                          f"BM25_rank={res['metadata']['bm25_rank']}, "
                          f"Dense_rank={res['metadata']['dense_rank']}")
            
            # Show context
            print(f"\n📚 CONTEXT:")
            print(f"  Documents: {result['context']['num_docs']}")
            print(f"  Avg relevance: {result['context']['avg_relevance']:.4f}")
            print(f"  Context length: {len(result['context']['combined_text'])} chars")
            
            # Show response
            print(f"\n💡 RESPONSE:")
            print(f"  {result['response']}")
            
            # Show timing
            print(f"\n⏱️  TIMING:")
            stats = result['pipeline_stats']
            print(f"  Planning: {stats['planning_time']*1000:.1f}ms")
            print(f"  Retrieval: {stats['retrieval_time']*1000:.1f}ms")
            print(f"  Aggregation: {stats['aggregation_time']*1000:.1f}ms")
            print(f"  Generation: {stats['generation_time']*1000:.1f}ms")
            print(f"  TOTAL: {stats['total_time']*1000:.1f}ms")
        
        # Delay between queries
        await asyncio.sleep(0.5)
    
    print("\n" + "=" * 80)
    print("Demo Complete!")
    print("=" * 80)
    
    # Summary
    successful = sum(1 for r in results if r['success'])
    avg_retrieval_time = sum(
        r['retrieval']['search_time_ms'] 
        for r in results if r['success']
    ) / successful if successful > 0 else 0
    
    print(f"\nSummary:")
    print(f"  Queries processed: {len(queries)}")
    print(f"  Successful: {successful}")
    print(f"  Avg retrieval time: {avg_retrieval_time:.1f}ms")
    print(f"  Using hybrid retrieval: BM25 + Dense + RRF + Query Expansion")
    print()


if __name__ == "__main__":
    asyncio.run(main())
