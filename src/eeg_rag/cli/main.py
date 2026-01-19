#!/usr/bin/env python3
"""
EEG-RAG CLI Application

Interactive command-line interface for the EEG-RAG system.
Allows users to query the EEG research corpus and get AI-powered answers.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eeg_rag.agents.base_agent import AgentQuery, AgentRegistry
from eeg_rag.agents.local_agent.local_data_agent import LocalDataAgent
from eeg_rag.agents.orchestrator.orchestrator_agent import OrchestratorAgent  
from eeg_rag.memory.memory_manager import MemoryManager
from eeg_rag.monitoring import PerformanceMonitor, monitor_performance
from eeg_rag.utils.common_utils import check_system_health, SystemStatus


class EEGRAGCLIApp:
    """
    Interactive CLI application for EEG-RAG system
    """
    
    def __init__(self):
        """Initialize CLI application"""
        self.setup_logging()
        
        # Create app directory
        app_dir = Path.home() / ".eeg-rag"
        app_dir.mkdir(exist_ok=True)
        
        # Initialize memory manager with correct parameters
        self.memory_manager = MemoryManager(
            db_path=app_dir / "memory.db",
            short_term_max_entries=100,
            short_term_ttl_hours=2.0
        )
        
        # Initialize performance monitor
        self.performance_monitor = PerformanceMonitor(
            max_metrics_history=500,
            performance_threshold_ms=2000.0,  # 2 second threshold for CLI operations
            memory_threshold_mb=200.0
        )
        
        # Initialize basic agent for demonstration
        # In full system, this would be the orchestrator
        self.agent = None
        self.session_id = "cli_session"
        
        print("🧠 EEG-RAG: AI-Powered EEG Research Assistant")
        print("=" * 50)
        
    def setup_logging(self):
        """Configure logging for CLI"""
        logging.basicConfig(
            level=logging.WARNING,  # Reduce noise in CLI
            format="%(levelname)s: %(message)s"
        )
        self.logger = logging.getLogger("eeg_rag_cli")
        
    async def initialize_system(self) -> bool:
        """
        Initialize the EEG-RAG system
        
        Returns:
            True if initialization successful, False otherwise
        """
        print("🔧 Initializing EEG-RAG system...")
        
        # Check system health
        health = check_system_health()
        print(f"📊 System Health: {health.status.value}")
        
        if health.status == SystemStatus.CRITICAL:
            print("❌ System resources are critically low!")
            for warning in health.warnings:
                print(f"   ⚠️  {warning}")
            print("Consider closing other applications before continuing.")
            
            response = input("Continue anyway? (y/N): ").lower()
            if response != 'y':
                return False
        
        elif health.status == SystemStatus.WARNING:
            print("⚠️  System resources under stress:")
            for warning in health.warnings:
                print(f"   • {warning}")
        
        # Initialize local data agent for demo
        try:
            print("📚 Initializing local data agent...")
            self.agent = LocalDataAgent()
            
            # Create memory directory if it doesn't exist
            memory_dir = Path.home() / ".eeg-rag"
            memory_dir.mkdir(exist_ok=True)
            
            print("✅ System initialization complete!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize system: {e}")
            self.logger.error(f"Initialization error: {e}")
            return False
    
    def display_help(self):
        """Display available commands"""
        help_text = """
Available Commands:
  query <question>     - Ask a question about EEG research
  stats               - Show system statistics  
  health              - Check system health
  memory              - Show memory usage
  clear               - Clear conversation history
  examples            - Show example queries
  help                - Show this help message
  quit/exit           - Exit the application

Example Usage:
  query What is the typical alpha frequency range?
  query How does sleep deprivation affect EEG patterns?
  query What are common EEG biomarkers for epilepsy?
"""
        print(help_text)
    
    def display_examples(self):
        """Display example queries"""
        examples = """
Example Queries:

🔬 Basic Facts:
  • What is the typical alpha frequency range in EEG?
  • What are the main EEG frequency bands?
  • How many electrodes are used in standard EEG?

🧭 Research Questions:
  • What EEG biomarkers are associated with Alzheimer's disease?
  • How does caffeine affect EEG patterns?
  • What are the differences between resting state and task-based EEG?

🔍 Technical Questions:
  • What is the optimal sampling rate for clinical EEG?
  • How do artifacts affect EEG signal quality?
  • What preprocessing steps are recommended for EEG analysis?

💊 Clinical Applications:
  • How is EEG used to diagnose epilepsy?
  • What EEG patterns indicate sleep disorders?
  • How does anesthesia affect EEG signals?
"""
        print(examples)
    
    async def process_query(self, query_text: str) -> Dict[str, Any]:
        """
        Process a user query through the EEG-RAG system
        
        Args:
            query_text: User's question
            
        Returns:
            Dict containing response and metadata
        """
        if not self.agent:
            return {
                "error": "System not initialized. Please restart the application.",
                "success": False
            }
        
        try:
            print(f"\n🤔 Processing: {query_text}")
            print("⏳ Searching knowledge base...")
            
            # Create agent query
            agent_query = AgentQuery(
                text=query_text,
                intent="factual",
                context={"session_id": self.session_id}
            )
            
            # Store query in memory
            await self.memory_manager.add_query(agent_query)
            
            # Execute query
            result = await self.agent.run(agent_query)
            
            # Store result in memory  
            await self.memory_manager.add_response(result)
            
            if result.success:
                return {
                    "success": True,
                    "answer": result.data,
                    "confidence": getattr(result, 'confidence', None),
                    "sources": getattr(result, 'sources', []),
                    "elapsed_time": result.elapsed_time
                }
            else:
                return {
                    "success": False,
                    "error": result.error or "Query processing failed",
                    "elapsed_time": result.elapsed_time
                }
                
        except Exception as e:
            self.logger.error(f"Query processing error: {e}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }
    
    def format_response(self, response: Dict[str, Any]):
        """Format and display query response"""
        if not response["success"]:
            print(f"\n❌ Error: {response['error']}")
            return
        
        print(f"\n🎯 Answer:")
        print("-" * 40)
        
        answer = response.get("answer", "No answer provided")
        if isinstance(answer, dict):
            # Handle structured responses
            if "content" in answer:
                print(answer["content"])
            else:
                print(json.dumps(answer, indent=2))
        else:
            print(answer)
        
        print("-" * 40)
        
        # Display metadata
        elapsed_time = response.get("elapsed_time", 0)
        print(f"⏱️  Response time: {elapsed_time:.3f}s")
        
        if "confidence" in response and response["confidence"]:
            print(f"🎯 Confidence: {response['confidence']:.2f}")
        
        if "sources" in response and response["sources"]:
            print(f"📚 Sources: {len(response['sources'])} documents")
    
    async def show_stats(self):
        """Display system statistics"""
        print("\n📊 EEG-RAG System Statistics")
        print("=" * 35)
        
        # Agent statistics
        if self.agent:
            stats = self.agent.get_statistics()
            print(f"🤖 Agent: {stats['agent_name']}")
            print(f"   Status: {stats['status']}")
            print(f"   Total queries: {stats['total_executions']}")
            print(f"   Success rate: {stats['success_rate']:.1%}")
            if stats['average_execution_time_seconds'] > 0:
                print(f"   Avg time: {stats['average_execution_time_seconds']:.3f}s")
        
        # Memory statistics
        memory_stats = await self.memory_manager.get_statistics()
        print(f"\n💭 Memory:")
        print(f"   Short-term entries: {memory_stats['short_term']['total_entries']}")
        print(f"   Long-term entries: {memory_stats['long_term']['total_entries']}")
        
        # System health
        health = check_system_health()
        print(f"\n🏥 System Health: {health.status.value}")
        print(f"   CPU: {health.cpu_percent:.1f}%")
        print(f"   Memory: {health.memory_percent:.1f}%") 
        print(f"   Disk: {health.disk_percent:.1f}%")
    
    async def clear_memory(self):
        """Clear conversation history"""
        try:
            await self.memory_manager.cleanup()
            if self.agent:
                self.agent.reset_statistics()
            print("✅ Conversation history cleared")
        except Exception as e:
            print(f"❌ Error clearing memory: {e}")
    
    async def run_interactive(self):
        """Run interactive CLI session"""
        if not await self.initialize_system():
            print("❌ Failed to initialize system. Exiting.")
            return
        
        print("\n💬 Interactive mode started. Type 'help' for commands or 'quit' to exit.")
        
        while True:
            try:
                # Get user input
                user_input = input("\n🧠 eeg-rag> ").strip()
                
                if not user_input:
                    continue
                
                # Parse command
                parts = user_input.split(maxsplit=1)
                command = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""
                
                # Handle commands
                if command in ["quit", "exit", "q"]:
                    print("👋 Goodbye!")
                    break
                
                elif command == "help":
                    self.display_help()
                
                elif command == "examples":
                    self.display_examples()
                
                elif command == "stats":
                    await self.show_stats()
                
                elif command == "health":
                    health = check_system_health()
                    print(f"\n🏥 System Status: {health.status.value}")
                    for warning in health.warnings:
                        print(f"   ⚠️  {warning}")
                
                elif command == "clear":
                    await self.clear_memory()
                
                elif command == "query":
                    if not args:
                        print("❌ Please provide a question. Example: query What is alpha rhythm?")
                        continue
                    
                    response = await self.process_query(args)
                    self.format_response(response)
                
                else:
                    # Treat entire input as query
                    response = await self.process_query(user_input)
                    self.format_response(response)
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                self.logger.error(f"Interactive error: {e}")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser"""
    parser = argparse.ArgumentParser(
        description="EEG-RAG: AI-Powered EEG Research Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Interactive mode
  %(prog)s --query "What is alpha rhythm?"    # Single query
  %(prog)s --stats                           # Show statistics
  %(prog)s --health                          # Check system health
        """
    )
    
    # Add arguments
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Execute a single query and exit"
    )
    
    parser.add_argument(
        "--stats",
        action="store_true", 
        help="Show system statistics and exit"
    )
    
    parser.add_argument(
        "--health",
        action="store_true",
        help="Check system health and exit"
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser


async def cli_main():
    """Main CLI entry point"""
    try:
        import click
        # Use Click CLI if available for extended commands
        from .commands import add_extended_commands
        
        @click.group()
        def main_cli():
            """EEG-RAG: AI-Powered EEG Research Assistant"""
            pass
        
        # Add extended commands
        add_extended_commands(main_cli)
        
        # Add basic interactive command
        @main_cli.command()
        @click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
        @click.option('--json', 'json_output', is_flag=True, help='Output in JSON format')
        async def interactive(verbose, json_output):
            """Start interactive CLI session"""
            app = EEGRAGCLIApp()
            await app.run_interactive()
        
        # Run Click CLI
        main_cli()
        
    except ImportError:
        # Fallback to basic CLI if Click not available
        parser = create_argument_parser()
        args = parser.parse_args()
        
        # Set up logging level
        if args.verbose:
            logging.getLogger().setLevel(logging.INFO)
        
        # Create CLI app
        app = EEGRAGCLIApp()
        
        # Handle single-shot commands
        if args.health:
            health = check_system_health()
            if args.json:
                print(json.dumps(health.to_dict(), indent=2))
            else:
                print(f"System Status: {health.status.value}")
                for warning in health.warnings:
                    print(f"  ⚠️  {warning}")
            return
        
        if args.stats:
            if not await app.initialize_system():
                print("❌ Failed to initialize system")
                return
            
            if args.json:
                stats = {}
                if app.agent:
                    stats["agent"] = app.agent.get_statistics()
                stats["memory"] = await app.memory_manager.get_statistics()
                stats["system"] = check_system_health().to_dict()
                print(json.dumps(stats, indent=2))
            else:
                await app.show_stats()
            return
        
        if args.query:
            if not await app.initialize_system():
                print("❌ Failed to initialize system")
                return
            
            response = await app.process_query(args.query)
            
            if args.json:
                print(json.dumps(response, indent=2))
        else:
            app.format_response(response)
        return
    
    # Default: interactive mode
    await app.run_interactive()


def main():
    """Synchronous entry point"""
    try:
        asyncio.run(cli_main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()