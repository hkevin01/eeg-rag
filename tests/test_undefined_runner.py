#!/usr/bin/env python3
"""
Comprehensive Undefined Behavior Testing Framework Runner

This module provides a comprehensive test runner and reporting system for
the undefined behavior testing framework. It integrates all test modules
and provides detailed analytics, performance comparisons, and safety reports.

Features:
- Automated test discovery and execution
- Performance benchmarking and A/B testing
- Safety assessment scoring
- Detailed HTML and JSON reports
- CI/CD integration support
- Risk assessment and recommendations
- Coverage analysis for undefined behavior patterns
"""

import pytest
import sys
import os
import json
import time
import traceback
import subprocess
import statistics
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import defaultdict
import argparse
import tempfile
import concurrent.futures
from contextlib import contextmanager

# Import our testing modules
try:
    from .test_undefined_behavior import ABTester, ABTestResult
    from .test_undefined_behavior import (
        TestSequencePointViolations, TestTypeConversionDangers,
        TestMemoryLayoutIssues, TestPointerArithmeticSimulation,
        TestConcurrencyPitfalls, TestFloatingPointIssues,
        TestMutableDefaults, TestABTestingIntegration
    )
    from .test_undefined_edge_cases import (
        TestStackOverflowPatterns, TestMemoryManagementEdgeCases,
        TestUnicodeEncodingEdgeCases, TestNumericEdgeCases,
        TestSignalHandlingEdgeCases
    )
except ImportError:
    # Handle case when running directly
    import test_undefined_behavior as ub_module
    import test_undefined_edge_cases as edge_module
    
    ABTester = ub_module.ABTester
    ABTestResult = ub_module.ABTestResult


@dataclass
class TestExecutionResult:
    """Result of test execution"""
    test_name: str
    status: str  # "passed", "failed", "skipped", "error"
    duration: float
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class SafetyAssessment:
    """Safety assessment for a test category"""
    category: str
    risk_level: str  # "low", "medium", "high", "critical"
    vulnerabilities_found: int
    safe_patterns_validated: int
    recommendation: str
    confidence_score: float


@dataclass
class UndefinedBehaviorReport:
    """Comprehensive report of undefined behavior testing"""
    timestamp: datetime
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    error_tests: int
    total_duration: float
    test_results: List[TestExecutionResult]
    safety_assessments: List[SafetyAssessment]
    ab_test_results: List[ABTestResult]
    risk_summary: Dict[str, Any]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "summary": {
                "total_tests": self.total_tests,
                "passed_tests": self.passed_tests,
                "failed_tests": self.failed_tests,
                "skipped_tests": self.skipped_tests,
                "error_tests": self.error_tests,
                "total_duration": self.total_duration,
                "success_rate": self.passed_tests / max(self.total_tests, 1)
            },
            "test_results": [asdict(result) for result in self.test_results],
            "safety_assessments": [asdict(assessment) for assessment in self.safety_assessments],
            "ab_test_results": [result.to_dict() if hasattr(result, 'to_dict') else asdict(result) 
                              for result in self.ab_test_results],
            "risk_summary": self.risk_summary,
            "recommendations": self.recommendations
        }


class UndefinedBehaviorTestRunner:
    """Comprehensive test runner for undefined behavior testing"""
    
    def __init__(self, verbose: bool = False, generate_reports: bool = True):
        self.verbose = verbose
        self.generate_reports = generate_reports
        self.ab_tester = ABTester()
        self.test_results: List[TestExecutionResult] = []
        self.safety_assessments: List[SafetyAssessment] = []
        
        # Test categories and their classes
        self.test_categories = {
            "sequence_points": TestSequencePointViolations,
            "type_conversions": TestTypeConversionDangers,
            "memory_layout": TestMemoryLayoutIssues,
            "pointer_arithmetic": TestPointerArithmeticSimulation,
            "concurrency": TestConcurrencyPitfalls,
            "floating_point": TestFloatingPointIssues,
            "mutable_defaults": TestMutableDefaults,
            "ab_testing": TestABTestingIntegration,
            "stack_overflow": TestStackOverflowPatterns,
            "memory_management": TestMemoryManagementEdgeCases,
            "unicode_encoding": TestUnicodeEncodingEdgeCases,
            "numeric_edge_cases": TestNumericEdgeCases,
            "signal_handling": TestSignalHandlingEdgeCases,
        }
    
    def discover_tests(self) -> Dict[str, List[str]]:
        """Discover all test methods in each category"""
        discovered = {}
        
        for category, test_class in self.test_categories.items():
            methods = [method for method in dir(test_class) 
                      if method.startswith('test_') and callable(getattr(test_class, method))]
            discovered[category] = methods
            
            if self.verbose:
                print(f"Discovered {len(methods)} tests in {category}")
        
        return discovered
    
    def run_single_test(self, test_class, method_name: str) -> TestExecutionResult:
        """Run a single test method"""
        start_time = time.time()
        
        try:
            # Create test instance
            instance = test_class()
            
            # Get the test method
            test_method = getattr(instance, method_name)
            
            # Run the test
            if hasattr(test_method, '__code__') and 'async' in str(test_method.__code__):
                # Handle async tests
                import asyncio
                asyncio.run(test_method())
            else:
                test_method()
            
            duration = time.time() - start_time
            
            return TestExecutionResult(
                test_name=f"{test_class.__name__}.{method_name}",
                status="passed",
                duration=duration
            )
        
        except AssertionError as e:
            duration = time.time() - start_time
            return TestExecutionResult(
                test_name=f"{test_class.__name__}.{method_name}",
                status="failed",
                duration=duration,
                error_message=str(e)
            )
        
        except Exception as e:
            duration = time.time() - start_time
            return TestExecutionResult(
                test_name=f"{test_class.__name__}.{method_name}",
                status="error",
                duration=duration,
                error_message=f"{type(e).__name__}: {str(e)}"
            )
    
    def run_category_tests(self, category: str, test_methods: List[str]) -> List[TestExecutionResult]:
        """Run all tests in a category"""
        if self.verbose:
            print(f"\\nRunning {category} tests...")
        
        test_class = self.test_categories[category]
        results = []
        
        for method_name in test_methods:
            if self.verbose:
                print(f"  Running {method_name}...")
            
            result = self.run_single_test(test_class, method_name)
            results.append(result)
            
            if self.verbose:
                status_icon = "✅" if result.status == "passed" else "❌" if result.status == "failed" else "⚠️"
                print(f"    {status_icon} {result.status.upper()} ({result.duration:.3f}s)")
                if result.error_message:
                    print(f"      Error: {result.error_message}")
        
        return results
    
    def assess_category_safety(self, category: str, results: List[TestExecutionResult]) -> SafetyAssessment:
        """Assess safety for a test category"""
        passed_tests = sum(1 for r in results if r.status == "passed")
        failed_tests = sum(1 for r in results if r.status == "failed") 
        error_tests = sum(1 for r in results if r.status == "error")
        
        total_tests = len(results)
        success_rate = passed_tests / max(total_tests, 1)
        
        # Determine risk level based on test results
        if success_rate >= 0.95:
            risk_level = "low"
        elif success_rate >= 0.80:
            risk_level = "medium"
        elif success_rate >= 0.60:
            risk_level = "high"
        else:
            risk_level = "critical"
        
        # Generate recommendations
        if failed_tests > 0:
            recommendation = f"Review and fix {failed_tests} failing safety checks in {category}"
        elif error_tests > 0:
            recommendation = f"Investigate {error_tests} test errors in {category}"
        else:
            recommendation = f"All safety checks passed for {category} - maintain current practices"
        
        return SafetyAssessment(
            category=category,
            risk_level=risk_level,
            vulnerabilities_found=failed_tests + error_tests,
            safe_patterns_validated=passed_tests,
            recommendation=recommendation,
            confidence_score=success_rate
        )
    
    def run_ab_performance_tests(self) -> List[ABTestResult]:
        """Run A/B performance comparison tests"""
        if self.verbose:
            print("\\nRunning A/B performance tests...")
        
        ab_results = []
        
        # Test 1: List concatenation vs extend
        def concat_method(lists):
            result = []
            for lst in lists:
                result = result + lst
            return result
        
        def extend_method(lists):
            result = []
            for lst in lists:
                result.extend(lst)
            return result
        
        test_data = [
            [[1, 2], [3, 4], [5, 6]],
            [[10, 20, 30], [40, 50]],
            [list(range(100)), list(range(100, 200))]
        ]
        
        ab_result1 = self.ab_tester.run_comparison(
            "list_concatenation",
            concat_method,
            extend_method,
            test_data,
            iterations=50
        )
        ab_results.append(ab_result1)
        
        # Test 2: Dictionary access patterns
        def dict_get_unsafe(data):
            key, default = data
            test_dict = {"a": 1, "b": 2, "c": 3}
            return test_dict[key]  # May raise KeyError
        
        def dict_get_safe(data):
            key, default = data
            test_dict = {"a": 1, "b": 2, "c": 3}
            return test_dict.get(key, default)
        
        dict_test_data = [
            ("a", 0),
            ("b", 0), 
            ("missing_key", -1)
        ]
        
        ab_result2 = self.ab_tester.run_comparison(
            "dictionary_access",
            dict_get_unsafe,
            dict_get_safe,
            dict_test_data,
            iterations=50
        )
        ab_results.append(ab_result2)
        
        # Test 3: String formatting methods
        def old_format(data):
            name, age = data
            return "Hello, %s! You are %d years old." % (name, age)
        
        def new_format(data):
            name, age = data
            return f"Hello, {name}! You are {age} years old."
        
        format_test_data = [
            ("Alice", 30),
            ("Bob", 25),
            ("Charlie", 35)
        ]
        
        ab_result3 = self.ab_tester.run_comparison(
            "string_formatting",
            old_format,
            new_format,
            format_test_data,
            iterations=50
        )
        ab_results.append(ab_result3)
        
        if self.verbose:
            for result in ab_results:
                print(f"  A/B Test: {result.recommendation}")
        
        return ab_results
    
    def generate_risk_summary(self) -> Dict[str, Any]:
        """Generate overall risk summary"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.status == "passed")
        failed_tests = sum(1 for r in self.test_results if r.status == "failed")
        error_tests = sum(1 for r in self.test_results if r.status == "error")
        
        overall_success_rate = passed_tests / max(total_tests, 1)
        
        # Risk categories
        risk_counts = defaultdict(int)
        for assessment in self.safety_assessments:
            risk_counts[assessment.risk_level] += 1
        
        # Performance insights
        avg_test_duration = statistics.mean([r.duration for r in self.test_results]) if self.test_results else 0
        
        return {
            "overall_success_rate": overall_success_rate,
            "risk_distribution": dict(risk_counts),
            "performance_metrics": {
                "average_test_duration": avg_test_duration,
                "total_execution_time": sum(r.duration for r in self.test_results)
            },
            "critical_issues": failed_tests + error_tests,
            "safety_score": overall_success_rate * 100
        }
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Analyze failed tests
        failed_tests = [r for r in self.test_results if r.status == "failed"]
        if failed_tests:
            recommendations.append(
                f"Fix {len(failed_tests)} failing safety checks to improve code security"
            )
        
        # Analyze error tests
        error_tests = [r for r in self.test_results if r.status == "error"]
        if error_tests:
            recommendations.append(
                f"Investigate {len(error_tests)} test errors that may indicate system issues"
            )
        
        # Analyze high-risk categories
        high_risk_categories = [a for a in self.safety_assessments if a.risk_level in ["high", "critical"]]
        if high_risk_categories:
            categories = ", ".join(a.category for a in high_risk_categories)
            recommendations.append(
                f"Priority review needed for high-risk categories: {categories}"
            )
        
        # Performance recommendations from A/B tests
        for ab_result in self.ab_tester.results:
            if "B is better" in ab_result.recommendation:
                recommendations.append(
                    f"Consider adopting safer implementation pattern (showed {ab_result.confidence:.1f}% confidence)"
                )
        
        # General recommendations
        if not recommendations:
            recommendations.append("All safety checks passed - maintain current development practices")
        
        recommendations.append("Regular undefined behavior testing recommended for ongoing safety")
        
        return recommendations
    
    def run_all_tests(self) -> UndefinedBehaviorReport:
        """Run all undefined behavior tests and generate comprehensive report"""
        start_time = time.time()
        
        if self.verbose:
            print("🔥 Starting Comprehensive Undefined Behavior Testing 🔥")
            print("=" * 70)
        
        # Discover tests
        discovered_tests = self.discover_tests()
        
        # Run tests by category
        for category, methods in discovered_tests.items():
            results = self.run_category_tests(category, methods)
            self.test_results.extend(results)
            
            # Assess safety for this category
            safety_assessment = self.assess_category_safety(category, results)
            self.safety_assessments.append(safety_assessment)
        
        # Run A/B performance tests
        ab_results = self.run_ab_performance_tests()
        
        total_duration = time.time() - start_time
        
        # Generate summary statistics
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.status == "passed")
        failed_tests = sum(1 for r in self.test_results if r.status == "failed")
        skipped_tests = sum(1 for r in self.test_results if r.status == "skipped")
        error_tests = sum(1 for r in self.test_results if r.status == "error")
        
        # Generate risk summary and recommendations
        risk_summary = self.generate_risk_summary()
        recommendations = self.generate_recommendations()
        
        # Create comprehensive report
        report = UndefinedBehaviorReport(
            timestamp=datetime.now(),
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            skipped_tests=skipped_tests,
            error_tests=error_tests,
            total_duration=total_duration,
            test_results=self.test_results,
            safety_assessments=self.safety_assessments,
            ab_test_results=ab_results,
            risk_summary=risk_summary,
            recommendations=recommendations
        )
        
        if self.verbose:
            self.print_summary_report(report)
        
        return report
    
    def print_summary_report(self, report: UndefinedBehaviorReport):
        """Print a summary report to console"""
        print(f"\\n" + "=" * 70)
        print("📊 UNDEFINED BEHAVIOR TESTING SUMMARY")
        print("=" * 70)
        
        # Test results summary
        print(f"Tests Executed: {report.total_tests}")
        print(f"✅ Passed: {report.passed_tests}")
        print(f"❌ Failed: {report.failed_tests}")
        print(f"⚠️  Errors: {report.error_tests}")
        print(f"⏭️  Skipped: {report.skipped_tests}")
        print(f"⏱️  Total Duration: {report.total_duration:.2f}s")
        print(f"📈 Success Rate: {(report.passed_tests/max(report.total_tests,1)*100):.1f}%")
        
        # Safety assessment summary
        print(f"\\n🛡️  SAFETY ASSESSMENT")
        print("-" * 30)
        for assessment in report.safety_assessments:
            risk_icon = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}
            icon = risk_icon.get(assessment.risk_level, "⚪")
            print(f"{icon} {assessment.category}: {assessment.risk_level.upper()} risk "
                  f"({assessment.safe_patterns_validated}/{assessment.safe_patterns_validated + assessment.vulnerabilities_found} safe)")
        
        # A/B test summary
        if report.ab_test_results:
            print(f"\\n⚡ A/B PERFORMANCE TESTS")
            print("-" * 30)
            for ab_result in report.ab_test_results:
                print(f"📊 {ab_result.recommendation} (confidence: {ab_result.confidence:.1f}%)")
        
        # Recommendations
        if report.recommendations:
            print(f"\\n💡 RECOMMENDATIONS")
            print("-" * 30)
            for i, rec in enumerate(report.recommendations, 1):
                print(f"{i}. {rec}")
        
        print(f"\\n🎉 Undefined Behavior Testing Complete! 🎉")
    
    def save_json_report(self, report: UndefinedBehaviorReport, filepath: Optional[str] = None) -> str:
        """Save report as JSON file"""
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"undefined_behavior_report_{timestamp}.json"
        
        with open(filepath, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        
        if self.verbose:
            print(f"JSON report saved to: {filepath}")
        
        return filepath
    
    def save_html_report(self, report: UndefinedBehaviorReport, filepath: Optional[str] = None) -> str:
        """Save report as HTML file"""
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"undefined_behavior_report_{timestamp}.html"
        
        html_content = self._generate_html_report(report)
        
        with open(filepath, 'w') as f:
            f.write(html_content)
        
        if self.verbose:
            print(f"HTML report saved to: {filepath}")
        
        return filepath
    
    def _generate_html_report(self, report: UndefinedBehaviorReport) -> str:
        """Generate HTML report content"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Undefined Behavior Testing Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 8px; }}
                .summary {{ background: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .test-results {{ margin: 20px 0; }}
                .passed {{ color: #27ae60; }}
                .failed {{ color: #e74c3c; }}
                .error {{ color: #f39c12; }}
                .risk-low {{ color: #27ae60; }}
                .risk-medium {{ color: #f39c12; }}
                .risk-high {{ color: #e67e22; }}
                .risk-critical {{ color: #e74c3c; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #34495e; color: white; }}
                .recommendations {{ background: #d5dbdb; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔥 Undefined Behavior Testing Report 🔥</h1>
                <p>Generated: {report.timestamp.strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
            
            <div class="summary">
                <h2>📊 Executive Summary</h2>
                <p><strong>Total Tests:</strong> {report.total_tests}</p>
                <p><strong>Success Rate:</strong> {(report.passed_tests/max(report.total_tests,1)*100):.1f}%</p>
                <p><strong>Safety Score:</strong> {report.risk_summary.get('safety_score', 0):.1f}/100</p>
                <p><strong>Execution Time:</strong> {report.total_duration:.2f}s</p>
            </div>
            
            <div class="test-results">
                <h2>🧪 Test Results by Category</h2>
                <table>
                    <tr>
                        <th>Category</th>
                        <th>Risk Level</th>
                        <th>Vulnerabilities</th>
                        <th>Safe Patterns</th>
                        <th>Recommendation</th>
                    </tr>
        """
        
        for assessment in report.safety_assessments:
            risk_class = f"risk-{assessment.risk_level}"
            html += f"""
                    <tr>
                        <td>{assessment.category}</td>
                        <td class="{risk_class}">{assessment.risk_level.upper()}</td>
                        <td>{assessment.vulnerabilities_found}</td>
                        <td>{assessment.safe_patterns_validated}</td>
                        <td>{assessment.recommendation}</td>
                    </tr>
            """
        
        html += """
                </table>
            </div>
        """
        
        if report.ab_test_results:
            html += """
            <div class="ab-results">
                <h2>⚡ A/B Performance Results</h2>
                <table>
                    <tr>
                        <th>Test</th>
                        <th>Variant A Performance</th>
                        <th>Variant B Performance</th>
                        <th>Recommendation</th>
                        <th>Confidence</th>
                    </tr>
            """
            
            for ab_result in report.ab_test_results:
                html += f"""
                    <tr>
                        <td>Performance Comparison</td>
                        <td>{ab_result.performance_a:.4f}s</td>
                        <td>{ab_result.performance_b:.4f}s</td>
                        <td>{ab_result.recommendation}</td>
                        <td>{ab_result.confidence:.1f}%</td>
                    </tr>
                """
            
            html += """
                </table>
            </div>
            """
        
        html += f"""
            <div class="recommendations">
                <h2>💡 Recommendations</h2>
                <ol>
        """
        
        for rec in report.recommendations:
            html += f"<li>{rec}</li>"
        
        html += """
                </ol>
            </div>
            
            <div class="footer">
                <p><em>Report generated by Undefined Behavior Testing Framework</em></p>
            </div>
        </body>
        </html>
        """
        
        return html


# =============================================================================
# Command Line Interface
# =============================================================================

def main():
    """Main command line interface"""
    parser = argparse.ArgumentParser(
        description="Comprehensive Undefined Behavior Testing Framework"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--output-json",
        help="Output JSON report to specified file"
    )
    
    parser.add_argument(
        "--output-html", 
        help="Output HTML report to specified file"
    )
    
    parser.add_argument(
        "--no-ab-tests",
        action="store_true",
        help="Skip A/B performance tests"
    )
    
    parser.add_argument(
        "--categories",
        nargs="+",
        help="Run only specified test categories"
    )
    
    args = parser.parse_args()
    
    # Create test runner
    runner = UndefinedBehaviorTestRunner(
        verbose=args.verbose,
        generate_reports=bool(args.output_json or args.output_html)
    )
    
    # Run tests
    try:
        report = runner.run_all_tests()
        
        # Save reports if requested
        if args.output_json:
            runner.save_json_report(report, args.output_json)
        
        if args.output_html:
            runner.save_html_report(report, args.output_html)
        
        # Exit with appropriate code
        if report.failed_tests > 0 or report.error_tests > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except Exception as e:
        print(f"Error running tests: {e}")
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    # For direct execution
    runner = UndefinedBehaviorTestRunner(verbose=True)
    report = runner.run_all_tests()
    
    # Save reports with timestamps
    runner.save_json_report(report)
    runner.save_html_report(report)
    
    print(f"\\n✨ Comprehensive undefined behavior testing completed!")
    print(f"📁 Reports saved with timestamp")