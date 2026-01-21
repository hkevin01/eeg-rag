"""
Comparison module for systematic review results.

Compares new extraction results against baseline studies (e.g., Roy et al. 2019)
to identify trends and changes over time.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class SystematicReviewComparator:
    """
    Compare systematic review extractions against baseline studies.
    
    Example:
        comparator = SystematicReviewComparator(
            baseline_path="roy_2019_data.csv"
        )
        comparison = comparator.compare(new_results_df)
        print(comparison.summary())
    """
    
    def __init__(self, baseline_path: Optional[str] = None):
        """
        Initialize comparator.
        
        Args:
            baseline_path: Path to baseline study data (CSV)
        """
        self.baseline_df = None
        if baseline_path:
            self.load_baseline(baseline_path)
    
    def load_baseline(self, path: str):
        """Load baseline study data."""
        self.baseline_df = pd.read_csv(path)
        logger.info(f"Loaded baseline data: {len(self.baseline_df)} papers from {path}")
    
    def compare(
        self,
        new_df: pd.DataFrame,
        comparison_fields: Optional[List[str]] = None
    ) -> 'ComparisonResults':
        """
        Compare new results against baseline.
        
        Args:
            new_df: DataFrame with new extraction results
            comparison_fields: Fields to compare. If None, compares all common fields.
        
        Returns:
            ComparisonResults object
        """
        if self.baseline_df is None:
            raise ValueError("Baseline data not loaded. Call load_baseline() first.")
        
        if comparison_fields is None:
            # Find common columns
            comparison_fields = list(
                set(self.baseline_df.columns) & set(new_df.columns)
            )
            comparison_fields = [f for f in comparison_fields if f not in [
                'paper_id', 'title', 'authors', 'doi', 'pmid', 'extraction_timestamp'
            ]]
        
        results = ComparisonResults()
        results.baseline_count = len(self.baseline_df)
        results.new_count = len(new_df)
        
        # Year distribution comparison
        if 'year' in comparison_fields:
            results.add_trend("year_distribution", self._compare_years(new_df))
        
        # Architecture trends
        if 'architecture_type' in comparison_fields:
            results.add_trend("architecture_shifts", self._compare_architectures(new_df))
        
        # Performance improvements
        if 'reported_accuracy' in comparison_fields:
            results.add_trend("performance_improvements", self._compare_performance(new_df))
        
        # Code availability trends
        if 'code_available' in comparison_fields:
            results.add_trend("reproducibility", self._compare_reproducibility(new_df))
        
        # Dataset usage
        if 'dataset_name' in comparison_fields:
            results.add_trend("dataset_usage", self._compare_datasets(new_df))
        
        # Task type distribution
        if 'task_type' in comparison_fields:
            results.add_trend("task_distribution", self._compare_tasks(new_df))
        
        return results
    
    def _compare_years(self, new_df: pd.DataFrame) -> Dict:
        """Compare year distributions."""
        baseline_years = self.baseline_df['year'].value_counts().sort_index()
        new_years = new_df['year'].value_counts().sort_index()
        
        return {
            "baseline_year_range": f"{baseline_years.index.min()}-{baseline_years.index.max()}",
            "new_year_range": f"{new_years.index.min()}-{new_years.index.max()}",
            "baseline_median": int(baseline_years.median()),
            "new_median": int(new_years.median()),
            "papers_per_year_baseline": baseline_years.mean(),
            "papers_per_year_new": new_years.mean(),
            "growth_rate": f"{((new_years.mean() - baseline_years.mean()) / baseline_years.mean() * 100):.1f}%"
        }
    
    def _compare_architectures(self, new_df: pd.DataFrame) -> Dict:
        """Compare architecture type distributions."""
        baseline_arch = self.baseline_df['architecture_type'].value_counts(normalize=True)
        new_arch = new_df['architecture_type'].value_counts(normalize=True)
        
        trends = {}
        for arch in set(baseline_arch.index) | set(new_arch.index):
            baseline_pct = baseline_arch.get(arch, 0) * 100
            new_pct = new_arch.get(arch, 0) * 100
            change = new_pct - baseline_pct
            
            trends[arch] = {
                "baseline_percentage": f"{baseline_pct:.1f}%",
                "new_percentage": f"{new_pct:.1f}%",
                "change": f"{change:+.1f}%",
                "trend": "📈" if change > 5 else "📉" if change < -5 else "→"
            }
        
        return trends
    
    def _compare_performance(self, new_df: pd.DataFrame) -> Dict:
        """Compare reported performance metrics."""
        baseline_acc = self.baseline_df['reported_accuracy'].dropna()
        new_acc = new_df['reported_accuracy'].dropna()
        
        return {
            "baseline_mean_accuracy": f"{baseline_acc.mean():.3f}",
            "new_mean_accuracy": f"{new_acc.mean():.3f}",
            "improvement": f"{(new_acc.mean() - baseline_acc.mean()):.3f}",
            "baseline_median": f"{baseline_acc.median():.3f}",
            "new_median": f"{new_acc.median():.3f}",
            "baseline_std": f"{baseline_acc.std():.3f}",
            "new_std": f"{new_acc.std():.3f}",
            "papers_with_metrics_baseline": f"{len(baseline_acc)} ({len(baseline_acc)/len(self.baseline_df)*100:.1f}%)",
            "papers_with_metrics_new": f"{len(new_acc)} ({len(new_acc)/len(new_df)*100:.1f}%)"
        }
    
    def _compare_reproducibility(self, new_df: pd.DataFrame) -> Dict:
        """Compare code/data availability trends."""
        def categorize_availability(df):
            if 'code_available' not in df.columns:
                return {}
            
            avail = df['code_available'].value_counts(normalize=True) * 100
            return {
                "github_link": avail.get("GitHub link found", 0),
                "available_on_request": avail.get("Code available upon request", 0),
                "not_available": avail.get("Not available", 0)
            }
        
        baseline_avail = categorize_availability(self.baseline_df)
        new_avail = categorize_availability(new_df)
        
        comparison = {}
        for category in ["github_link", "available_on_request", "not_available"]:
            baseline_pct = baseline_avail.get(category, 0)
            new_pct = new_avail.get(category, 0)
            change = new_pct - baseline_pct
            
            comparison[category] = {
                "baseline": f"{baseline_pct:.1f}%",
                "new": f"{new_pct:.1f}%",
                "change": f"{change:+.1f}%"
            }
        
        # Calculate reproducibility score improvement
        baseline_score = baseline_avail.get("github_link", 0)
        new_score = new_avail.get("github_link", 0)
        
        comparison["reproducibility_score_change"] = f"{new_score - baseline_score:+.1f}%"
        comparison["interpretation"] = (
            "Improved" if new_score > baseline_score else
            "Declined" if new_score < baseline_score else
            "Stable"
        )
        
        return comparison
    
    def _compare_datasets(self, new_df: pd.DataFrame) -> Dict:
        """Compare dataset usage patterns."""
        baseline_datasets = self.baseline_df['dataset_name'].value_counts().head(10)
        new_datasets = new_df['dataset_name'].value_counts().head(10)
        
        return {
            "most_common_baseline": baseline_datasets.index[0] if len(baseline_datasets) > 0 else "N/A",
            "most_common_new": new_datasets.index[0] if len(new_datasets) > 0 else "N/A",
            "unique_datasets_baseline": self.baseline_df['dataset_name'].nunique(),
            "unique_datasets_new": new_df['dataset_name'].nunique(),
            "emerging_datasets": list(set(new_datasets.index) - set(baseline_datasets.index))[:5]
        }
    
    def _compare_tasks(self, new_df: pd.DataFrame) -> Dict:
        """Compare task type distributions."""
        baseline_tasks = self.baseline_df['task_type'].value_counts(normalize=True) * 100
        new_tasks = new_df['task_type'].value_counts(normalize=True) * 100
        
        tasks_comparison = {}
        for task in set(baseline_tasks.index) | set(new_tasks.index):
            baseline_pct = baseline_tasks.get(task, 0)
            new_pct = new_tasks.get(task, 0)
            
            tasks_comparison[task] = {
                "baseline": f"{baseline_pct:.1f}%",
                "new": f"{new_pct:.1f}%",
                "change": f"{new_pct - baseline_pct:+.1f}%"
            }
        
        return tasks_comparison


class ComparisonResults:
    """Container for comparison results with formatting methods."""
    
    def __init__(self):
        self.baseline_count = 0
        self.new_count = 0
        self.trends: Dict[str, Dict] = {}
    
    def add_trend(self, trend_name: str, trend_data: Dict):
        """Add a trend comparison."""
        self.trends[trend_name] = trend_data
    
    def summary(self) -> str:
        """Generate formatted summary of comparison."""
        lines = [
            "="*80,
            "SYSTEMATIC REVIEW COMPARISON SUMMARY",
            "="*80,
            f"\nBaseline Papers: {self.baseline_count}",
            f"New Papers: {self.new_count}",
            f"Total Growth: {self.new_count - self.baseline_count} papers ({(self.new_count/self.baseline_count - 1)*100:.1f}% increase)",
            "\n" + "="*80,
        ]
        
        for trend_name, trend_data in self.trends.items():
            lines.append(f"\n{trend_name.upper().replace('_', ' ')}")
            lines.append("-"*80)
            self._format_trend_data(trend_data, lines, indent=2)
        
        lines.append("\n" + "="*80)
        return "\n".join(lines)
    
    def _format_trend_data(self, data: Dict, lines: List[str], indent: int = 0):
        """Recursively format trend data."""
        prefix = " " * indent
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{prefix}{key}:")
                self._format_trend_data(value, lines, indent + 2)
            else:
                lines.append(f"{prefix}{key}: {value}")
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "baseline_count": self.baseline_count,
            "new_count": self.new_count,
            "trends": self.trends
        }


class ReproducibilityScorer:
    """
    Score papers on reproducibility based on code/data availability.
    
    Example:
        scorer = ReproducibilityScorer()
        df['reproducibility_score'] = df.apply(
            lambda row: scorer.score_paper(row), axis=1
        )
    """
    
    # Scoring rubric
    SCORES = {
        "github_public": 10,
        "github_private": 7,
        "code_available_request": 5,
        "code_available_publication": 3,
        "dataset_public": 8,
        "dataset_restricted": 4,
        "no_code": 0,
        "no_data": 0
    }
    
    def score_paper(self, paper: Dict) -> Tuple[int, str, List[str]]:
        """
        Score a single paper's reproducibility.
        
        Args:
            paper: Dict or DataFrame row with paper metadata
        
        Returns:
            (score, category, justification_list)
        """
        score = 0
        justifications = []
        
        # Code availability
        code_avail = str(paper.get('code_available', '')).lower()
        
        if 'github' in code_avail or 'gitlab' in code_avail or 'bitbucket' in code_avail:
            score += self.SCORES['github_public']
            justifications.append("✅ Public code repository")
        elif 'available upon request' in code_avail:
            score += self.SCORES['code_available_request']
            justifications.append("⚠️  Code available upon request")
        elif 'upon publication' in code_avail or 'upon acceptance' in code_avail:
            score += self.SCORES['code_available_publication']
            justifications.append("⏳ Code will be available")
        else:
            score += self.SCORES['no_code']
            justifications.append("❌ No code availability")
        
        # Dataset availability
        dataset = str(paper.get('dataset_name', '')).lower()
        
        # Known public datasets
        public_datasets = [
            'tusz', 'chb-mit', 'bonn', 'physionet', 'deap', 'seed',
            'bci competition', 'mnist', 'cifar', 'imagenet'
        ]
        
        if any(pd in dataset for pd in public_datasets):
            score += self.SCORES['dataset_public']
            justifications.append("✅ Uses public dataset")
        elif 'private' in dataset or 'clinical' in dataset:
            score += self.SCORES['dataset_restricted']
            justifications.append("⚠️  Uses restricted/private dataset")
        elif dataset and dataset != 'nan':
            score += self.SCORES['dataset_restricted']
            justifications.append("⚠️  Dataset availability unclear")
        else:
            score += self.SCORES['no_data']
            justifications.append("❌ No dataset information")
        
        # Categorize
        if score >= 15:
            category = "Fully Reproducible"
        elif score >= 10:
            category = "Partially Reproducible"
        elif score >= 5:
            category = "Limited Reproducibility"
        else:
            category = "Not Reproducible"
        
        return score, category, justifications
    
    def score_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score all papers in a dataset."""
        results = []
        for _, paper in df.iterrows():
            score, category, justifications = self.score_paper(paper)
            results.append({
                'reproducibility_score': score,
                'reproducibility_category': category,
                'reproducibility_justification': "; ".join(justifications)
            })
        
        result_df = pd.DataFrame(results)
        return pd.concat([df, result_df], axis=1)
    
    def generate_report(self, df: pd.DataFrame) -> str:
        """Generate reproducibility report for a dataset."""
        scored_df = self.score_dataset(df)
        
        categories = scored_df['reproducibility_category'].value_counts()
        mean_score = scored_df['reproducibility_score'].mean()
        
        report = f"""
REPRODUCIBILITY REPORT
{'='*80}

Total Papers Analyzed: {len(scored_df)}
Mean Reproducibility Score: {mean_score:.2f} / 18

Category Distribution:
{'-'*80}
"""
        for cat, count in categories.items():
            pct = count / len(scored_df) * 100
            report += f"  {cat:.<40} {count:>5} ({pct:>5.1f}%)\n"
        
        report += f"\n{'='*80}\n"
        return report
