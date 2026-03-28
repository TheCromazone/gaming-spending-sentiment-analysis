"""
Gaming Spending Sentiment Analysis - Main Analysis Module

Analyzes self-reported gaming spending patterns to empirically test
behavioral economics theses about loot boxes vs battle passes.

This module:
- Loads scraped Reddit data (or synthetic fallback)
- Performs VADER sentiment analysis on spending discussions
- Categorizes posts by monetization model (loot box vs battle pass)
- Extracts spending amounts via regex
- Tests correlation between loss aversion language and reported spending
- Generates publication-quality visualizations
"""

import re
import csv
import logging
from pathlib import Path
from typing import Tuple, Dict, List
from dataclasses import dataclass

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# Try to import optional dependencies, provide fallbacks
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Try to import NLTK, fall back to lightweight implementation if unavailable
try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style for professional visualizations
if SEABORN_AVAILABLE:
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        plt.style.use('default')
    sns.set_palette("husl")
else:
    plt.style.use('default')


def ttest_ind(group1, group2):
    """
    Simple implementation of independent samples t-test when scipy unavailable.

    Args:
        group1, group2: Arrays of values

    Returns:
        Tuple of (t_statistic, p_value)
    """
    if len(group1) < 2 or len(group2) < 2:
        return 0, 1.0

    mean1 = np.mean(group1)
    mean2 = np.mean(group2)
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    n1 = len(group1)
    n2 = len(group2)

    # Pooled variance
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)

    # Standard error
    se = math.sqrt(pooled_var * (1/n1 + 1/n2))

    # t-statistic
    t_stat = (mean1 - mean2) / se if se > 0 else 0

    # Very rough p-value approximation (use scipy for accuracy)
    # For demonstration: two-tailed test approximation
    p_value = 2 * (1 - norm_cdf(abs(t_stat)))

    return t_stat, p_value


def norm_cdf(x):
    """Approximate standard normal CDF using error function."""
    # Using approximation for standard normal CDF
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


@dataclass
class SpendingPost:
    """Represents an analyzed post about gaming spending."""
    post_id: str
    subreddit: str
    text: str
    model_type: str  # 'loot_box' or 'battle_pass'
    behavioral_pattern: str
    sentiment_score: float  # VADER compound score (-1 to 1)
    sentiment_label: str  # 'positive', 'neutral', 'negative'
    spending_amount: float
    has_loss_aversion: bool
    has_fomo: bool
    has_sunk_cost: bool


class SimpleSentimentAnalyzer:
    """
    Lightweight sentiment analyzer when NLTK is unavailable.

    Uses lexicon-based approach with sentiment word scoring.
    Provides VADER-compatible output for compatibility.
    """

    POSITIVE_WORDS = {
        'good', 'great', 'excellent', 'amazing', 'awesome', 'love',
        'best', 'worth', 'happy', 'enjoy', 'fun', 'fantastic',
        'brilliant', 'wonderful', 'perfect', 'great deal', 'bargain',
        'reasonable', 'fair', 'value', 'satisfaction'
    }

    NEGATIVE_WORDS = {
        'bad', 'terrible', 'awful', 'horrible', 'hate', 'waste',
        'regret', 'stupid', 'worst', 'disappointing', 'scam', 'sad',
        'angry', 'annoying', 'broken', 'useless', 'rip off', 'unfair',
        'wasted', 'lost', 'spent too much', 'expensive', 'ripoff'
    }

    INTENSIFIERS = {'very', 'extremely', 'so', 'really', 'absolutely'}
    NEGATORS = {'not', 'no', 'never', "don't", "doesn't", "didn't"}

    def polarity_scores(self, text):
        """Calculate sentiment scores in VADER-compatible format."""
        text_lower = text.lower()
        words = text_lower.split()

        positive_score = 0
        negative_score = 0

        for i, word in enumerate(words):
            # Check for negation
            is_negated = (i > 0 and words[i-1] in self.NEGATORS)
            is_intensified = (i > 0 and words[i-1] in self.INTENSIFIERS)

            multiplier = 1.5 if is_intensified else 1.0

            if word in self.POSITIVE_WORDS:
                if is_negated:
                    negative_score += 0.5 * multiplier
                else:
                    positive_score += 0.5 * multiplier
            elif word in self.NEGATIVE_WORDS:
                if is_negated:
                    positive_score += 0.5 * multiplier
                else:
                    negative_score += 0.5 * multiplier

        # Normalize scores
        total = positive_score + negative_score
        if total == 0:
            return {'compound': 0.0, 'pos': 0.0, 'neu': 1.0, 'neg': 0.0}

        pos_ratio = positive_score / total if total > 0 else 0
        neg_ratio = negative_score / total if total > 0 else 0

        # Calculate compound score
        compound = (positive_score - negative_score) / (total if total > 0 else 1)
        compound = max(-1, min(1, compound))  # Clamp to [-1, 1]

        return {
            'compound': compound,
            'pos': pos_ratio,
            'neu': 1 - (pos_ratio + neg_ratio),
            'neg': neg_ratio
        }


class GamingSpendingAnalyzer:
    """
    Analyzes gaming spending data using behavioral economics frameworks.

    Tests thesis claims: loss aversion language correlates with loot box
    spending, while battle passes show more rational spending patterns.
    Uses VADER sentiment analysis (or lightweight fallback) and linguistic pattern matching.
    """

    # Loss aversion markers - indicate regret or sunk cost framing
    LOSS_AVERSION_WORDS = {
        'regret', 'wasted', 'lost', 'waste', 'threw away', 'mistake',
        'shouldn\'t have', 'shouldn\'t', 'bad luck', 'duplicate',
        'duplicate protection', 'down', 'spent too much',
        'can\'t believe', 'ruined'
    }

    # FOMO (Fear of Missing Out) markers
    FOMO_WORDS = {
        'fomo', 'limited', 'time-limited', 'limited time', 'event ended',
        'everyone had', 'keep up', 'forced', 'pressure', 'pressured',
        'miss out', 'missed out', 'exclusive', 'seasonal'
    }

    # Sunk cost fallacy markers - past spending justifies future spending
    SUNK_COST_WORDS = {
        'already spent', 'already put', 'committed', 'invested',
        'can\'t stop', 'might as well', 'have to', 'see it through',
        'can\'t quit', 'this far in', 'too far in'
    }

    # Loot box discussion keywords
    LOOTBOX_KEYWORDS = {
        'loot box', 'loot boxes', 'pack', 'packs', 'rng', 'odds',
        'gambling', 'randomized', 'duplicate', 'bad luck', 'case',
        'chest', 'crate', 'ultimate team', 'myteam', 'fifa points', 'vc'
    }

    # Battle pass discussion keywords
    BATTLEPASS_KEYWORDS = {
        'battle pass', 'season pass', 'pass', 'seasonal', 'tier',
        'guaranteed', 'grind', 'challenge', 'reward track',
        'predictable', 'content season', 'battle royale season'
    }

    # Spending amount extraction patterns (dollars)
    SPENDING_PATTERNS = [
        r'\$(\d+(?:\.\d{2})?)',  # $100, $99.99
        r'(\d+)\s*dollars',      # 100 dollars
        r'(\d+)\s*bucks',        # 100 bucks
        r'\£(\d+(?:\.\d{2})?)',  # £100
        r'spent\s*(?:about\s*)?(?:around\s*)?(\d+)',  # spent 100
    ]

    def __init__(self, data_path: str = None):
        """
        Initialize analyzer with optional data path.

        Args:
            data_path: Path to CSV file with Reddit data
        """
        self.data_path = data_path
        self.df = None
        self.analyzed_posts = []

        # Use NLTK VADER if available, otherwise use lightweight implementation
        if NLTK_AVAILABLE:
            self.sia = SentimentIntensityAnalyzer()
            logger.info("Using NLTK VADER sentiment analyzer")
        else:
            self.sia = SimpleSentimentAnalyzer()
            logger.info("Using lightweight fallback sentiment analyzer")

        logger.info("Initialized GamingSpendingAnalyzer")

    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV or generate synthetic data if file not found.

        Returns:
            DataFrame with post data
        """
        if self.data_path and Path(self.data_path).exists():
            logger.info(f"Loading data from {self.data_path}")
            self.df = pd.read_csv(self.data_path)
        else:
            logger.info("CSV not found, generating synthetic data...")
            self._generate_synthetic_data()

        logger.info(f"Loaded {len(self.df)} posts for analysis")
        return self.df

    def _generate_synthetic_data(self):
        """Generate synthetic Reddit data for analysis."""
        try:
            from sample_data import generate_dataset
            posts = generate_dataset(num_posts=350)
            self.df = pd.DataFrame(posts)
            logger.info(f"Generated synthetic dataset with {len(posts)} posts")
        except ImportError:
            logger.error("Could not import sample_data module")
            self.df = pd.DataFrame()

    def _extract_spending_amount(self, text: str) -> float:
        """
        Extract spending amount from text using regex patterns.

        Args:
            text: Post text to analyze

        Returns:
            Spending amount in dollars, or 0 if not found
        """
        text_lower = text.lower()
        amounts = []

        for pattern in self.SPENDING_PATTERNS:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            amounts.extend([float(m) for m in matches if m])

        # Return the maximum amount mentioned (most conservative estimate)
        return max(amounts) if amounts else 0.0

    def _categorize_model_type(self, text: str) -> str:
        """
        Categorize post as loot_box or battle_pass based on keywords.

        Args:
            text: Post text

        Returns:
            'loot_box', 'battle_pass', or 'mixed'
        """
        text_lower = text.lower()
        lootbox_count = sum(
            1 for keyword in self.LOOTBOX_KEYWORDS
            if keyword in text_lower
        )
        battlepass_count = sum(
            1 for keyword in self.BATTLEPASS_KEYWORDS
            if keyword in text_lower
        )

        if lootbox_count > battlepass_count:
            return 'loot_box'
        elif battlepass_count > lootbox_count:
            return 'battle_pass'
        else:
            return 'mixed'

    def _detect_behavioral_pattern(self, text: str) -> Tuple[str, bool, bool, bool]:
        """
        Detect behavioral economics patterns in text.

        Args:
            text: Post text

        Returns:
            Tuple of (primary_pattern, has_loss_aversion, has_fomo, has_sunk_cost)
        """
        text_lower = text.lower()

        loss_aversion = any(
            word in text_lower for word in self.LOSS_AVERSION_WORDS
        )
        fomo = any(word in text_lower for word in self.FOMO_WORDS)
        sunk_cost = any(word in text_lower for word in self.SUNK_COST_WORDS)

        # Determine primary pattern
        if loss_aversion and fomo:
            primary = 'loss_aversion_fomo'
        elif loss_aversion and sunk_cost:
            primary = 'loss_aversion_sunk_cost'
        elif fomo and sunk_cost:
            primary = 'fomo_sunk_cost'
        elif loss_aversion:
            primary = 'loss_aversion'
        elif fomo:
            primary = 'fomo'
        elif sunk_cost:
            primary = 'sunk_cost'
        else:
            primary = 'rational'

        return primary, loss_aversion, fomo, sunk_cost

    def _analyze_sentiment(self, text: str) -> Tuple[float, str]:
        """
        Analyze sentiment using VADER (Valence Aware Dictionary and sEntiment Reasoner).

        VADER is particularly effective for social media text and handles emoticons,
        slang, and informal language common in Reddit posts.

        Args:
            text: Post text

        Returns:
            Tuple of (compound_score, sentiment_label)
        """
        scores = self.sia.polarity_scores(text)
        compound = scores['compound']

        # Classify sentiment based on VADER thresholds
        if compound >= 0.05:
            label = 'positive'
        elif compound <= -0.05:
            label = 'negative'
        else:
            label = 'neutral'

        return compound, label

    def analyze_posts(self) -> pd.DataFrame:
        """
        Perform sentiment and behavioral analysis on all posts.

        Returns:
            DataFrame with analysis results
        """
        if self.df is None or self.df.empty:
            logger.error("No data loaded. Call load_data() first.")
            return pd.DataFrame()

        logger.info("Analyzing posts...")
        analyzed = []

        for idx, row in self.df.iterrows():
            text = str(row.get('text', ''))
            post_id = str(row.get('post_id', f'post_{idx}'))

            # Sentiment analysis
            sentiment_score, sentiment_label = self._analyze_sentiment(text)

            # Extract spending amount
            spending = self._extract_spending_amount(text)

            # Determine model type
            model_type = self._categorize_model_type(text)

            # Detect behavioral patterns
            pattern, loss_av, fomo, sunk_cost = self._detect_behavioral_pattern(text)

            post = SpendingPost(
                post_id=post_id,
                subreddit=str(row.get('subreddit', 'unknown')),
                text=text[:500],  # Truncate for storage
                model_type=model_type,
                behavioral_pattern=pattern,
                sentiment_score=sentiment_score,
                sentiment_label=sentiment_label,
                spending_amount=spending,
                has_loss_aversion=loss_av,
                has_fomo=fomo,
                has_sunk_cost=sunk_cost,
            )
            analyzed.append(post)

        self.analyzed_posts = analyzed
        logger.info(f"Analyzed {len(analyzed)} posts")

        # Convert to DataFrame for further analysis
        results_df = pd.DataFrame([
            {
                'post_id': p.post_id,
                'subreddit': p.subreddit,
                'model_type': p.model_type,
                'behavioral_pattern': p.behavioral_pattern,
                'sentiment_score': p.sentiment_score,
                'sentiment_label': p.sentiment_label,
                'spending_amount': p.spending_amount,
                'has_loss_aversion': p.has_loss_aversion,
                'has_fomo': p.has_fomo,
                'has_sunk_cost': p.has_sunk_cost,
            }
            for p in analyzed
        ])

        return results_df

    def compute_statistics(self, results_df: pd.DataFrame) -> Dict:
        """
        Compute statistical summaries and hypothesis tests.

        Tests thesis claim: Loss aversion correlates with loot box spending.

        Args:
            results_df: DataFrame with analyzed posts

        Returns:
            Dictionary of statistical results
        """
        stats_results = {}

        # Filter posts with spending amounts for correlation analysis
        spending_data = results_df[results_df['spending_amount'] > 0].copy()
        logger.info(f"Posts mentioning spending: {len(spending_data)}")

        # 1. Sentiment by model type
        sentiment_by_model = results_df.groupby('model_type')['sentiment_score'].agg([
            'mean', 'std', 'median', 'count'
        ])
        stats_results['sentiment_by_model'] = sentiment_by_model
        logger.info(f"\nSentiment by Model Type:\n{sentiment_by_model}")

        # 2. Spending amounts by model type
        if not spending_data.empty:
            spending_by_model = spending_data.groupby('model_type')['spending_amount'].agg([
                'mean', 'std', 'median', 'min', 'max', 'count'
            ])
            stats_results['spending_by_model'] = spending_by_model
            logger.info(f"\nSpending by Model Type:\n{spending_by_model}")

            # 3. Test correlation: loss aversion <-> spending amount
            if len(spending_data) > 2:
                correlation = spending_data['has_loss_aversion'].astype(int).corr(
                    spending_data['spending_amount']
                )
                stats_results['loss_aversion_spending_correlation'] = correlation

                # Perform t-test
                loss_av_spending = spending_data[
                    spending_data['has_loss_aversion']
                ]['spending_amount'].values
                no_loss_av_spending = spending_data[
                    ~spending_data['has_loss_aversion']
                ]['spending_amount'].values

                if len(loss_av_spending) > 0 and len(no_loss_av_spending) > 0:
                    if SCIPY_AVAILABLE:
                        t_stat, p_value = stats.ttest_ind(
                            loss_av_spending,
                            no_loss_av_spending
                        )
                    else:
                        t_stat, p_value = ttest_ind(
                            loss_av_spending,
                            no_loss_av_spending
                        )
                    stats_results['loss_aversion_ttest'] = {
                        'mean_loss_av': np.mean(loss_av_spending),
                        'mean_no_loss_av': np.mean(no_loss_av_spending),
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }

                logger.info(
                    f"\nLoss Aversion <-> Spending Correlation: {correlation:.4f}"
                )

            # 4. Behavioral pattern distribution
            pattern_dist = results_df['behavioral_pattern'].value_counts()
            stats_results['behavioral_patterns'] = pattern_dist
            logger.info(f"\nBehavioral Pattern Distribution:\n{pattern_dist}")

        # 5. Sentiment distribution
        sentiment_dist = results_df['sentiment_label'].value_counts()
        stats_results['sentiment_distribution'] = sentiment_dist
        logger.info(f"\nSentiment Distribution:\n{sentiment_dist}")

        # 6. Subreddit analysis
        subreddit_dist = results_df['subreddit'].value_counts()
        stats_results['subreddit_distribution'] = subreddit_dist
        logger.info(f"\nSubreddit Distribution:\n{subreddit_dist}")

        return stats_results

    def create_visualizations(self, results_df: pd.DataFrame, stats_results: Dict):
        """
        Generate publication-quality visualizations.

        Creates 5 plots: sentiment distribution, spending by model type,
        behavioral patterns, correlation heatmap, and subreddit breakdown.

        Args:
            results_df: DataFrame with analyzed posts
            stats_results: Dictionary of computed statistics
        """
        output_dir = Path(__file__).parent / 'visualizations'
        output_dir.mkdir(exist_ok=True)

        logger.info(f"Creating visualizations in {output_dir}")

        # 1. Sentiment Distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        sentiment_counts = results_df['sentiment_label'].value_counts()
        colors = ['#2ecc71', '#95a5a6', '#e74c3c']
        sentiment_counts.plot(kind='bar', ax=ax, color=colors, edgecolor='black', linewidth=1.2)
        ax.set_title('Sentiment Distribution in Gaming Spending Discussions', fontsize=14, fontweight='bold')
        ax.set_xlabel('Sentiment', fontsize=12)
        ax.set_ylabel('Number of Posts', fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container)

        plt.tight_layout()
        plt.savefig(output_dir / 'sentiment_distribution.png', dpi=300, bbox_inches='tight')
        logger.info("Saved: sentiment_distribution.png")
        plt.close()

        # 2. Spending by Model Type
        spending_data = results_df[results_df['spending_amount'] > 0].copy()
        if not spending_data.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            model_types = spending_data['model_type'].unique()

            spending_by_model = []
            for model in model_types:
                model_data = spending_data[spending_data['model_type'] == model]['spending_amount']
                spending_by_model.append(model_data.values)

            bp = ax.boxplot(spending_by_model, labels=model_types, patch_artist=True)

            for patch, color in zip(bp['boxes'], ['#3498db', '#e67e22', '#9b59b6']):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.set_title('Reported Spending Amounts by Monetization Model', fontsize=14, fontweight='bold')
            ax.set_ylabel('Spending Amount ($USD)', fontsize=12)
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / 'spending_by_model.png', dpi=300, bbox_inches='tight')
            logger.info("Saved: spending_by_model.png")
            plt.close()

        # 3. Behavioral Pattern Distribution
        fig, ax = plt.subplots(figsize=(12, 6))
        pattern_counts = results_df['behavioral_pattern'].value_counts().head(8)
        pattern_counts.plot(kind='barh', ax=ax, color='#16a085', edgecolor='black', linewidth=1.2)
        ax.set_title('Behavioral Economics Patterns in Spending Posts', fontsize=14, fontweight='bold')
        ax.set_xlabel('Number of Posts', fontsize=12)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)

        for container in ax.containers:
            ax.bar_label(container)

        plt.tight_layout()
        plt.savefig(output_dir / 'behavioral_patterns.png', dpi=300, bbox_inches='tight')
        logger.info("Saved: behavioral_patterns.png")
        plt.close()

        # 4. Loss Aversion and Spending Scatter
        if not spending_data.empty and len(spending_data) > 10:
            fig, ax = plt.subplots(figsize=(11, 7))

            # Separate by loss aversion
            loss_av = spending_data[spending_data['has_loss_aversion']]
            no_loss_av = spending_data[~spending_data['has_loss_aversion']]

            ax.scatter(
                range(len(loss_av)), loss_av['spending_amount'].values,
                alpha=0.6, s=80, color='#e74c3c', label='Loss Aversion Language',
                edgecolors='black', linewidth=0.5
            )
            ax.scatter(
                range(len(loss_av), len(loss_av) + len(no_loss_av)),
                no_loss_av['spending_amount'].values,
                alpha=0.6, s=80, color='#3498db', label='No Loss Aversion',
                edgecolors='black', linewidth=0.5
            )

            ax.set_title('Loss Aversion Language vs. Reported Spending', fontsize=14, fontweight='bold')
            ax.set_ylabel('Spending Amount ($USD)', fontsize=12)
            ax.set_xlabel('Posts (sorted by language type)', fontsize=12)
            ax.legend(fontsize=11, loc='upper right')
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / 'loss_aversion_spending.png', dpi=300, bbox_inches='tight')
            logger.info("Saved: loss_aversion_spending.png")
            plt.close()

        # 5. Sentiment by Model Type
        fig, ax = plt.subplots(figsize=(12, 6))
        model_types = results_df['model_type'].unique()

        sentiment_data = []
        positions = []
        labels = []

        for i, model in enumerate(model_types):
            model_df = results_df[results_df['model_type'] == model]
            sentiment_data.append(model_df['sentiment_score'].values)
            positions.append(i)
            labels.append(model)

        bp = ax.boxplot(sentiment_data, positions=positions, labels=labels, patch_artist=True)

        colors = ['#3498db', '#e67e22', '#9b59b6']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_title('Sentiment Score Distribution by Monetization Model', fontsize=14, fontweight='bold')
        ax.set_ylabel('VADER Sentiment Score (compound)', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'sentiment_by_model.png', dpi=300, bbox_inches='tight')
        logger.info("Saved: sentiment_by_model.png")
        plt.close()

        logger.info(f"\nAll visualizations saved to {output_dir}")

    def save_results_csv(self, results_df: pd.DataFrame, filename: str = 'analysis_results.csv'):
        """
        Save analysis results to CSV for further examination.

        Args:
            results_df: DataFrame with analysis results
            filename: Output filename
        """
        output_path = Path(__file__).parent / filename
        results_df.to_csv(output_path, index=False)
        logger.info(f"Saved detailed results to {output_path}")


def main():
    """Run complete analysis pipeline."""
    # Initialize analyzer
    analyzer = GamingSpendingAnalyzer(data_path='synthetic_gaming_data.csv')

    # Load data
    analyzer.load_data()

    # Analyze posts
    results_df = analyzer.analyze_posts()

    # Compute statistics
    stats_results = analyzer.compute_statistics(results_df)

    # Create visualizations
    analyzer.create_visualizations(results_df, stats_results)

    # Save results
    analyzer.save_results_csv(results_df)

    # Print summary
    print("\n" + "="*70)
    print("GAMING SPENDING SENTIMENT ANALYSIS - SUMMARY REPORT")
    print("="*70)
    print(f"\nTotal posts analyzed: {len(results_df)}")
    print(f"Posts mentioning spending: {len(results_df[results_df['spending_amount'] > 0])}")
    print(f"\nSentiment Distribution:")
    print(results_df['sentiment_label'].value_counts())
    print(f"\nMonetization Model Distribution:")
    print(results_df['model_type'].value_counts())
    print(f"\nKey Findings:")
    if 'loss_aversion_ttest' in stats_results:
        test_results = stats_results['loss_aversion_ttest']
        print(f"  - Mean spending (Loss Aversion): ${test_results['mean_loss_av']:.2f}")
        print(f"  - Mean spending (No Loss Aversion): ${test_results['mean_no_loss_av']:.2f}")
        print(f"  - T-test p-value: {test_results['p_value']:.4f} "
              f"({'SIGNIFICANT' if test_results['significant'] else 'NOT SIGNIFICANT'})")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
