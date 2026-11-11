# Gaming Spending Sentiment Analysis

A data-science pipeline for exploring behavioral economics hypotheses about loot boxes and battle passes through spending discussions and sentiment analysis.

**Default runs use generated synthetic posts.** These exercise the pipeline and charts; their correlations and p-values are not evidence about real players or validation of the thesis. No collected Reddit dataset is included.

## Overview

This project analyzes self-reported gaming spending patterns from Reddit communities to investigate behavioral economics hypotheses about monetization models. The core research question is rooted in the Reed College Economics thesis "The Transition from Loot Boxes to Battle Passes" (Matthew Cromaz, 2023), which proposes that:

1. **Loot boxes** trigger loss aversion and regret due to randomized outcomes
2. **Battle passes** enable more rational spending decisions due to predictable, guaranteed rewards
3. Loss aversion language correlates with higher reported spending on loot boxes

The code can analyze a supplied Reddit CSV with sentiment analysis and behavioral pattern detection. Empirical testing requires a separately collected, documented dataset; the bundled generator is a demonstration fixture.

## Theoretical Framework

### Behavioral Economics Concepts

**Loss Aversion (Kahneman & Tversky, 1979)**
- Players feel the pain of losses approximately twice as intensely as equivalent gains
- Random loot boxes create frequent "losses" (duplicate pulls, bad luck)
- Expected utility is negative, yet players persist due to loss aversion and the gambler's fallacy

**FOMO (Fear of Missing Out)**
- Time-limited cosmetics and seasonal content pressure immediate spending
- Players experience losses (missed exclusive items) more acutely than equivalent gains
- Both loot boxes and battle passes exploit this, but differently

**Sunk Cost Fallacy**
- Past spending justifies future spending ("I've already spent $100, might as well continue")
- More prevalent in randomized systems where players chase past investments
- Combined with loss aversion to create spending cycles

**Rational Spending**
- Battle pass pricing is transparent and predictable (~$9.99-19.99/season)
- Guaranteed rewards reduce loss aversion triggers
- Players can calculate cost-per-item rationally

### Thesis Hypothesis

Players discussing loot boxes show higher loss aversion language, which correlates with larger reported spending amounts. This reflects the psychological mechanisms driving players to "chase" losses in randomized systems. Battle pass discussions show more rational framing.

## Project Structure

```
gaming-spending-sentiment-analysis/
├── scraper.py              # Reddit PRAW scraper with rate limiting
├── analyzer.py             # Main analysis engine with VADER sentiment analysis
├── sample_data.py          # Synthetic data generator (350+ posts)
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── synthetic_gaming_data.csv     # Generated synthetic dataset
├── analysis_results.csv         # Detailed analysis output
└── visualizations/        # Output directory for plots
    ├── sentiment_distribution.png
    ├── spending_by_model.png
    ├── behavioral_patterns.png
    ├── loss_aversion_spending.png
    └── sentiment_by_model.png
```

## Installation

### Requirements
- Python and pip. The pinned NumPy/pandas/SciPy stack was installed and checked with Python 3.9; newer Python releases may require dependency updates.

### Setup

1. Clone the project, or start in an existing checkout:
```bash
git clone https://github.com/TheCromazone/gaming-spending-sentiment-analysis.git
cd gaming-spending-sentiment-analysis
```

2. Create an environment and install the pinned dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\Activate.ps1` instead.

3. Optionally download the VADER lexicon:
```bash
python -c "import nltk; nltk.download('vader_lexicon')"
```

If NLTK or its lexicon is unavailable, the analyzer logs that it is using its lightweight keyword fallback. This avoids a required runtime download, but the fallback is a different scoring method; record which analyzer produced research outputs.

## Usage

### Quick Start: Analyze Synthetic Data

To run analysis with automatically-generated synthetic Reddit data:

```bash
# Generate synthetic dataset (if not already created)
python sample_data.py

# Run complete analysis
python analyzer.py
```

This generates:
- 350+ realistic Reddit posts with behavioral patterns
- Sentiment analysis using VADER when its lexicon is installed, otherwise the lightweight fallback
- Statistical correlations and hypothesis tests
- 5 publication-quality visualizations
- CSV file with detailed results

### Using Real Reddit Data

To scrape real data from Reddit:

1. **Configure Reddit API access**:
   - Visit https://www.reddit.com/prefs/apps
   - Create a "script" application
   - Note your `client_id` and `client_secret`

2. **Scrape posts**:
```bash
export REDDIT_CLIENT_ID="your_client_id"
export REDDIT_CLIENT_SECRET="your_client_secret"
python scraper.py
```

3. **Analyze the scraper's CSV explicitly**:
```python
from analyzer import GamingSpendingAnalyzer

analyzer = GamingSpendingAnalyzer(data_path="reddit_gaming_spending_data.csv")
analyzer.load_data()
results = analyzer.analyze_posts()
statistics = analyzer.compute_statistics(results)
analyzer.create_visualizations(results, statistics)
analyzer.save_results_csv(results)
```

Run this snippet from the repository root after verifying the CSV exists. `python analyzer.py` selects `synthetic_gaming_data.csv`, and `load_data()` generates synthetic data if its requested file is missing. Do not treat a missing-input fallback as a successful real-data run. For an existing CSV, include a `text` column; `post_id` and `subreddit` preserve source identifiers.

The scraper is a separate, opt-in network operation and was not used in the offline checks. Keep credentials in your environment. The default scraper CSV, generated demo CSV, analysis results, and chart directory are ignored by Git; custom input filenames need their own ignore entry or storage outside the checkout.

The scraper fetches posts from:
- **r/FIFA** - FIFA Ultimate Team (pack-based loot boxes)
- **r/NBA2k** - MyTeam (loot box heavy)
- **r/FortNiteBR** - Battle passes + cosmetics
- **r/GlobalOffensive** - Case system (loot boxes)

## Methodology

### Data Collection

**Scraper** (`scraper.py`):
- Uses PRAW library for authenticated Reddit API access
- Searches each target subreddit for spending-related keywords
- Implements rate limiting (2s between requests) to respect Reddit policies
- Captures: post ID, subreddit, title, body text, score, comments, timestamp
- Error handling for network issues and API limits

**Synthetic Data** (`sample_data.py`):
- Generates 350+ realistic Reddit posts matching real discussion patterns
- Includes behavioral patterns with weighted probabilities:
  - Loss aversion (25%)
  - FOMO (20%)
  - Sunk cost fallacy (20%)
  - Rational spending (35%)
- Realistic spending amounts ($5-$1000)
- Natural language patterns mimicking actual Reddit discussions

### Analysis

**Sentiment Analysis** (`analyzer.py`):

VADER (Valence Aware Dictionary and sEntiment Reasoner) is used for sentiment analysis because:
- Designed specifically for social media text and internet slang
- Handles emoticons, sarcasm, and informal language effectively
- Provides compound sentiment score (-1 = most negative, +1 = most positive)
- Thresholds: compound >= 0.05 (positive), <= -0.05 (negative), else neutral

**Behavioral Pattern Detection**:

Posts are categorized for four key behavioral economics patterns:

1. **Loss Aversion**: Words like "regret", "wasted", "lost", "mistake" indicate players experiencing or recounting losses
2. **FOMO**: Words like "limited", "time-limited", "exclusive", "miss out" indicate temporal scarcity pressure
3. **Sunk Cost**: Phrases like "already spent", "committed", "can't stop", "might as well" indicate past spending justifying future spending
4. **Rational**: Absence of the above; cost-benefit analysis of spending

**Monetization Model Classification**:

Posts are categorized as discussing loot boxes or battle passes based on keyword density:
- **Loot Box Keywords**: "pack", "rng", "odds", "gambling", "duplicate", "case", "FIFA Points", "VC"
- **Battle Pass Keywords**: "battle pass", "seasonal", "tier", "guaranteed", "grind", "challenge"

**Spending Amount Extraction**:

Regex patterns extract mentioned spending amounts:
- Direct: "$100", "$99.99"
- Contextual: "spent 100", "100 dollars"
- Currency variations: "£100"

### Statistical Tests

**Hypothesis Testing**:
- Tests correlation between loss aversion language and reported spending amounts
- Two-sample t-test comparing spending: posts with vs without loss aversion language
- Tests null hypothesis: no difference in spending between groups
- Significance threshold: p < 0.05

## Key Findings

### Expected Results (from Thesis Hypothesis)

Based on the behavioral economics framework:

1. **Sentiment Differences**:
   - Loot box posts should show more negative sentiment (regret, loss aversion)
   - Battle pass posts should show more neutral/positive sentiment (value framing)

2. **Spending Correlation**:
   - Posts with loss aversion language should report higher spending (chasing losses)
   - Battle pass discussions should show lower spending amounts with more predictable spending patterns

3. **Behavioral Pattern Distribution**:
   - Loot boxes: Heavy loss aversion and FOMO language
   - Battle passes: More rational spending justifications

4. **Model Type Impact**:
   - Loot box discussions: Higher variability in spending (driven by randomness)
   - Battle pass discussions: More consistent spending (predictable pricing)

### Example Output

The following is illustrative output, not a reproducible benchmark. The generator is random, so counts, spending means, and significance change between runs:

```
======================================================================
GAMING SPENDING SENTIMENT ANALYSIS - SUMMARY REPORT
======================================================================

Total posts analyzed: 350
Posts mentioning spending: 287

Sentiment Distribution:
negative      145
neutral       102
positive       58

Monetization Model Distribution:
loot_box       189
battle_pass    112
mixed           49

Key Findings:
  - Mean spending (Loss Aversion): $156.42
  - Mean spending (No Loss Aversion): $98.73
  - T-test p-value: 0.0234 (SIGNIFICANT)
======================================================================
```

## Visualizations

The analysis generates 5 publication-quality charts:

1. **sentiment_distribution.png**: Distribution of positive/neutral/negative sentiment across all posts
2. **spending_by_model.png**: Box plot comparing spending amounts by monetization model
3. **behavioral_patterns.png**: Horizontal bar chart of behavioral pattern frequencies
4. **loss_aversion_spending.png**: Scatter plot showing spending amounts grouped by loss aversion language presence
5. **sentiment_by_model.png**: Box plot comparing sentiment scores between loot boxes and battle passes

All charts are saved as high-resolution PNG files (300 DPI) suitable for publication.

## Data Files

### Input
- `synthetic_gaming_data.csv` - Generated by `sample_data.py` (350+ posts)
- `reddit_gaming_spending_data.csv` - Output from real scraping (if using Reddit API)

### Output
- `analysis_results.csv` - Detailed results for each post including all analysis fields
- `visualizations/*.png` - 5 publication-quality charts

## Advanced Usage

### Custom Analysis

Modify `analyzer.py` to:
- Add additional behavioral markers to `LOSS_AVERSION_WORDS`, `FOMO_WORDS`, etc.
- Change sentiment classification thresholds (currently 0.05/-0.05)
- Add new statistical tests (chi-square, ANOVA, regression)
- Filter data by subreddit, time period, or spending range

### Extending the Scraper

The scraper can be extended to:
- Collect comments (currently only posts)
- Track temporal patterns (spending over time)
- Analyze user history and repeat spending
- Include additional subreddits (r/ValorAnt, r/Genshin, r/ApexLegends, etc.)

### Data Persistence

To avoid re-scraping:
```python
from analyzer import GamingSpendingAnalyzer

analyzer = GamingSpendingAnalyzer(data_path='reddit_gaming_spending_data.csv')
data = analyzer.load_data()
```

## Validation & Limitations

### Strengths
- VADER sentiment analysis validated on social media text
- Behavioral pattern detection based on established economics literature
- Synthetic data enables offline pipeline testing without Reddit credentials
- Statistical tests summarize the selected input; they do not verify a hypothesis on their own

### Limitations
- **Self-report bias**: Spending amounts are self-reported, not verified
- **Selection bias**: Reddit users may not represent all gaming populations
- **Temporal bias**: Recent posts may reflect current trends (seasonal passes vs loot boxes)
- **Sentiment analysis limitations**: VADER doesn't capture context-dependent irony
- **Causality**: Correlations don't prove causation between loss aversion language and spending
- **Synthetic data**: Templates and sampling rules are assumptions, not observations. Results from generated posts cannot establish population effects

### Recommendations for Production Use
1. Validate spending amounts against in-game transaction records (with user consent)
2. Include demographic information to control for age, experience level, income
3. Conduct longitudinal analysis tracking individuals over time
4. Use multiple sentiment analysis methods (TextBlob, transformer models) for robustness
5. Control for game popularity cycles (new releases, seasonal events)

## Connection to Original Thesis

This project operationalizes the behavioral economics theories presented in "The Transition from Loot Boxes to Battle Passes" (Cromaz, 2023) by:

1. **Preparing hypothesis tests**: Quantifying the relationship between loss aversion language and reported spending in a supplied dataset
2. **Operationalizing concepts**: Converting abstract behavioral economics concepts into measurable linguistic patterns
3. **Separating evidence from fixtures**: Real-data claims require collected Reddit discussions and a documented sampling method; synthetic demonstrations do not provide that evidence
4. **Creating reproducible analysis**: Enabling others to extend this methodology to new data or games

The analysis demonstrates that behavioral economics concepts (loss aversion, FOMO, sunk cost) can be measured through text analysis, providing a bridge between economic theory and data science methods.

## Future Research Directions

1. **Sentiment evolution**: Track how individual players' sentiment changes as spending increases
2. **Peer effects**: Analyze whether spending discussions create FOMO in readers
3. **Developer impact**: Correlate game changes (loot box odds, pass pricing) with spending sentiment
4. **Cross-game comparison**: Compare behavioral patterns across different gaming genres
5. **Intervention testing**: Model impact of transparency policies (disclosing odds, spending limits)
6. **Longitudinal tracking**: Follow individual Reddit users to measure causal spending effects

## Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib**: Visualization
- **seaborn**: Statistical data visualization
- **nltk**: Natural Language Toolkit (VADER sentiment)
- **textblob**: Simplified NLP (alternative sentiment analysis)
- **praw**: Python Reddit API Wrapper (for scraping)
- **requests**: HTTP library (for fallback API calls)
- **scikit-learn**: Machine learning (data preprocessing)
- **scipy**: Scientific computing (statistical tests)

## License

This project is provided as a portfolio piece demonstrating data science and behavioral economics knowledge. Use freely for educational and research purposes.

## Author Notes

This project demonstrates:
- **Data Collection**: Building production-grade scrapers with rate limiting and error handling
- **Sentiment Analysis**: Applying NLP to real-world behavioral economics questions
- **Statistical Analysis**: Hypothesis testing and correlation analysis
- **Data Visualization**: Creating publication-quality charts
- **Code Quality**: Production-grade Python with logging, type hints, and documentation
- **Research Integration**: Connecting behavioral economics theory to empirical data science

The synthetic data generator is particularly useful because it:
- Provides privacy-respecting test data; set a random seed when a repeatable fixture is needed
- Mimics real behavioral patterns from gaming communities
- Enables local testing without API credentials
- Serves as a template for other behavioral economics projects

This represents a bridge between academic economics research and practical data science, demonstrating how theoretical frameworks can be operationalized through computational analysis.

## Contact & Questions

For questions about the methodology, implementation, or extension of this project, refer to the inline documentation in each Python file. The code is extensively commented to explain behavioral economics concepts and technical implementations.
