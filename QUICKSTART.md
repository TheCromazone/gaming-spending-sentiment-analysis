# Quick start: offline synthetic demonstration

Use the pinned environment with Python 3.9, the version used for the installation and pipeline checks. Newer Python versions may need dependency updates.

```bash
git clone https://github.com/TheCromazone/gaming-spending-sentiment-analysis.git
cd gaming-spending-sentiment-analysis
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python sample_data.py
python analyzer.py
```

If the repository is already cloned, start with environment creation. On Windows, activate with `.venv\Scripts\Activate.ps1`.

The default run needs no Reddit credentials. It creates these local, Git-ignored outputs:

| Output | Contents |
| --- | --- |
| `synthetic_gaming_data.csv` | 350 generated posts |
| `analysis_results.csv` | Sentiment, spending extraction, and behavioral labels |
| `visualizations/` | Five PNG charts for the default sample |

Run from the repository root so the command-line entry point finds the generated CSV. `python sample_data.py` replaces that CSV with a new random sample. Results and charts are also overwritten by later runs.

## Interpret the output

The templates and random sampling rules create the demonstration data. A correlation or small p-value on those posts does **not** establish a real-world effect or validate the thesis. Numerical examples in older reports describe particular synthetic runs and will not necessarily match a new run.

The analyzer uses NLTK VADER if its lexicon is installed. Otherwise it logs a warning and uses the bundled lightweight sentiment fallback. To opt into VADER, download its data once:

```bash
python -c "import nltk; nltk.download('vader_lexicon')"
```

Record the scoring method when comparing results; the fallback and VADER are different analyzers.

## Verify without network access

```bash
python -m unittest discover -s tests -v
```

The tests cover positive and negative fallback scoring, including the case where NLTK is installed but its lexicon is missing. The offline pipeline was also checked with 350 generated rows, an analysis CSV, and five chart files. That checks execution, not the validity of an economic conclusion.

For a headless environment, run charts with `MPLBACKEND=Agg python analyzer.py` on macOS/Linux.

## Analyze collected data

Follow [Using Real Reddit Data](README.md#using-real-reddit-data). The scraper is a separate network operation requiring configured Reddit API access.

Pass `reddit_gaming_spending_data.csv` explicitly to `GamingSpendingAnalyzer(data_path=...)`. Running `python analyzer.py` alone still selects the synthetic file. A missing CSV causes a synthetic fallback, so verify the requested file and the log before interpreting a run as collected data.

## Troubleshooting

- **Missing package:** activate the environment and rerun `python -m pip install -r requirements.txt`.
- **Installation fails on a new Python release:** use the verified Python 3.9 path, or review and test dependency updates in a separate environment.
- **Unexpected results:** confirm the input filename and analyzer choice in the log. Generated samples vary between runs.
- **Missing charts:** check the error output and the writable `visualizations/` directory, which is created automatically.
