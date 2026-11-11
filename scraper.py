"""
Reddit scraper for gaming spending data.

This module scrapes Reddit posts from major gaming subreddits about spending,
loot boxes, and battle passes using the PRAW API. Implements rate limiting,
error handling, and CSV export functionality.

Note: Requires valid PRAW credentials (client_id, client_secret, user_agent).
For testing without credentials, use the sample_data module instead.
"""

import praw
import csv
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import requests


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RedditPost:
    """Data class representing a Reddit post."""
    post_id: str
    subreddit: str
    title: str
    text: str
    created_utc: str
    score: int
    num_comments: int
    url: str
    author: str


class RedditGamingSpendingScraper:
    """
    Scrapes Reddit posts about gaming spending from multiple subreddits.

    Uses PRAW for Reddit API access and implements rate limiting to respect
    Reddit's API policies. Posts are filtered for content about spending,
    loot boxes, and battle passes.
    """

    # Target subreddits with games known for spending discussions
    TARGET_SUBREDDITS = [
        'FIFA',        # FIFA Ultimate Team - heavy loot box focus
        'NBA2k',       # MyTeam - loot box heavy
        'FortNiteBR',  # Battle passes + cosmetics
        'GlobalOffensive'  # Cases and cosmetics
    ]

    # Keywords that indicate spending/monetization discussion
    SPENDING_KEYWORDS = [
        'spend', 'spent', 'cost', 'price', 'money', 'dollar', 'loot box',
        'battle pass', 'pass', 'season pass', 'cosmetic', 'skin',
        'pack', 'case', 'key', 'gambling', 'rng', 'odds', 'addiction',
        'pay-to-win', 'p2w', 'vbucks', 'fifa points', 'vc'
    ]

    # Rate limiting parameters
    REQUEST_DELAY_SECONDS = 2  # Delay between API requests

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent: Optional[str] = None
    ):
        """
        Initialize the scraper.

        Args:
            client_id: Reddit app client ID
            client_secret: Reddit app client secret
            user_agent: User agent string for API requests
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent or 'GamingSpendingAnalyzer/1.0'
        self.reddit = None
        self.posts = []
        self.last_request_time = 0

    def authenticate(self) -> bool:
        """
        Authenticate with Reddit API using PRAW.

        Returns:
            True if authentication successful, False otherwise
        """
        try:
            if not all([self.client_id, self.client_secret]):
                logger.warning(
                    "PRAW credentials not provided. "
                    "Use sample_data.py for synthetic data instead."
                )
                return False

            self.reddit = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent=self.user_agent
            )

            # Test authentication
            _ = self.reddit.user.me()
            logger.info("Successfully authenticated with Reddit API")
            return True

        except Exception as e:
            logger.error(f"Reddit authentication failed: {e}")
            return False

    def _apply_rate_limit(self):
        """Apply rate limiting between API requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.REQUEST_DELAY_SECONDS:
            time.sleep(self.REQUEST_DELAY_SECONDS - elapsed)
        self.last_request_time = time.time()

    def _is_spending_post(self, post) -> bool:
        """
        Check if post content relates to spending/monetization.

        Args:
            post: PRAW submission object

        Returns:
            True if post discusses spending, False otherwise
        """
        combined_text = (post.title + " " + post.selftext).lower()
        return any(keyword in combined_text for keyword in self.SPENDING_KEYWORDS)

    def scrape_subreddit(self, subreddit_name: str, limit: int = 500) -> List[RedditPost]:
        """
        Scrape posts from a specific subreddit.

        Args:
            subreddit_name: Name of subreddit to scrape
            limit: Maximum number of posts to retrieve

        Returns:
            List of RedditPost objects
        """
        if not self.reddit:
            logger.error("Not authenticated. Call authenticate() first.")
            return []

        subreddit_posts = []
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            logger.info(f"Scraping r/{subreddit_name}...")

            # Search for spending-related posts
            for post in subreddit.search(
                ' OR '.join(self.SPENDING_KEYWORDS[:5]),
                sort='new',
                time_filter='month',
                limit=limit
            ):
                self._apply_rate_limit()

                if self._is_spending_post(post):
                    reddit_post = RedditPost(
                        post_id=post.id,
                        subreddit=post.subreddit.display_name,
                        title=post.title,
                        text=post.selftext,
                        created_utc=datetime.fromtimestamp(
                            post.created_utc
                        ).isoformat(),
                        score=post.score,
                        num_comments=post.num_comments,
                        url=post.url,
                        author=post.author.name if post.author else '[deleted]'
                    )
                    subreddit_posts.append(reddit_post)
                    logger.debug(f"Collected post: {post.title[:60]}...")

            logger.info(f"Scraped {len(subreddit_posts)} posts from r/{subreddit_name}")

        except Exception as e:
            logger.error(f"Error scraping r/{subreddit_name}: {e}")

        return subreddit_posts

    def scrape_all_subreddits(self, limit: int = 500) -> List[RedditPost]:
        """
        Scrape posts from all target subreddits.

        Args:
            limit: Maximum number of posts per subreddit

        Returns:
            List of all collected RedditPost objects
        """
        all_posts = []
        for subreddit in self.TARGET_SUBREDDITS:
            posts = self.scrape_subreddit(subreddit, limit=limit)
            all_posts.extend(posts)
            time.sleep(1)  # Extra delay between subreddits

        logger.info(f"Total posts collected: {len(all_posts)}")
        self.posts = all_posts
        return all_posts

    def save_to_csv(self, filename: str = 'reddit_gaming_spending_data.csv') -> Path:
        """
        Save scraped posts to CSV file.

        Args:
            filename: Output CSV filename

        Returns:
            Path to saved CSV file
        """
        output_path = Path(__file__).parent / filename

        if not self.posts:
            logger.warning("No posts to save")
            return output_path

        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=RedditPost.__dataclass_fields__.keys())
                writer.writeheader()
                writer.writerows([asdict(post) for post in self.posts])

            logger.info(f"Saved {len(self.posts)} posts to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error saving to CSV: {e}")
            raise

    @staticmethod
    def load_from_csv(filename: str) -> List[Dict]:
        """
        Load previously scraped data from CSV.

        Args:
            filename: CSV filename to load

        Returns:
            List of post dictionaries
        """
        csv_path = Path(__file__).parent / filename
        posts = []

        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                posts = list(reader)
            logger.info(f"Loaded {len(posts)} posts from {csv_path}")
        except FileNotFoundError:
            logger.warning(f"CSV file not found: {csv_path}")

        return posts


def main():
    """
    Example usage: scrape Reddit and save data.

    Note: Requires PRAW credentials. Set these via environment variables:
    REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, or pass to __init__
    """
    import os

    # Get credentials from environment (or set directly)
    client_id = os.getenv('REDDIT_CLIENT_ID')
    client_secret = os.getenv('REDDIT_CLIENT_SECRET')

    scraper = RedditGamingSpendingScraper(
        client_id=client_id,
        client_secret=client_secret
    )

    if scraper.authenticate():
        posts = scraper.scrape_all_subreddits(limit=500)
        scraper.save_to_csv()
        print(f"\nSuccessfully scraped {len(posts)} posts")
    else:
        print(
            "\nNo PRAW credentials provided. "
            "Run sample_data.py to generate synthetic data instead:\n"
            "  python sample_data.py"
        )


if __name__ == '__main__':
    main()
