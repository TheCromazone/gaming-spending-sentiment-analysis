"""
Generates realistic synthetic Reddit data for gaming spending analysis.

This module creates synthetic posts that mimic real discussions from gaming subreddits
about spending patterns, loot boxes, and battle passes. Posts include realistic
behavioral economics patterns: loss aversion language, FOMO triggers, sunk cost fallacy,
and anchor pricing effects.
"""

import random
import csv
from datetime import datetime, timedelta
from pathlib import Path


# Behavioral patterns commonly found in gaming spending discussions
LOSS_AVERSION_PATTERNS = [
    "I regret spending ${amount} on loot boxes, should have saved it",
    "Wasted ${amount} trying to get that one card",
    "Lost ${amount} to bad pack luck this season",
    "I'm down ${amount} from duplicate pulls",
    "Threw away ${amount} chasing that elite player",
    "Already spent ${amount} and still don't have the good items",
    "Can't believe I dropped ${amount} on this game",
    "My ${amount} investment didn't pay off at all",
]

FOMO_PATTERNS = [
    "Had to spend ${amount} before the event ended, limited time offer",
    "Everyone had the pass so I spent ${amount} to keep up",
    "Limited edition drops forced me to spend ${amount} immediately",
    "Felt pressured to spend ${amount} or miss out on content",
    "The FOMO got me, spent ${amount} on the seasonal pass",
    "Time-limited items made me spend ${amount} I didn't plan to",
    "Missed out so I rage-spent ${amount} on the new set",
]

SUNK_COST_PATTERNS = [
    "Already spent ${amount}, might as well keep going",
    "I've put ${amount} into this already, can't stop now",
    "${amount} deep in this battle pass season, might as well finish it",
    "After spending ${amount}, I need to make it worth it",
    "Committed ${amount} to the system, not quitting",
    "With ${amount} invested, I have to see this through",
]

RATIONAL_SPENDING_PATTERNS = [
    "Spent ${amount} on battle pass, worth the value since I play daily",
    "Fair price point at ${amount} for seasonal content",
    "${amount} is reasonable for a battle pass with 100+ items",
    "The battle pass at ${amount} breaks down to cents per hour played",
    "Budgeted ${amount} monthly for cosmetics, within limits",
    "Value justify ${amount} spending: ${hours} hours/month gameplay",
    "Battle pass is better value than loot boxes at ${amount}",
]

LOOTBOX_DISCUSSION = [
    "loot box system",
    "pack odds",
    "RNG nightmare",
    "gambling mechanics",
    "randomized rewards",
    "duplicate protection needed",
    "pay-to-win loot boxes",
    "loot box addiction",
]

BATTLEPASS_DISCUSSION = [
    "battle pass season",
    "seasonal pass",
    "pass rewards",
    "guaranteed rewards",
    "battle pass grind",
    "seasonal unlock system",
    "predictable rewards",
    "pass value",
]

GAME_SUBREDDITS = ["r/FIFA", "r/NBA2k", "r/FortNiteBR", "r/GlobalOffensive"]

SPENDING_AMOUNTS = [
    5, 9.99, 10, 15, 19.99, 20, 25, 29.99, 30, 40, 49.99, 50,
    75, 99.99, 100, 150, 199.99, 200, 250, 300, 500, 750, 1000
]


def generate_post(post_id):
    """Generate a single synthetic Reddit post about gaming spending."""

    # Randomly choose game/subreddit
    subreddit = random.choice(GAME_SUBREDDITS)

    # Randomly choose spending model (loot box or battle pass focused)
    model_type = random.choice(["loot_box", "battle_pass"])

    # Randomly choose behavioral pattern with weighted probabilities
    behavior_choice = random.choices(
        ["loss_aversion", "fomo", "sunk_cost", "rational"],
        weights=[0.25, 0.20, 0.20, 0.35],
        k=1
    )[0]

    spending_amount = random.choice(SPENDING_AMOUNTS)

    # Generate the post text
    if behavior_choice == "loss_aversion":
        template = random.choice(LOSS_AVERSION_PATTERNS)
        text = template.format(amount=spending_amount)
    elif behavior_choice == "fomo":
        template = random.choice(FOMO_PATTERNS)
        text = template.format(amount=spending_amount)
    elif behavior_choice == "sunk_cost":
        template = random.choice(SUNK_COST_PATTERNS)
        text = template.format(amount=spending_amount)
    else:  # rational
        template = random.choice(RATIONAL_SPENDING_PATTERNS)
        text = template.format(amount=spending_amount, hours=random.randint(10, 100))

    # Add relevant terminology based on model type
    if model_type == "loot_box":
        text += " " + random.choice(LOOTBOX_DISCUSSION)
    else:
        text += " " + random.choice(BATTLEPASS_DISCUSSION)

    # Generate synthetic timestamp (past 6 months)
    days_ago = random.randint(0, 180)
    created_utc = (datetime.utcnow() - timedelta(days=days_ago)).isoformat()

    return {
        'post_id': f'synthetic_{post_id}',
        'subreddit': subreddit,
        'title': f'Spending discussion {post_id}',
        'text': text,
        'created_utc': created_utc,
        'score': random.randint(-5, 500),
        'num_comments': random.randint(0, 150),
        'model_type': model_type,
        'behavioral_pattern': behavior_choice,
        'spending_mentioned': spending_amount,
    }


def generate_dataset(num_posts=350):
    """Generate synthetic dataset of Reddit posts about gaming spending."""
    posts = [generate_post(i) for i in range(num_posts)]
    return posts


def save_dataset(posts, filename='synthetic_gaming_data.csv'):
    """Save generated posts to CSV file."""
    output_path = Path(__file__).parent / filename

    fieldnames = [
        'post_id', 'subreddit', 'title', 'text', 'created_utc',
        'score', 'num_comments', 'model_type', 'behavioral_pattern',
        'spending_mentioned'
    ]

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(posts)

    print(f"Saved {len(posts)} synthetic posts to {output_path}")
    return output_path


if __name__ == '__main__':
    posts = generate_dataset(num_posts=350)
    save_dataset(posts)
