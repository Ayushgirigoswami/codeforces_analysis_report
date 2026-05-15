#!/usr/bin/env python3
"""
Codeforces Division Contest Analyzer
------------------------------------
Fetches all Div 1, Div 2, Div 3, Div 4, and Div 1+Div 2 contests from Codeforces,
analyzes every problem from those contests, and generates interactive visualizations
and detailed analysis to uncover hidden patterns in problem ratings, difficulty
trends, topic distribution, and contest difficulty progression.

Author: AI Assistant
Date: 2026-05-15
"""

import requests
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from datetime import datetime
import time
import logging
import re
import math
import statistics
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import webbrowser
import sys
import os
from itertools import chain
import warnings
warnings.filterwarnings('ignore')

# Optional: tqdm for progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Create a dummy tqdm if not available
    class tqdm:
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable
            self.total = kwargs.get('total', None)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def update(self, n=1):
            pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# SECTION 1: API FETCHING FUNCTIONS
# =============================================================================

class CodeforcesAPI:
    """Handles all Codeforces API calls with rate limiting and error handling."""
    
    BASE_URL = "https://codeforces.com/api"
    RATE_LIMIT = 2  # seconds between requests to be safe
    
    def __init__(self):
        self.last_request_time = 0
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def _rate_limit(self):
        """Enforce rate limiting between API calls."""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.RATE_LIMIT:
            time.sleep(self.RATE_LIMIT - elapsed)
        self.last_request_time = time.time()
    
    def _fetch(self, endpoint: str, params: Dict = None) -> Dict:
        """Fetch data from a Codeforces API endpoint."""
        self._rate_limit()
        url = f"{self.BASE_URL}/{endpoint}"
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            if data.get('status') == 'OK':
                return data.get('result', [])
            else:
                logger.error(f"API error for {endpoint}: {data.get('comment', 'Unknown error')}")
                return []
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {endpoint}: {e}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for {endpoint}: {e}")
            return []
    
    def get_contest_list(self, gym: bool = False) -> List[Dict]:
        """Fetch list of all contests."""
        logger.info("Fetching contest list...")
        result = self._fetch("contest.list", params={"gym": "false" if not gym else "true"})
        logger.info(f"Fetched {len(result)} contests")
        return result
    
    def get_problemset(self) -> Dict:
        """Fetch all problems with their ratings and tags."""
        logger.info("Fetching problemset...")
        result = self._fetch("problemset.problems")
        if isinstance(result, dict) and 'problems' in result:
            logger.info(f"Fetched {len(result['problems'])} problems")
            return result
        elif isinstance(result, list):
            # Sometimes API returns directly a list? Handle both.
            logger.info(f"Fetched {len(result)} problems (list format)")
            return {'problems': result, 'problemStatistics': []}
        else:
            logger.warning("Unexpected problemset response format")
            return {'problems': [], 'problemStatistics': []}


def extract_division(contest_name: str) -> str:
    """
    Extract division from contest name using pattern matching.
    
    Patterns:
    - "Div. 1+Div. 2" or "Div1+Div2" or similar
    - "Div. 1" or "Div1"
    - "Div. 2" or "Div2"
    - "Div. 3" or "Div3"
    - "Div. 4" or "Div4"
    - Educational rounds are treated separately
    """
    name_clean = contest_name.upper()
    
    # Check for combined division first
    if re.search(r'DIV[\s.]*1[\s+&]*DIV[\s.]*2|DIV[\s.]*1\+DIV[\s.]*2|DIV1\+DIV2', name_clean):
        return "Div. 1+Div. 2"
    
    # Check for single divisions
    if re.search(r'DIV[\s.]*1\b', name_clean):
        return "Div. 1"
    elif re.search(r'DIV[\s.]*2\b', name_clean):
        return "Div. 2"
    elif re.search(r'DIV[\s.]*3\b', name_clean):
        return "Div. 3"
    elif re.search(r'DIV[\s.]*4\b', name_clean):
        return "Div. 4"
    elif "EDUCATIONAL" in name_clean:
        return "Educational"
    else:
        return "Unknown"


def is_rated_contest(contest: Dict) -> bool:
    """Determine if a contest is rated for participants."""
    # Div contests are typically rated for eligible participants
    phase = contest.get('phase', '')
    if phase not in ['FINISHED', 'CODING']:
        return False
    
    name = contest.get('name', '')
    division = extract_division(name)
    if division in ["Div. 1", "Div. 2", "Div. 3", "Div. 4", "Div. 1+Div. 2"]:
        return True
    # Also include Educational Rounds (rated for Div. 2 participants)
    if "EDUCATIONAL" in name.upper() and "ROUND" in name.upper():
        return True
    return False


def filter_contests_by_division(contests: List[Dict], target_divisions: List[str]) -> List[Dict]:
    """
    Filter contests to include only specified divisions.
    
    Args:
        contests: List of contest dictionaries from API
        target_divisions: List of division strings to include
    
    Returns:
        Filtered list of contests
    """
    filtered = []
    for contest in contests:
        division = extract_division(contest.get('name', ''))
        if division in target_divisions:
            contest['division'] = division
            filtered.append(contest)
    return filtered


def get_problem_rating_map(problemset: Dict) -> Dict:
    """
    Create a mapping from problem identifier (contestId + index) to rating.
    
    Args:
        problemset: Dictionary containing problems list
    
    Returns:
        Dictionary with key f"{contestId}{index}" -> rating
    """
    rating_map = {}
    problems = problemset.get('problems', [])
    
    for problem in problems:
        contest_id = problem.get('contestId')
        index = problem.get('index')
        rating = problem.get('rating')
        if contest_id and index:
            key = f"{contest_id}{index}"
            rating_map[key] = rating if rating is not None else 0
    
    logger.info(f"Problem rating map created with {len(rating_map)} entries")
    return rating_map


def get_problem_tags_map(problemset: Dict) -> Dict:
    """Create mapping from problem identifier to its tags."""
    tags_map = {}
    problems = problemset.get('problems', [])
    
    for problem in problems:
        contest_id = problem.get('contestId')
        index = problem.get('index')
        tags = problem.get('tags', [])
        if contest_id and index:
            key = f"{contest_id}{index}"
            tags_map[key] = tags
    
    return tags_map


def fetch_all_relevant_data() -> Tuple[pd.DataFrame, Dict, Dict]:
    """
    Main function to fetch and process all contest data.
    
    Returns:
        Tuple containing:
        - DataFrame with problem-level data
        - Problem rating map
        - Problem tags map
    """
    api = CodeforcesAPI()
    
    # Fetch contest list
    contests_raw = api.get_contest_list(gym=False)
    if not contests_raw:
        logger.error("Failed to fetch contest list")
        return pd.DataFrame(), {}, {}
    
    # Filter for relevant divisions
    target_divs = ["Div. 1", "Div. 2", "Div. 3", "Div. 4", "Div. 1+Div. 2"]
    filtered_contests = filter_contests_by_division(contests_raw, target_divs)
    logger.info(f"Found {len(filtered_contests)} target division contests")
    
    # Fetch problemset for ratings and tags
    problemset_data = api.get_problemset()
    if not problemset_data:
        logger.error("Failed to fetch problemset")
        return pd.DataFrame(), {}, {}
    
    rating_map = get_problem_rating_map(problemset_data)
    tags_map = get_problem_tags_map(problemset_data)
    
    # Fetch problem statistics (solved counts)
    problem_stats = problemset_data.get('problemStatistics', [])
    solved_count_map = {}
    for stat in problem_stats:
        contest_id = stat.get('contestId')
        index = stat.get('index')
        solved_count = stat.get('solvedCount', 0)
        if contest_id and index:
            key = f"{contest_id}{index}"
            solved_count_map[key] = solved_count
    
    # Build contest problems data
    all_problems_data = []
    
    # Use tqdm for progress if available
    contest_iter = tqdm(filtered_contests, desc="Processing contests") if HAS_TQDM else filtered_contests
    
    for contest in contest_iter:
        contest_id = contest.get('id')
        contest_name = contest.get('name', '')
        division = contest.get('division', 'Unknown')
        contest_phase = contest.get('phase', '')
        start_time = contest.get('startTimeSeconds', 0)
        duration = contest.get('durationSeconds', 0)
        
        # Convert start_time to datetime
        contest_date = datetime.fromtimestamp(start_time) if start_time else None
        
        # Fetch contest problems via problemset? We'll use the global problemset and filter.
        # However, contest-specific problems are not directly available via API.
        # We need to scrape or infer from problemset. The problemset includes
        # contestId for each problem, so we can filter by contestId.
        
        # Problems from this contest (filter by contestId)
        contest_problems = [
            p for p in problemset_data.get('problems', [])
            if p.get('contestId') == contest_id
        ]
        
        if not contest_problems:
            # Some contests (especially older ones) may not have problems in the global problemset?
            # We can try to fetch contest standings to get problems.
            standings = api._fetch("contest.standings", params={
                "contestId": contest_id,
                "from": 1,
                "count": 1,
                "showUnofficial": "false"
            })
            if standings and 'problems' in standings:
                contest_problems = standings['problems']
            else:
                # Skip contests without problem data
                continue
        
        for problem in contest_problems:
            problem_index = problem.get('index', '')
            problem_name = problem.get('name', '')
            problem_rating = problem.get('rating', 0)
            problem_tags = problem.get('tags', [])
            
            # Use rating_map if available
            prob_key = f"{contest_id}{problem_index}"
            if prob_key in rating_map:
                problem_rating = rating_map[prob_key]
            elif problem_rating == 0:
                problem_rating = None  # Unrated problem
            
            # Get solved count
            solved_count = solved_count_map.get(prob_key, 0)
            
            # Determine problem position in contest (A, B, C, etc.)
            # Extract letter from index (e.g., 'A', 'B1', 'B2')
            position = re.sub(r'[^A-Za-z]', '', problem_index).upper()
            if not position:
                position = problem_index
            # Keep only first letter for main problem order
            main_position = position[0] if position else '?'
            
            all_problems_data.append({
                'contest_id': contest_id,
                'contest_name': contest_name,
                'division': division,
                'contest_date': contest_date,
                'contest_phase': contest_phase,
                'duration_hours': round(duration / 3600, 1),
                'problem_index': problem_index,
                'problem_name': problem_name,
                'position_letter': main_position,
                'rating': problem_rating,
                'tags': problem_tags,
                'solved_count': solved_count
            })
        
        # Small delay to avoid overwhelming API
        time.sleep(0.5)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_problems_data)
    
    # Add derived columns
    if not df.empty:
        df['position_num'] = df['position_letter'].map({
            'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9, 'J': 10,
            'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10
        }).fillna(0).astype(int)
        
        # Clean ratings
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        
        # Extract year for trend analysis
        df['contest_year'] = pd.to_datetime(df['contest_date']).dt.year
        df['contest_month'] = pd.to_datetime(df['contest_date']).dt.month
        
        logger.info(f"Processed {len(df)} problems from {df['contest_id'].nunique()} contests")
    else:
        logger.warning("No problem data collected")
    
    return df, rating_map, tags_map


# =============================================================================
# SECTION 2: DATA ANALYSIS & STATISTICS
# =============================================================================

class ContestAnalyzer:
    """Perform comprehensive analysis on contest problem data."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        
    def get_division_summary(self) -> pd.DataFrame:
        """Generate summary statistics by division."""
        summary = []
        divisions = self.df['division'].unique()
        
        for div in divisions:
            div_df = self.df[self.df['division'] == div]
            problems_per_contest = div_df.groupby('contest_id').size()
            
            summary.append({
                'division': div,
                'total_contests': div_df['contest_id'].nunique(),
                'total_problems': len(div_df),
                'avg_problems_per_contest': len(div_df) / div_df['contest_id'].nunique(),
                'avg_rating': div_df['rating'].mean(),
                'median_rating': div_df['rating'].median(),
                'min_rating': div_df['rating'].min(),
                'max_rating': div_df['rating'].max(),
                'std_rating': div_df['rating'].std(),
                'problems_with_ratings': div_df['rating'].notna().sum(),
                'percentage_rated': (div_df['rating'].notna().sum() / len(div_df)) * 100
            })
        
        return pd.DataFrame(summary)
    
    def get_rating_by_position(self) -> pd.DataFrame:
        """Calculate average rating for each problem position across divisions."""
        # Filter valid positions and ratings
        valid_df = self.df[(self.df['position_num'] > 0) & (self.df['rating'].notna())]
        
        # Group by division and position
        position_rating = valid_df.groupby(['division', 'position_letter'])['rating'].agg([
            'mean', 'median', 'std', 'count'
        ]).reset_index()
        
        return position_rating
    
    def get_rating_trend_over_time(self) -> pd.DataFrame:
        """Analyze how problem ratings have changed over the years."""
        valid_df = self.df[(self.df['contest_year'] > 2010) & (self.df['rating'].notna())]
        
        trend = valid_df.groupby(['contest_year', 'position_letter'])['rating'].mean().reset_index()
        return trend
    
    def get_topic_frequency(self) -> pd.DataFrame:
        """Count frequency of problem tags across divisions."""
        # Explode tags list into individual rows
        tags_series = self.df['tags'].explode()
        tags_series = tags_series[tags_series.notna()]
        
        # Count by division
        tags_by_div = {}
        for div in self.df['division'].unique():
            div_tags = self.df[self.df['division'] == div]['tags'].explode()
            div_tags = div_tags[div_tags.notna()]
            tags_by_div[div] = Counter(div_tags)
        
        # Create summary DataFrame
        all_tags = set()
        for counter in tags_by_div.values():
            all_tags.update(counter.keys())
        
        freq_data = []
        for tag in sorted(all_tags):
            row = {'tag': tag}
            for div in tags_by_div:
                row[div] = tags_by_div[div].get(tag, 0)
            freq_data.append(row)
        
        freq_df = pd.DataFrame(freq_data)
        
        # Add total column
        tag_cols = [col for col in freq_df.columns if col != 'tag']
        freq_df['total'] = freq_df[tag_cols].sum(axis=1)
        
        return freq_df.sort_values('total', ascending=False)
    
    def get_difficulty_progression(self) -> pd.DataFrame:
        """
        Analyze the rating increase between consecutive problems in contests.
        A typical contest shows a pattern: A < B < C < D < E in difficulty.
        """
        valid_df = self.df[(self.df['position_num'] > 0) & (self.df['rating'].notna())]
        
        progression = []
        divisions = valid_df['division'].unique()
        
        for div in divisions:
            div_df = valid_df[valid_df['division'] == div]
            # For each contest, sort by position and calculate differences
            for contest_id in div_df['contest_id'].unique():
                contest_df = div_df[div_df['contest_id'] == contest_id].sort_values('position_num')
                positions = contest_df['position_letter'].tolist()
                ratings = contest_df['rating'].tolist()
                
                for i in range(len(ratings) - 1):
                    diff = ratings[i+1] - ratings[i]
                    progression.append({
                        'division': div,
                        'from_position': positions[i],
                        'to_position': positions[i+1],
                        'rating_diff': diff
                    })
        
        prog_df = pd.DataFrame(progression)
        
        # Average difference per transition
        avg_diff = prog_df.groupby(['division', 'from_position', 'to_position'])['rating_diff'].agg(
            ['mean', 'median', 'std']
        ).reset_index()
        
        return avg_diff
    
    def get_common_topics_for_position(self, top_n: int = 10) -> Dict:
        """
        Find the most common topics for each problem position.
        This reveals hidden patterns about which topics appear at which positions.
        """
        valid_df = self.df[self.df['position_num'] > 0].copy()
        valid_df = valid_df[valid_df['tags'].notna()]
        
        result = {}
        positions = sorted(valid_df['position_letter'].unique())
        
        for pos in positions:
            pos_df = valid_df[valid_df['position_letter'] == pos]
            all_tags = []
            for tags in pos_df['tags'].tolist():
                all_tags.extend(tags)
            
            tag_counter = Counter(all_tags)
            result[pos] = dict(tag_counter.most_common(top_n))
        
        return result
    
    def get_contest_difficulty_variance(self) -> pd.DataFrame:
        """
        Calculate rating standard deviation within each contest.
        High variance indicates mixed difficulty, low variance indicates consistent difficulty.
        """
        valid_df = self.df[self.df['rating'].notna()]
        
        contest_variance = valid_df.groupby(['contest_id', 'division', 'contest_name'])['rating'].agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).reset_index()
        
        # Sort by variance to find most consistent and most varied contests
        contest_variance = contest_variance.sort_values('std', ascending=False)
        
        return contest_variance
    
    def get_solve_rate_patterns(self) -> pd.DataFrame:
        """
        Analyze problem solving difficulty based on solved_count.
        Hidden pattern: Problems with many solves but high rating might be easy for rating.
        """
        valid_df = self.df[self.df['rating'].notna() & (self.df['solved_count'] > 0)]
        
        # Calculate solve density: solves per problem
        valid_df['solve_density'] = valid_df['solved_count']
        
        # Find anomalies: high rating but many solves (possibly easier than rating suggests)
        # Also low rating but few solves (possibly harder than rating suggests)
        
        # Normalize solved_count per division
        # Use percentiles within each position
        anomalies = []
        
        divisions = valid_df['division'].unique()
        for div in divisions:
            div_df = valid_df[valid_df['division'] == div]
            for pos in div_df['position_letter'].unique():
                pos_df = div_df[div_df['position_letter'] == pos]
                if len(pos_df) < 5:
                    continue
                
                # Calculate threshold for high solves (top 10% solved)
                high_threshold = pos_df['solved_count'].quantile(0.9)
                low_threshold = pos_df['solved_count'].quantile(0.1)
                
                # High rating but high solves (unusually easy for its rating)
                high_solve_high_rating = pos_df[
                    (pos_df['solved_count'] >= high_threshold) & 
                    (pos_df['rating'] > pos_df['rating'].median())
                ]
                
                # Low rating but low solves (unusually hard for its rating)
                low_solve_low_rating = pos_df[
                    (pos_df['solved_count'] <= low_threshold) & 
                    (pos_df['rating'] < pos_df['rating'].median())
                ]
                
                for _, row in high_solve_high_rating.iterrows():
                    anomalies.append({
                        'contest_id': row['contest_id'],
                        'problem_index': row['problem_index'],
                        'problem_name': row['problem_name'],
                        'division': div,
                        'position': pos,
                        'rating': row['rating'],
                        'solved_count': row['solved_count'],
                        'anomaly_type': 'easier_than_rating'
                    })
                
                for _, row in low_solve_low_rating.iterrows():
                    anomalies.append({
                        'contest_id': row['contest_id'],
                        'problem_index': row['problem_index'],
                        'problem_name': row['problem_name'],
                        'division': div,
                        'position': pos,
                        'rating': row['rating'],
                        'solved_count': row['solved_count'],
                        'anomaly_type': 'harder_than_rating'
                    })
        
        anomalies_df = pd.DataFrame(anomalies)
        return anomalies_df


# =============================================================================
# SECTION 3: INTERACTIVE VISUALIZATIONS
# =============================================================================

class InteractiveCharts:
    """Generate interactive Plotly visualizations."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.figures = {}
    
    def plot_division_rating_distribution(self) -> go.Figure:
        """Box plot of rating distribution by division."""
        valid_df = self.df[self.df['rating'].notna()]
        
        # Order divisions logically
        div_order = ["Div. 1", "Div. 2", "Div. 3", "Div. 4", "Div. 1+Div. 2", "Educational"]
        available_divs = [d for d in div_order if d in valid_df['division'].unique()]
        
        fig = go.Figure()
        
        for div in available_divs:
            div_df = valid_df[valid_df['division'] == div]
            fig.add_trace(go.Box(
                y=div_df['rating'],
                name=div,
                boxmean='sd',
                marker_color='lightblue',
                line_color='navy'
            ))
        
        fig.update_layout(
            title="Problem Rating Distribution by Division",
            yaxis_title="Problem Rating",
            xaxis_title="Division",
            template="plotly_dark",
            height=600,
            hovermode='closest'
        )
        
        self.figures['rating_distribution'] = fig
        return fig
    
    def plot_position_rating_heatmap(self) -> go.Figure:
        """Heatmap showing average rating for each position per division."""
        valid_df = self.df[(self.df['position_num'] > 0) & (self.df['rating'].notna())]
        
        # Pivot table: divisions as rows, positions as columns
        pivot_df = valid_df.pivot_table(
            index='division',
            columns='position_letter',
            values='rating',
            aggfunc='mean'
        )
        
        # Sort columns by typical contest order (A, B, C, D, E, F, G, H)
        col_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        pivot_df = pivot_df.reindex(columns=[c for c in col_order if c in pivot_df.columns])
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns,
            y=pivot_df.index,
            colorscale='Viridis',
            text=pivot_df.values.round(0),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False,
            colorbar=dict(title="Rating")
        ))
        
        fig.update_layout(
            title="Average Problem Rating by Position and Division",
            xaxis_title="Problem Position (A, B, C, ...)",
            yaxis_title="Division",
            template="plotly_dark",
            height=500
        )
        
        self.figures['position_heatmap'] = fig
        return fig
    
    def plot_rating_trend_line(self) -> go.Figure:
        """Line chart showing rating trends over the years by position."""
        valid_df = self.df[(self.df['contest_year'] > 2010) & (self.df['rating'].notna())]
        
        positions = ['A', 'B', 'C', 'D', 'E']
        colors = px.colors.qualitative.Set1
        
        fig = go.Figure()
        
        for i, pos in enumerate(positions):
            pos_df = valid_df[valid_df['position_letter'] == pos]
            if pos_df.empty:
                continue
            
            yearly_mean = pos_df.groupby('contest_year')['rating'].mean().reset_index()
            
            fig.add_trace(go.Scatter(
                x=yearly_mean['contest_year'],
                y=yearly_mean['rating'],
                mode='lines+markers',
                name=f'Problem {pos}',
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=6)
            ))
        
        fig.update_layout(
            title="Problem Rating Trends Over Time (2011-Present)",
            xaxis_title="Year",
            yaxis_title="Average Rating",
            template="plotly_dark",
            height=550,
            hovermode='x unified'
        )
        
        self.figures['rating_trend'] = fig
        return fig
    
    def plot_rating_increment_sankey(self) -> go.Figure:
        """Sankey diagram showing rating progression from problem to problem."""
        valid_df = self.df[(self.df['position_num'] > 0) & (self.df['rating'].notna())]
        
        # Bin ratings into categories for better visualization
        rating_bins = [(0, 800), (800, 1200), (1200, 1600), (1600, 2000), 
                      (2000, 2400), (2400, 2800), (2800, 3200), (3200, 4000)]
        
        def rating_to_cat(r):
            for low, high in rating_bins:
                if low <= r < high:
                    return f"{low}-{high}"
            return "4000+"
        
        # Collect transitions
        transitions = []
        for contest_id in valid_df['contest_id'].unique():
            contest_df = valid_df[valid_df['contest_id'] == contest_id].sort_values('position_num')
            ratings = contest_df['rating'].tolist()
            positions = contest_df['position_letter'].tolist()
            
            for i in range(len(ratings) - 1):
                from_cat = rating_to_cat(ratings[i])
                to_cat = rating_to_cat(ratings[i+1])
                transitions.append((from_cat, to_cat, positions[i]))
        
        # Count transitions
        transition_counts = {}
        for from_cat, to_cat, pos in transitions:
            key = (from_cat, to_cat)
            transition_counts[key] = transition_counts.get(key, 0) + 1
        
        # Prepare sankey data
        all_nodes = set()
        for from_cat, to_cat in transition_counts.keys():
            all_nodes.add(from_cat)
            all_nodes.add(to_cat)
        
        nodes = list(all_nodes)
        node_index = {node: i for i, node in enumerate(nodes)}
        
        sources = []
        targets = []
        values = []
        for (from_cat, to_cat), count in transition_counts.items():
            sources.append(node_index[from_cat])
            targets.append(node_index[to_cat])
            values.append(count)
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=nodes,
                color="lightblue"
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color="rgba(0, 150, 200, 0.4)"
            )
        )])
        
        fig.update_layout(
            title="Rating Progression Between Consecutive Problems",
            font=dict(size=12),
            template="plotly_dark",
            height=600
        )
        
        self.figures['rating_sankey'] = fig
        return fig
    
    def plot_topic_sunburst(self) -> go.Figure:
        """Sunburst chart showing topic distribution by division and position."""
        valid_df = self.df[self.df['tags'].notna()]
        
        # Explode tags and create hierarchical data
        sunburst_data = []
        for _, row in valid_df.iterrows():
            for tag in row['tags']:
                sunburst_data.append({
                    'division': row['division'],
                    'position': row['position_letter'],
                    'tag': tag
                })
        
        sb_df = pd.DataFrame(sunburst_data)
        
        # Count occurrences
        counts = sb_df.groupby(['division', 'position', 'tag']).size().reset_index(name='count')
        
        # Limit to top tags overall for readability
        top_tags = counts.groupby('tag')['count'].sum().nlargest(20).index
        counts = counts[counts['tag'].isin(top_tags)]
        
        fig = px.sunburst(
            counts,
            path=['division', 'position', 'tag'],
            values='count',
            title='Topic Distribution by Division and Problem Position',
            color='count',
            color_continuous_scale='Viridis',
            maxdepth=3
        )
        
        fig.update_layout(
            template="plotly_dark",
            height=700,
            margin=dict(t=50, l=0, r=0, b=0)
        )
        
        self.figures['topic_sunburst'] = fig
        return fig
    
    def plot_rating_vs_solved_scatter(self) -> go.Figure:
        """Scatter plot showing relationship between rating and solve count."""
        valid_df = self.df[self.df['rating'].notna() & (self.df['solved_count'] > 0)]
        
        fig = px.scatter(
            valid_df,
            x='rating',
            y='solved_count',
            color='division',
            size='rating',
            hover_data=['contest_name', 'problem_name', 'position_letter'],
            title='Problem Rating vs Number of Solves',
            labels={'rating': 'Problem Rating', 'solved_count': 'Number of Solves'},
            template="plotly_dark",
            opacity=0.7
        )
        
        fig.update_layout(
            height=550,
            xaxis=dict(title="Problem Rating"),
            yaxis=dict(title="Number of Solves (log scale)", type="log")
        )
        
        self.figures['rating_solved'] = fig
        return fig
    
    def create_dashboard(self) -> go.Figure:
        """
        Create a comprehensive dashboard with multiple subplots.
        """
        valid_df = self.df[self.df['rating'].notna()]
        
        # Create subplots: 2x2 grid
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Rating Distribution by Division',
                           'Rating by Position (Avg)',
                           'Rating Trend Over Time',
                           'Topic Frequency'),
            specs=[[{'type': 'box'}, {'type': 'bar'}],
                   [{'type': 'scatter'}, {'type': 'bar'}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # Subplot 1: Box plot
        div_order = ["Div. 1", "Div. 2", "Div. 3", "Div. 4", "Div. 1+Div. 2"]
        for i, div in enumerate(div_order):
            div_df = valid_df[valid_df['division'] == div]
            if not div_df.empty:
                fig.add_trace(
                    go.Box(y=div_df['rating'], name=div, showlegend=False),
                    row=1, col=1
                )
        
        # Subplot 2: Bar chart of ratings by position
        pos_rating = valid_df.groupby('position_letter')['rating'].mean().reset_index()
        pos_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        pos_rating = pos_rating[pos_rating['position_letter'].isin(pos_order)]
        fig.add_trace(
            go.Bar(x=pos_rating['position_letter'], y=pos_rating['rating'],
                  marker_color='coral', name='Avg Rating'),
            row=1, col=2
        )
        
        # Subplot 3: Rating trend over years
        yearly_rating = valid_df[valid_df['contest_year'] > 2010].groupby('contest_year')['rating'].mean().reset_index()
        fig.add_trace(
            go.Scatter(x=yearly_rating['contest_year'], y=yearly_rating['rating'],
                      mode='lines+markers', line=dict(color='lightgreen'),
                      name='Yearly Avg', showlegend=False),
            row=2, col=1
        )
        
        # Subplot 4: Topic frequency bar chart (top 15)
        tags_series = valid_df['tags'].explode()
        tags_series = tags_series[tags_series.notna()]
        top_tags = tags_series.value_counts().head(15)
        fig.add_trace(
            go.Bar(x=top_tags.values, y=top_tags.index, orientation='h',
                  marker_color='mediumpurple', name='Topic Frequency'),
            row=2, col=2
        )
        
        fig.update_layout(
            title=dict(
                text="Codeforces Contest Analysis Dashboard",
                font=dict(size=24),
                x=0.5
            ),
            template="plotly_dark",
            height=900,
            showlegend=False,
            hovermode='closest'
        )
        
        fig.update_xaxes(title_text="Division", row=1, col=1)
        fig.update_yaxes(title_text="Rating", row=1, col=1)
        fig.update_xaxes(title_text="Problem Position", row=1, col=2)
        fig.update_yaxes(title_text="Average Rating", row=1, col=2)
        fig.update_xaxes(title_text="Year", row=2, col=1)
        fig.update_yaxes(title_text="Average Rating", row=2, col=1)
        fig.update_xaxes(title_text="Number of Problems", row=2, col=2)
        fig.update_yaxes(title_text="Topic", row=2, col=2)
        
        self.figures['dashboard'] = fig
        return fig
    
    def plot_difficulty_progression_violin(self) -> go.Figure:
        """Violin plot showing rating differences between consecutive problems."""
        # Calculate differences per contest
        valid_df = self.df[(self.df['position_num'] > 0) & (self.df['rating'].notna())]
        
        diffs = []
        for contest_id in valid_df['contest_id'].unique():
            contest_df = valid_df[valid_df['contest_id'] == contest_id].sort_values('position_num')
            ratings = contest_df['rating'].tolist()
            positions = contest_df['position_letter'].tolist()
            
            for i in range(len(ratings) - 1):
                diff = ratings[i+1] - ratings[i]
                diffs.append({
                    'from': positions[i],
                    'to': positions[i+1],
                    'diff': diff
                })
        
        diff_df = pd.DataFrame(diffs)
        
        fig = px.violin(diff_df, x='to', y='diff', box=True, points='outliers',
                        color='to', title='Rating Increase Between Consecutive Problems',
                        labels={'to': 'Problem Position', 'diff': 'Rating Increase'},
                        template="plotly_dark")
        
        fig.update_layout(
            height=550,
            yaxis=dict(title="Rating Increase"),
            xaxis=dict(title="Next Problem Position")
        )
        
        self.figures['difficulty_progression'] = fig
        return fig


# =============================================================================
# SECTION 4: HIDDEN PATTERN DETECTION
# =============================================================================

class HiddenPatternDetector:
    """Detect hidden patterns and anomalies in contest data."""
    
    def __init__(self, df: pd.DataFrame, analyzer: ContestAnalyzer):
        self.df = df
        self.analyzer = analyzer
        self.patterns = []
    
    def detect_topic_inflation(self) -> Dict:
        """
        Detect if certain topics have become more or less common over time.
        """
        valid_df = self.df[self.df['tags'].notna() & (self.df['contest_year'] > 2010)]
        
        # Group by year and tag
        tags_by_year = defaultdict(Counter)
        
        for _, row in valid_df.iterrows():
            year = row['contest_year']
            for tag in row['tags']:
                tags_by_year[year][tag] += 1
        
        # Calculate total problems per year
        total_by_year = valid_df.groupby('contest_year').size()
        
        # Calculate frequency per year
        freq_by_year = {}
        for year in tags_by_year:
            total = total_by_year[year]
            freq_by_year[year] = {tag: count/total for tag, count in tags_by_year[year].items()}
        
        # Find trending topics (increase over time)
        trending = {}
        years = sorted(freq_by_year.keys())
        top_tags = set()
        for year_data in freq_by_year.values():
            top_tags.update(year_data.keys())
        
        for tag in top_tags:
            freqs = []
            for year in years:
                if year in freq_by_year and tag in freq_by_year[year]:
                    freqs.append(freq_by_year[year][tag])
                else:
                    freqs.append(0)
            
            if len(freqs) >= 5:
                # Simple trend detection: compare first 3 years vs last 3 years
                first_avg = np.mean(freqs[:3]) if len(freqs) >= 3 else 0
                last_avg = np.mean(freqs[-3:]) if len(freqs) >= 3 else 0
                change = last_avg - first_avg
                
                if change > 0.02:  # 2% increase
                    trending[tag] = {'trend': 'increasing', 'change': change, 'freqs': freqs}
                elif change < -0.02:  # 2% decrease
                    trending[tag] = {'trend': 'decreasing', 'change': change, 'freqs': freqs}
        
        self.patterns.append({
            'name': 'Topic Inflation/Deflation',
            'description': 'Topics that have become more or less common over time',
            'data': trending
        })
        
        return trending
    
    def detect_rating_clustering(self) -> Dict:
        """
        Detect if problems tend to cluster around specific rating ranges.
        """
        valid_df = self.df[self.df['rating'].notna()]
        
        # Round ratings to nearest 25
        valid_df['rating_rounded'] = (valid_df['rating'] / 25).round() * 25
        
        rating_counts = valid_df['rating_rounded'].value_counts().sort_index()
        
        # Find peaks (local maxima)
        peaks = []
        ratings = sorted(rating_counts.index)
        
        for i, r in enumerate(ratings):
            count = rating_counts[r]
            left_count = rating_counts[ratings[i-1]] if i > 0 else 0
            right_count = rating_counts[ratings[i+1]] if i < len(ratings)-1 else 0
            
            if count > left_count and count > right_count and count > np.median(rating_counts.values) * 1.5:
                peaks.append(r)
        
        # Find gaps (ratings with very few problems)
        gaps = []
        for i in range(len(ratings)-1):
            diff = ratings[i+1] - ratings[i]
            if diff > 100 and rating_counts[ratings[i+1]] > 10 and rating_counts[ratings[i]] > 10:
                gaps.append((ratings[i], ratings[i+1], diff))
        
        result = {
            'clusters': peaks,
            'gaps': gaps,
            'distribution': rating_counts.to_dict()
        }
        
        self.patterns.append({
            'name': 'Rating Clustering & Gaps',
            'description': 'Ratings where problems cluster (popular difficulty) and gaps (rare difficulty)',
            'data': result
        })
        
        return result
    
    def detect_contest_symmetry(self) -> Dict:
        """
        Detect if contests have symmetrical difficulty curves.
        """
        valid_df = self.df[self.df['rating'].notna()]
        
        symmetry_score = []
        for contest_id in valid_df['contest_id'].unique():
            contest_df = valid_df[valid_df['contest_id'] == contest_id].sort_values('position_num')
            ratings = contest_df['rating'].tolist()
            
            if len(ratings) >= 3:
                # Compare first and last halves
                half = len(ratings) // 2
                first_half = ratings[:half]
                second_half = ratings[-half:]
                
                # Reverse second half for comparison
                second_reversed = second_half[::-1]
                
                # Compute similarity (1 - normalized difference)
                max_len = min(len(first_half), len(second_reversed))
                if max_len > 0:
                    diff_sum = sum(abs(a - b) for a, b in zip(first_half[:max_len], second_reversed[:max_len]))
                    max_possible_diff = max_len * 2000
                    similarity = 1 - (diff_sum / max_possible_diff)
                    symmetry_score.append(similarity)
        
        avg_symmetry = np.mean(symmetry_score) if symmetry_score else 0
        
        result = {
            'average_symmetry': avg_symmetry,
            'interpretation': 'High symmetry (>0.7) indicates contests are balanced, low symmetry (<0.4) indicates front-loaded difficulty'
        }
        
        self.patterns.append({
            'name': 'Contest Symmetry Analysis',
            'description': 'How balanced the difficulty curve is within contests',
            'data': result
        })
        
        return result
    
    def detect_position_difficulty_leaps(self) -> Dict:
        """
        Detect positions where difficulty jumps are unusually large or small.
        """
        valid_df = self.df[(self.df['position_num'] > 0) & (self.df['rating'].notna())]
        
        differences = []
        for contest_id in valid_df['contest_id'].unique():
            contest_df = valid_df[valid_df['contest_id'] == contest_id].sort_values('position_num')
            ratings = contest_df['rating'].tolist()
            positions = contest_df['position_letter'].tolist()
            
            for i in range(len(ratings) - 1):
                diff = ratings[i+1] - ratings[i]
                differences.append({
                    'from': positions[i],
                    'to': positions[i+1],
                    'diff': diff
                })
        
        diff_df = pd.DataFrame(differences)
        
        # Find transitions with unusually high/low average differences
        abnormal_jumps = {}
        transitions = diff_df.groupby(['from', 'to'])['diff'].agg(['mean', 'std', 'count']).reset_index()
        
        for _, row in transitions.iterrows():
            mean_diff = row['mean']
            std_diff = row['std']
            count = row['count']
            
            if count >= 10:  # Sufficient data
                if mean_diff > 300:
                    abnormal_jumps[f"{row['from']}→{row['to']}"] = {
                        'type': 'large_jump',
                        'mean': mean_diff,
                        'std': std_diff,
                        'count': count
                    }
                elif mean_diff < 50:
                    abnormal_jumps[f"{row['from']}→{row['to']}"] = {
                        'type': 'small_jump',
                        'mean': mean_diff,
                        'std': std_diff,
                        'count': count
                    }
        
        result = {
            'abnormal_jumps': abnormal_jumps,
            'overall_stats': {
                'mean_diff': diff_df['diff'].mean(),
                'median_diff': diff_df['diff'].median(),
                'std_diff': diff_df['diff'].std()
            }
        }
        
        self.patterns.append({
            'name': 'Difficulty Leap Analysis',
            'description': 'Identifying unusual difficulty jumps between consecutive problems',
            'data': result
        })
        
        return result
    
    def detect_educational_round_impact(self) -> Dict:
        """
        Compare Educational Rounds with regular Div contests.
        """
        educational_mask = self.df['contest_name'].str.contains('Educational', case=False, na=False)
        edu_df = self.df[educational_mask & self.df['rating'].notna()]
        regular_df = self.df[~educational_mask & self.df['division'].isin(['Div. 1', 'Div. 2', 'Div. 3', 'Div. 4']) & self.df['rating'].notna()]
        
        if edu_df.empty:
            return {'error': 'No educational rounds found'}
        
        edu_avg = edu_df.groupby('position_letter')['rating'].mean()
        reg_avg = regular_df.groupby('position_letter')['rating'].mean()
        
        # Compare
        comparison = {}
        for pos in set(edu_avg.index) | set(reg_avg.index):
            edu_val = edu_avg.get(pos, 0)
            reg_val = reg_avg.get(pos, 0)
            diff = edu_val - reg_val if reg_val != 0 else 0
            comparison[pos] = {
                'educational_rating': edu_val,
                'regular_rating': reg_val,
                'difference': diff,
                'harder_or_easier': 'harder' if diff > 0 else 'easier' if diff < 0 else 'similar'
            }
        
        result = {
            'comparison': comparison,
            'educational_rounds_avg': edu_df['rating'].mean(),
            'regular_rounds_avg': regular_df['rating'].mean(),
            'educational_problems': len(edu_df),
            'regular_problems': len(regular_df)
        }
        
        self.patterns.append({
            'name': 'Educational Round vs Regular Round',
            'description': 'How Educational Rounds compare to regular division contests',
            'data': result
        })
        
        return result
    
    def generate_hidden_patterns_report(self) -> str:
        """Generate a textual report of all detected hidden patterns."""
        report = []
        report.append("=" * 80)
        report.append("HIDDEN PATTERNS & ANOMALIES DETECTION REPORT")
        report.append("=" * 80)
        report.append("")
        
        for pattern in self.patterns:
            report.append(f"\n🔍 {pattern['name']}")
            report.append("-" * 50)
            report.append(f"📝 {pattern['description']}")
            
            data = pattern['data']
            
            if pattern['name'] == 'Topic Inflation/Deflation':
                if data:
                    increasing = [t for t, v in data.items() if v['trend'] == 'increasing']
                    decreasing = [t for t, v in data.items() if v['trend'] == 'decreasing']
                    
                    if increasing:
                        report.append(f"📈 Increasing topics: {', '.join(sorted(increasing, key=lambda t: data[t]['change'], reverse=True)[:10])}")
                    if decreasing:
                        report.append(f"📉 Decreasing topics: {', '.join(sorted(decreasing, key=lambda t: data[t]['change'])[:10])}")
                else:
                    report.append("No significant topic trends detected.")
            
            elif pattern['name'] == 'Rating Clustering & Gaps':
                clusters = data.get('clusters', [])
                gaps = data.get('gaps', [])
                if clusters:
                    report.append(f"🎯 Popular difficulty clusters: {clusters[:10]}")
                if gaps:
                    report.append(f"⏸️  Rating gaps: {gaps[:5]}")
            
            elif pattern['name'] == 'Contest Symmetry Analysis':
                symmetry = data.get('average_symmetry', 0)
                interpretation = data.get('interpretation', '')
                report.append(f"⚖️ Average contest symmetry score: {symmetry:.3f}")
                report.append(f"💡 {interpretation}")
            
            elif pattern['name'] == 'Difficulty Leap Analysis':
                jumps = data.get('abnormal_jumps', {})
                if jumps:
                    for transition, info in jumps.items():
                        report.append(f"⚠️ {transition}: {info['type']} (avg {info['mean']:.0f} rating change)")
                overall = data.get('overall_stats', {})
                report.append(f"📊 Average difficulty jump: {overall.get('mean_diff', 0):.0f} rating points")
            
            elif pattern['name'] == 'Educational Round vs Regular Round':
                comp = data.get('comparison', {})
                edu_avg = data.get('educational_rounds_avg', 0)
                reg_avg = data.get('regular_rounds_avg', 0)
                diff = edu_avg - reg_avg
                report.append(f"📚 Educational rounds average rating: {edu_avg:.0f}")
                report.append(f"📊 Regular rounds average rating: {reg_avg:.0f}")
                report.append(f"🔄 Difference: {diff:+.0f} ({'harder' if diff > 0 else 'easier' if diff < 0 else 'similar'})")
                report.append("Position-specific comparison:")
                for pos in sorted(comp.keys())[:6]:
                    report.append(f"   Problem {pos}: Educational {comp[pos]['educational_rating']:.0f} vs Regular {comp[pos]['regular_rating']:.0f}")
            
            report.append("")
        
        return "\n".join(report)


# =============================================================================
# SECTION 5: MAIN EXECUTION & HTML EXPORT
# =============================================================================

def save_html_report(
    df: pd.DataFrame,
    analyzer: ContestAnalyzer,
    charts: InteractiveCharts,
    pattern_detector: HiddenPatternDetector,
    output_path: str = "codeforces_analysis_report.html"
) -> str:
    """Generate and save a comprehensive HTML report."""
    
    # Generate all charts
    charts.plot_division_rating_distribution()
    charts.plot_position_rating_heatmap()
    charts.plot_rating_trend_line()
    charts.plot_rating_increment_sankey()
    charts.plot_topic_sunburst()
    charts.plot_rating_vs_solved_scatter()
    charts.plot_difficulty_progression_violin()
    charts.create_dashboard()
    
    # Get analysis results
    division_summary = analyzer.get_division_summary()
    rating_by_position = analyzer.get_rating_by_position()
    topic_freq = analyzer.get_topic_frequency()
    difficulty_progression = analyzer.get_difficulty_progression()
    anomalies = analyzer.get_solve_rate_patterns()
    
    # Get hidden patterns report
    hidden_patterns_report = pattern_detector.generate_hidden_patterns_report()
    
    # Start building HTML
    html_parts = []
    
    html_parts.append("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Codeforces Division Contest Analysis</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #0f172a;
                color: #f1f5f9;
            }
            .container {
                max-width: 1400px;
                margin: 0 auto;
            }
            h1 {
                text-align: center;
                color: #38bdf8;
                margin-bottom: 10px;
            }
            .subtitle {
                text-align: center;
                color: #94a3b8;
                margin-bottom: 40px;
            }
            .card {
                background-color: #1e293b;
                border-radius: 12px;
                padding: 20px;
                margin-bottom: 30px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            }
            .card h2 {
                color: #facc15;
                margin-top: 0;
                border-bottom: 2px solid #334155;
                padding-bottom: 10px;
            }
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }
            .stat-box {
                background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
                border-radius: 12px;
                padding: 20px;
                text-align: center;
                border: 1px solid #334155;
            }
            .stat-number {
                font-size: 2.5em;
                font-weight: bold;
                color: #38bdf8;
            }
            .stat-label {
                font-size: 0.9em;
                color: #94a3b8;
                margin-top: 8px;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 15px 0;
            }
            th, td {
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #334155;
            }
            th {
                background-color: #2d3748;
                color: #facc15;
            }
            tr:hover {
                background-color: #2d3748;
            }
            .badge {
                display: inline-block;
                padding: 4px 8px;
                border-radius: 20px;
                font-size: 0.8em;
                font-weight: bold;
            }
            .badge-easy { background-color: #22c55e; color: #000; }
            .badge-medium { background-color: #eab308; color: #000; }
            .badge-hard { background-color: #ef4444; color: #fff; }
            .hidden-patterns {
                background-color: #1a1f2e;
                border-left: 4px solid #f59e0b;
                padding: 15px;
                font-family: monospace;
                font-size: 14px;
                white-space: pre-wrap;
                overflow-x: auto;
            }
            footer {
                text-align: center;
                padding: 20px;
                color: #64748b;
                font-size: 0.8em;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 Codeforces Contest Analysis</h1>
            <div class="subtitle">Comprehensive Analysis of Div 1, Div 2, Div 3, Div 4 & Div 1+Div 2 Contests</div>
    """)
    
    # Statistics summary
    total_contests = df['contest_id'].nunique()
    total_problems = len(df)
    rated_problems = df['rating'].notna().sum()
    avg_rating = df['rating'].mean()
    
    html_parts.append(f"""
            <div class="stats-grid">
                <div class="stat-box"><div class="stat-number">{total_contests}</div><div class="stat-label">Total Contests Analyzed</div></div>
                <div class="stat-box"><div class="stat-number">{total_problems:,}</div><div class="stat-label">Total Problems Analyzed</div></div>
                <div class="stat-box"><div class="stat-number">{rated_problems:,}</div><div class="stat-label">Rated Problems</div></div>
                <div class="stat-box"><div class="stat-number">{avg_rating:.0f}</div><div class="stat-label">Average Problem Rating</div></div>
            </div>
    """)
    
    # Division Summary Table
    html_parts.append("""
            <div class="card">
                <h2>📋 Division Summary</h2>
    """)
    html_parts.append(division_summary.to_html(classes='table', index=False))
    html_parts.append("""
            </div>
    """)
    
    # Rating by Position Table
    if not rating_by_position.empty:
        html_parts.append("""
            <div class="card">
                <h2>📊 Average Rating by Problem Position</h2>
        """)
        pivot_table = rating_by_position.pivot(index='division', columns='position_letter', values='mean')
        html_parts.append(pivot_table.to_html(classes='table'))
        html_parts.append("""
            </div>
        """)
    
    # Topic Frequency Table (Top 20)
    if not topic_freq.empty:
        html_parts.append("""
            <div class="card">
                <h2>🏷️ Topic Frequency by Division (Top 20)</h2>
        """)
        top20 = topic_freq.head(20)
        html_parts.append(top20.to_html(classes='table', index=False))
        html_parts.append("""
            </div>
        """)
    
    # Interactive Charts
    html_parts.append("""
            <div class="card">
                <h2>📈 Interactive Charts</h2>
    """)
    
    # Add each chart
    for name, fig in charts.figures.items():
        if fig:
            html_parts.append(f"<h3>{fig.layout.title.text if hasattr(fig.layout, 'title') else name}</h3>")
            html_parts.append(fig.to_html(full_html=False, include_plotlyjs='cdn'))
            html_parts.append("<br>")
    
    html_parts.append("""
            </div>
    """)
    
    # Hidden Patterns Section
    html_parts.append("""
            <div class="card">
                <h2>🔮 Hidden Patterns & Anomalies</h2>
                <div class="hidden-patterns">
    """)
    html_parts.append(f"<pre>{hidden_patterns_report}</pre>")
    html_parts.append("""
                </div>
            </div>
    """)
    
    # Anomalies Table (if any)
    if not anomalies.empty:
        html_parts.append("""
            <div class="card">
                <h2>⚠️ Problem Anomalies (Unusually Easy/Hard for Rating)</h2>
        """)
        html_parts.append(anomalies.head(20).to_html(classes='table', index=False))
        html_parts.append("""
            </div>
        """)
    
    # Footer
    html_parts.append(f"""
            <footer>
                Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
                Data fetched from Codeforces API | Analysis covers Div 1, Div 2, Div 3, Div 4, and Div 1+Div 2 contests
            </footer>
        </div>
    </body>
    </html>
    """)
    
    full_html = "\n".join(html_parts)
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_html)
    
    logger.info(f"Report saved to {output_path}")
    return output_path


def main():
    """Main execution function."""
    print("=" * 80)
    print("Codeforces Division Contest Analyzer")
    print("Analyzing all Div 1, Div 2, Div 3, Div 4, and Div 1+Div 2 contests")
    print("=" * 80)
    print()
    
    # Step 1: Fetch data
    print("📡 Fetching data from Codeforces API...")
    df, rating_map, tags_map = fetch_all_relevant_data()
    
    if df.empty:
        print("❌ No data fetched. Please check your internet connection and try again.")
        return
    
    print(f"✅ Fetched {len(df)} problems from {df['contest_id'].nunique()} contests")
    print()
    
    # Step 2: Analyze data
    print("🔬 Performing analysis...")
    analyzer = ContestAnalyzer(df)
    
    # Step 3: Generate charts
    print("📊 Generating interactive charts...")
    charts = InteractiveCharts(df)
    
    # Step 4: Detect hidden patterns
    print("🔍 Detecting hidden patterns...")
    pattern_detector = HiddenPatternDetector(df, analyzer)
    pattern_detector.detect_topic_inflation()
    pattern_detector.detect_rating_clustering()
    pattern_detector.detect_contest_symmetry()
    pattern_detector.detect_position_difficulty_leaps()
    pattern_detector.detect_educational_round_impact()
    
    # Step 5: Save HTML report
    print("💾 Saving HTML report...")
    output_path = save_html_report(df, analyzer, charts, pattern_detector)
    
    print()
    print("=" * 80)
    print(f"✅ Analysis complete! Report saved to: {output_path}")
    print("=" * 80)
    print()
    
    # Optionally open in browser
    open_browser = input("Open report in browser? (y/n): ").lower().strip()
    if open_browser == 'y':
        webbrowser.open(f"file://{os.path.abspath(output_path)}")
    
    print("\nThank you for using Codeforces Contest Analyzer!")


if __name__ == "__main__":
    main()