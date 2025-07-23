#!/usr/bin/env python3
"""
User Preferences and Rating System
Handles user ratings, preferences, and personalized recommendations
"""

import json
import os
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class UserPreferencesManager:
    """Manages user preferences, ratings, and viewing history"""
    
    def __init__(self, user_data_dir: str = "user_data"):
        self.user_data_dir = user_data_dir
        os.makedirs(user_data_dir, exist_ok=True)
        
        # File paths
        self.ratings_file = os.path.join(user_data_dir, "user_ratings.json")
        self.watchlist_file = os.path.join(user_data_dir, "watchlist.json")
        self.preferences_file = os.path.join(user_data_dir, "preferences.json")
        self.history_file = os.path.join(user_data_dir, "viewing_history.json")
        
        # Load existing data
        self.user_ratings = self._load_json_file(self.ratings_file, {})
        self.watchlist = self._load_json_file(self.watchlist_file, [])
        self.preferences = self._load_json_file(self.preferences_file, {})
        self.viewing_history = self._load_json_file(self.history_file, [])
    
    def _load_json_file(self, filepath: str, default_value):
        """Load JSON file with error handling"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load {filepath}: {e}")
        return default_value
    
    def _save_json_file(self, filepath: str, data):
        """Save data to JSON file"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save {filepath}: {e}")
    
    def add_rating(self, movie_id: str, movie_title: str, rating: float, user_id: str = "default"):
        """Add or update a movie rating"""
        if user_id not in self.user_ratings:
            self.user_ratings[user_id] = {}
        
        self.user_ratings[user_id][movie_id] = {
            'rating': rating,
            'title': movie_title,
            'timestamp': datetime.now().isoformat()
        }
        
        self._save_json_file(self.ratings_file, self.user_ratings)
        logger.info(f"Added rating {rating} for movie {movie_title}")
    
    def get_user_ratings(self, user_id: str = "default") -> Dict:
        """Get all ratings for a user"""
        return self.user_ratings.get(user_id, {})
    
    def get_average_rating(self, user_id: str = "default") -> float:
        """Get user's average rating"""
        ratings = self.get_user_ratings(user_id)
        if not ratings:
            return 0.0
        
        total_rating = sum(r['rating'] for r in ratings.values())
        return total_rating / len(ratings)
    
    def add_to_watchlist(self, movie_data: Dict, user_id: str = "default"):
        """Add movie to watchlist"""
        # Check if movie already in watchlist
        for item in self.watchlist:
            if item.get('id') == movie_data.get('id') and item.get('user_id') == user_id:
                return False  # Already in watchlist
        
        watchlist_item = movie_data.copy()
        watchlist_item['user_id'] = user_id
        watchlist_item['added_timestamp'] = datetime.now().isoformat()
        
        self.watchlist.append(watchlist_item)
        self._save_json_file(self.watchlist_file, self.watchlist)
        
        logger.info(f"Added {movie_data.get('title')} to watchlist")
        return True
    
    def remove_from_watchlist(self, movie_id: str, user_id: str = "default"):
        """Remove movie from watchlist"""
        original_length = len(self.watchlist)
        self.watchlist = [
            item for item in self.watchlist 
            if not (item.get('id') == movie_id and item.get('user_id') == user_id)
        ]
        
        if len(self.watchlist) < original_length:
            self._save_json_file(self.watchlist_file, self.watchlist)
            logger.info(f"Removed movie {movie_id} from watchlist")
            return True
        return False
    
    def get_watchlist(self, user_id: str = "default") -> List[Dict]:
        """Get user's watchlist"""
        return [item for item in self.watchlist if item.get('user_id') == user_id]
    
    def add_to_viewing_history(self, movie_data: Dict, user_id: str = "default"):
        """Add movie to viewing history"""
        history_item = {
            'movie_id': movie_data.get('id'),
            'title': movie_data.get('title'),
            'user_id': user_id,
            'viewed_timestamp': datetime.now().isoformat()
        }
        
        # Remove if already exists to avoid duplicates
        self.viewing_history = [
            item for item in self.viewing_history 
            if not (item.get('movie_id') == movie_data.get('id') and item.get('user_id') == user_id)
        ]
        
        self.viewing_history.append(history_item)
        
        # Keep only last 100 items per user
        user_history = [item for item in self.viewing_history if item.get('user_id') == user_id]
        if len(user_history) > 100:
            # Remove oldest items
            user_history.sort(key=lambda x: x.get('viewed_timestamp', ''))
            items_to_remove = user_history[:-100]
            for item in items_to_remove:
                self.viewing_history.remove(item)
        
        self._save_json_file(self.history_file, self.viewing_history)
    
    def get_viewing_history(self, user_id: str = "default", limit: int = 50) -> List[Dict]:
        """Get user's viewing history"""
        user_history = [item for item in self.viewing_history if item.get('user_id') == user_id]
        user_history.sort(key=lambda x: x.get('viewed_timestamp', ''), reverse=True)
        return user_history[:limit]
    
    def update_preferences(self, preferences: Dict, user_id: str = "default"):
        """Update user preferences"""
        if user_id not in self.preferences:
            self.preferences[user_id] = {}
        
        self.preferences[user_id].update(preferences)
        self.preferences[user_id]['updated_timestamp'] = datetime.now().isoformat()
        
        self._save_json_file(self.preferences_file, self.preferences)
    
    def get_preferences(self, user_id: str = "default") -> Dict:
        """Get user preferences"""
        return self.preferences.get(user_id, {})
    
    def get_favorite_genres(self, user_id: str = "default", limit: int = 5) -> List[str]:
        """Get user's favorite genres based on ratings"""
        ratings = self.get_user_ratings(user_id)
        if not ratings:
            return []
        
        # This would need movie data to extract genres
        # For now, return from preferences
        prefs = self.get_preferences(user_id)
        return prefs.get('favorite_genres', [])[:limit]
    
    def get_recommendation_weights(self, user_id: str = "default") -> Dict[str, float]:
        """Get personalized recommendation weights based on user behavior"""
        prefs = self.get_preferences(user_id)
        ratings = self.get_user_ratings(user_id)
        
        # Default weights
        weights = {
            'content': 0.4,
            'genre': 0.3,
            'popularity': 0.2,
            'collaborative': 0.1
        }
        
        # Adjust based on user preferences
        if prefs.get('prefer_popular_movies'):
            weights['popularity'] += 0.1
            weights['content'] -= 0.05
            weights['genre'] -= 0.05
        
        if prefs.get('prefer_genre_similarity'):
            weights['genre'] += 0.1
            weights['content'] -= 0.05
            weights['popularity'] -= 0.05
        
        # Adjust based on rating behavior
        if len(ratings) > 10:
            # User has rated many movies, increase collaborative filtering
            weights['collaborative'] += 0.1
            weights['popularity'] -= 0.1
        
        # Ensure weights sum to 1
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        
        return weights
    
    def get_user_stats(self, user_id: str = "default") -> Dict:
        """Get user statistics"""
        ratings = self.get_user_ratings(user_id)
        watchlist = self.get_watchlist(user_id)
        history = self.get_viewing_history(user_id)
        
        stats = {
            'total_ratings': len(ratings),
            'average_rating': self.get_average_rating(user_id),
            'watchlist_size': len(watchlist),
            'movies_viewed': len(history),
            'favorite_genres': self.get_favorite_genres(user_id),
            'recommendation_weights': self.get_recommendation_weights(user_id)
        }
        
        if ratings:
            ratings_values = [r['rating'] for r in ratings.values()]
            stats.update({
                'highest_rating': max(ratings_values),
                'lowest_rating': min(ratings_values),
                'rating_std': pd.Series(ratings_values).std() if len(ratings_values) > 1 else 0
            })
        
        return stats
    
    def export_user_data(self, user_id: str = "default") -> Dict:
        """Export all user data"""
        return {
            'ratings': self.get_user_ratings(user_id),
            'watchlist': self.get_watchlist(user_id),
            'preferences': self.get_preferences(user_id),
            'viewing_history': self.get_viewing_history(user_id),
            'stats': self.get_user_stats(user_id)
        }
    
    def clear_user_data(self, user_id: str = "default"):
        """Clear all data for a user"""
        # Remove user ratings
        if user_id in self.user_ratings:
            del self.user_ratings[user_id]
            self._save_json_file(self.ratings_file, self.user_ratings)
        
        # Remove user watchlist items
        self.watchlist = [item for item in self.watchlist if item.get('user_id') != user_id]
        self._save_json_file(self.watchlist_file, self.watchlist)
        
        # Remove user preferences
        if user_id in self.preferences:
            del self.preferences[user_id]
            self._save_json_file(self.preferences_file, self.preferences)
        
        # Remove user viewing history
        self.viewing_history = [item for item in self.viewing_history if item.get('user_id') != user_id]
        self._save_json_file(self.history_file, self.viewing_history)
        
        logger.info(f"Cleared all data for user {user_id}")
