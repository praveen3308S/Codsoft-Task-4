#!/usr/bin/env python3
"""
Enhanced TMDB API Service
Provides robust API interactions with The Movie Database with caching, rate limiting, and error handling
"""

import requests
import time
import json
import os
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from cachetools import TTLCache
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TMDBAPIService:
    """Enhanced TMDB API service with robust error handling and caching"""
    
    def __init__(self, api_key: str = None):
        # Use provided API key or default
        self.api_key = api_key or "6177b4297dff132d300422e0343471fb"
        self.base_url = "https://api.themoviedb.org/3"
        self.image_base_url = "https://image.tmdb.org/t/p"
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'MovieRecommender/1.0',
            'Accept': 'application/json'
        })
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.25  # 4 requests per second max
        
        # In-memory cache with TTL (Time To Live)
        self.cache = TTLCache(maxsize=1000, ttl=3600)  # 1 hour TTL
        
        # Persistent cache directory
        self.cache_dir = "cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Image size configurations
        self.image_sizes = {
            'poster': ['w92', 'w154', 'w185', 'w342', 'w500', 'w780', 'original'],
            'backdrop': ['w300', 'w780', 'w1280', 'original'],
            'profile': ['w45', 'w185', 'h632', 'original']
        }
    
    def _rate_limit(self):
        """Implement rate limiting to respect API limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _get_cache_key(self, endpoint: str, params: Dict) -> str:
        """Generate cache key for request"""
        # Sort params for consistent cache keys
        sorted_params = sorted(params.items()) if params else []
        return f"{endpoint}_{hash(str(sorted_params))}"
    
    def _load_persistent_cache(self, cache_key: str) -> Optional[Any]:
        """Load data from persistent cache"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        if os.path.exists(cache_file):
            try:
                # Check if cache is still valid (24 hours)
                file_age = time.time() - os.path.getmtime(cache_file)
                if file_age < 86400:  # 24 hours
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
                else:
                    os.remove(cache_file)  # Remove expired cache
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_key}: {e}")
        
        return None
    
    def _save_persistent_cache(self, cache_key: str, data: Any):
        """Save data to persistent cache"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_key}: {e}")
    
    def make_request(self, endpoint: str, params: Dict = None, use_cache: bool = True) -> Optional[Dict]:
        """Make API request with caching and error handling"""
        if params is None:
            params = {}
        
        # Add API key
        params['api_key'] = self.api_key
        
        # Check cache first
        cache_key = self._get_cache_key(endpoint, params)
        
        if use_cache:
            # Check in-memory cache
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # Check persistent cache
            cached_data = self._load_persistent_cache(cache_key)
            if cached_data:
                self.cache[cache_key] = cached_data
                return cached_data
        
        # Rate limiting
        self._rate_limit()
        
        # Make request
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Cache the response
                if use_cache:
                    self.cache[cache_key] = data
                    self._save_persistent_cache(cache_key, data)
                
                return data
            
            elif response.status_code == 429:  # Rate limited
                logger.warning("Rate limited, waiting...")
                time.sleep(1)
                return self.make_request(endpoint, params, use_cache)
            
            else:
                logger.error(f"API Error {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error(f"Timeout for endpoint: {endpoint}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {endpoint}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error for {endpoint}: {e}")
            return None
    
    def get_image_url(self, image_path: str, image_type: str = 'poster', size: str = 'w500') -> str:
        """Get full image URL with fallback"""
        if not image_path:
            return self._get_placeholder_image(image_type)
        
        # Validate size
        if size not in self.image_sizes.get(image_type, []):
            size = 'w500' if image_type == 'poster' else 'w780'
        
        return f"{self.image_base_url}/{size}{image_path}"
    
    def _get_placeholder_image(self, image_type: str) -> str:
        """Get placeholder image URL"""
        placeholders = {
            'poster': "https://via.placeholder.com/500x750/2c3e50/ecf0f1?text=No+Poster",
            'backdrop': "https://via.placeholder.com/1280x720/2c3e50/ecf0f1?text=No+Backdrop",
            'profile': "https://via.placeholder.com/185x278/2c3e50/ecf0f1?text=No+Photo"
        }
        return placeholders.get(image_type, placeholders['poster'])
    
    def search_movies(self, query: str, page: int = 1) -> Optional[Dict]:
        """Search for movies"""
        return self.make_request("search/movie", {
            "query": query,
            "page": page,
            "include_adult": False
        })
    
    def get_movie_details(self, movie_id: int) -> Optional[Dict]:
        """Get detailed movie information"""
        return self.make_request(f"movie/{movie_id}", {
            "append_to_response": "credits,videos,similar,recommendations"
        })
    
    def get_popular_movies(self, page: int = 1) -> Optional[Dict]:
        """Get popular movies"""
        return self.make_request("movie/popular", {"page": page})
    
    def get_top_rated_movies(self, page: int = 1) -> Optional[Dict]:
        """Get top rated movies"""
        return self.make_request("movie/top_rated", {"page": page})
    
    def get_now_playing_movies(self, page: int = 1) -> Optional[Dict]:
        """Get now playing movies"""
        return self.make_request("movie/now_playing", {"page": page})
    
    def get_upcoming_movies(self, page: int = 1) -> Optional[Dict]:
        """Get upcoming movies"""
        return self.make_request("movie/upcoming", {"page": page})
    
    def get_movie_genres(self) -> Optional[Dict]:
        """Get list of movie genres"""
        return self.make_request("genre/movie/list")
    
    def discover_movies(self, **kwargs) -> Optional[Dict]:
        """Discover movies with filters"""
        return self.make_request("discover/movie", kwargs)
    
    def get_person_details(self, person_id: int) -> Optional[Dict]:
        """Get person details"""
        return self.make_request(f"person/{person_id}")
    
    def clear_cache(self):
        """Clear all caches"""
        self.cache.clear()
        
        # Clear persistent cache
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl'):
                    os.remove(os.path.join(self.cache_dir, filename))
            logger.info("Cache cleared successfully")
        except Exception as e:
            logger.error(f"Failed to clear persistent cache: {e}")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        persistent_files = 0
        try:
            persistent_files = len([f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')])
        except:
            pass
        
        return {
            'memory_cache_size': len(self.cache),
            'memory_cache_maxsize': self.cache.maxsize,
            'persistent_cache_files': persistent_files
        }
