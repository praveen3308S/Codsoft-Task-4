#!/usr/bin/env python3
"""
Hybrid Movie Recommendation Engine
Combines content-based filtering, collaborative filtering, and popularity-based recommendations
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import pickle
import os
from typing import List, Dict, Tuple, Optional, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class HybridRecommendationEngine:
    """Advanced recommendation engine with multiple algorithms"""
    
    def __init__(self):
        self.movies_df = None
        self.content_similarity_matrix = None
        self.genre_similarity_matrix = None
        self.cast_similarity_matrix = None
        self.user_ratings = {}  # Store user ratings for collaborative filtering
        self.popularity_scores = None
        
        # TF-IDF vectorizers
        self.content_vectorizer = None
        self.genre_vectorizer = None
        
        # Model cache directory
        self.model_cache_dir = "models"
        os.makedirs(self.model_cache_dir, exist_ok=True)
    
    def load_movie_data(self, movies_df: pd.DataFrame):
        """Load and preprocess movie data"""
        self.movies_df = movies_df.copy()
        
        # Create combined features for content-based filtering
        self._create_content_features()
        
        # Calculate popularity scores
        self._calculate_popularity_scores()
        
        # Build similarity matrices
        self._build_similarity_matrices()
    
    def _create_content_features(self):
        """Create combined content features"""
        # Combine title, overview, and other text features
        self.movies_df['content_features'] = (
            self.movies_df['title'].fillna('').astype(str) + ' ' +
            self.movies_df['overview'].fillna('').astype(str)
        )
        
        # Create genre features if available
        if 'genres' in self.movies_df.columns:
            self.movies_df['genre_features'] = self.movies_df['genres'].fillna('').astype(str)
        else:
            self.movies_df['genre_features'] = ''
        
        # Create cast features if available
        if 'cast' in self.movies_df.columns:
            self.movies_df['cast_features'] = self.movies_df['cast'].fillna('').astype(str)
        else:
            self.movies_df['cast_features'] = ''
    
    def _calculate_popularity_scores(self):
        """Calculate popularity scores based on vote average and count"""
        # Weighted rating formula (IMDB's weighted rating)
        min_votes = self.movies_df['vote_count'].quantile(0.6)  # 60th percentile
        mean_rating = self.movies_df['vote_average'].mean()
        
        def weighted_rating(row):
            v = row['vote_count']
            R = row['vote_average']
            return (v / (v + min_votes) * R) + (min_votes / (v + min_votes) * mean_rating)
        
        self.movies_df['popularity_score'] = self.movies_df.apply(weighted_rating, axis=1)
        
        # Normalize popularity scores
        max_pop = self.movies_df['popularity_score'].max()
        min_pop = self.movies_df['popularity_score'].min()
        self.movies_df['normalized_popularity'] = (
            (self.movies_df['popularity_score'] - min_pop) / (max_pop - min_pop)
        )
    
    def _build_similarity_matrices(self):
        """Build various similarity matrices"""
        cache_file = os.path.join(self.model_cache_dir, "similarity_matrices.pkl")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.content_similarity_matrix = cached_data['content']
                    self.genre_similarity_matrix = cached_data['genre']
                    self.cast_similarity_matrix = cached_data['cast']
                    self.content_vectorizer = cached_data['content_vectorizer']
                    self.genre_vectorizer = cached_data['genre_vectorizer']
                logger.info("Loaded cached similarity matrices")
                return
            except Exception as e:
                logger.warning(f"Failed to load cached matrices: {e}")
        
        # Build content-based similarity
        self.content_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        
        content_matrix = self.content_vectorizer.fit_transform(self.movies_df['content_features'])
        self.content_similarity_matrix = cosine_similarity(content_matrix)
        
        # Build genre-based similarity
        if self.movies_df['genre_features'].str.len().sum() > 0:
            self.genre_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english'
            )
            genre_matrix = self.genre_vectorizer.fit_transform(self.movies_df['genre_features'])
            self.genre_similarity_matrix = cosine_similarity(genre_matrix)
        else:
            self.genre_similarity_matrix = np.zeros((len(self.movies_df), len(self.movies_df)))
        
        # Build cast-based similarity
        if self.movies_df['cast_features'].str.len().sum() > 0:
            cast_vectorizer = TfidfVectorizer(
                max_features=2000,
                stop_words='english'
            )
            cast_matrix = cast_vectorizer.fit_transform(self.movies_df['cast_features'])
            self.cast_similarity_matrix = cosine_similarity(cast_matrix)
        else:
            self.cast_similarity_matrix = np.zeros((len(self.movies_df), len(self.movies_df)))
        
        # Cache the matrices
        try:
            cache_data = {
                'content': self.content_similarity_matrix,
                'genre': self.genre_similarity_matrix,
                'cast': self.cast_similarity_matrix,
                'content_vectorizer': self.content_vectorizer,
                'genre_vectorizer': self.genre_vectorizer
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info("Cached similarity matrices")
        except Exception as e:
            logger.warning(f"Failed to cache matrices: {e}")
    
    def get_content_based_recommendations(self, movie_title: str, num_recommendations: int = 10) -> List[Dict]:
        """Get content-based recommendations"""
        try:
            # Find movie index
            movie_indices = self.movies_df[
                self.movies_df['title'].str.lower() == movie_title.lower()
            ].index
            
            if len(movie_indices) == 0:
                return []
            
            movie_idx = movie_indices[0]
            
            # Get similarity scores
            sim_scores = list(enumerate(self.content_similarity_matrix[movie_idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
            
            # Get recommended movies
            recommendations = []
            for idx, score in sim_scores:
                movie_data = self.movies_df.iloc[idx].to_dict()
                movie_data['similarity_score'] = score
                movie_data['recommendation_type'] = 'content_based'
                recommendations.append(movie_data)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in content-based recommendations: {e}")
            return []
    
    def get_genre_based_recommendations(self, movie_title: str, num_recommendations: int = 10) -> List[Dict]:
        """Get genre-based recommendations"""
        try:
            movie_indices = self.movies_df[
                self.movies_df['title'].str.lower() == movie_title.lower()
            ].index
            
            if len(movie_indices) == 0:
                return []
            
            movie_idx = movie_indices[0]
            
            # Get similarity scores
            sim_scores = list(enumerate(self.genre_similarity_matrix[movie_idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
            
            recommendations = []
            for idx, score in sim_scores:
                if score > 0:  # Only include movies with genre similarity
                    movie_data = self.movies_df.iloc[idx].to_dict()
                    movie_data['similarity_score'] = score
                    movie_data['recommendation_type'] = 'genre_based'
                    recommendations.append(movie_data)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in genre-based recommendations: {e}")
            return []
    
    def get_popularity_based_recommendations(self, num_recommendations: int = 10) -> List[Dict]:
        """Get popularity-based recommendations"""
        try:
            # Sort by popularity score
            popular_movies = self.movies_df.nlargest(num_recommendations, 'popularity_score')
            
            recommendations = []
            for _, movie in popular_movies.iterrows():
                movie_data = movie.to_dict()
                movie_data['similarity_score'] = movie['normalized_popularity']
                movie_data['recommendation_type'] = 'popularity_based'
                recommendations.append(movie_data)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in popularity-based recommendations: {e}")
            return []
    
    def get_hybrid_recommendations(self, movie_title: str, num_recommendations: int = 10, 
                                 weights: Dict[str, float] = None) -> List[Dict]:
        """Get hybrid recommendations combining multiple approaches"""
        if weights is None:
            weights = {
                'content': 0.5,
                'genre': 0.3,
                'popularity': 0.2
            }
        
        try:
            # Get recommendations from different approaches
            content_recs = self.get_content_based_recommendations(movie_title, num_recommendations * 2)
            genre_recs = self.get_genre_based_recommendations(movie_title, num_recommendations * 2)
            popularity_recs = self.get_popularity_based_recommendations(num_recommendations)
            
            # Combine and weight recommendations
            movie_scores = {}
            
            # Process content-based recommendations
            for rec in content_recs:
                movie_id = rec.get('id', rec.get('movie_id', rec['title']))
                if movie_id not in movie_scores:
                    movie_scores[movie_id] = {'data': rec, 'score': 0}
                movie_scores[movie_id]['score'] += rec['similarity_score'] * weights['content']
            
            # Process genre-based recommendations
            for rec in genre_recs:
                movie_id = rec.get('id', rec.get('movie_id', rec['title']))
                if movie_id not in movie_scores:
                    movie_scores[movie_id] = {'data': rec, 'score': 0}
                movie_scores[movie_id]['score'] += rec['similarity_score'] * weights['genre']
            
            # Process popularity-based recommendations
            for rec in popularity_recs:
                movie_id = rec.get('id', rec.get('movie_id', rec['title']))
                if movie_id not in movie_scores:
                    movie_scores[movie_id] = {'data': rec, 'score': 0}
                movie_scores[movie_id]['score'] += rec['similarity_score'] * weights['popularity']
            
            # Sort by combined score and return top recommendations
            sorted_movies = sorted(
                movie_scores.values(),
                key=lambda x: x['score'],
                reverse=True
            )[:num_recommendations]
            
            # Prepare final recommendations
            recommendations = []
            for item in sorted_movies:
                movie_data = item['data'].copy()
                movie_data['hybrid_score'] = item['score']
                movie_data['recommendation_type'] = 'hybrid'
                recommendations.append(movie_data)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in hybrid recommendations: {e}")
            return self.get_content_based_recommendations(movie_title, num_recommendations)
    
    def add_user_rating(self, user_id: str, movie_title: str, rating: float):
        """Add user rating for collaborative filtering"""
        if user_id not in self.user_ratings:
            self.user_ratings[user_id] = {}
        self.user_ratings[user_id][movie_title] = rating
    
    def get_movie_by_title(self, title: str) -> Optional[Dict]:
        """Get movie data by title"""
        try:
            movie = self.movies_df[
                self.movies_df['title'].str.lower() == title.lower()
            ].iloc[0]
            return movie.to_dict()
        except:
            return None
    
    def clear_cache(self):
        """Clear all cached models"""
        try:
            cache_file = os.path.join(self.model_cache_dir, "similarity_matrices.pkl")
            if os.path.exists(cache_file):
                os.remove(cache_file)
            logger.info("Recommendation engine cache cleared")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
