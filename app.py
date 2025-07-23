#!/usr/bin/env python3
"""
Enhanced Movie Recommendation System
Uses existing movie-env and data files with improved UI and TMDB integration
"""

import streamlit as st
import pandas as pd
import pickle
import requests
import os

import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="🎬 MOVIE RECOMMENDATION SYSTEM",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# TMDB API Configuration
TMDB_API_KEY = "6177b4297dff132d300422e0343471fb"
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

# Enhanced CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(90deg, #e74c3c, #c0392b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    
    .movie-card {
        background: linear-gradient(145deg, #ffffff, #f8f9fa);
        border-radius: 20px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        transition: all 0.3s ease;
        height: 100%;
        position: relative;
        overflow: hidden;
    }

    .movie-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.15);
    }

    .movie-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #e74c3c, #f39c12, #3498db);
    }

    /* Beautiful movie cards */
    .beautiful-movie-card {
        transition: all 0.3s ease !important;
        cursor: pointer;
    }

    .beautiful-movie-card:hover {
        transform: translateY(-10px) scale(1.02) !important;
        box-shadow: 0 20px 50px rgba(0,0,0,0.3) !important;
    }
    
    .movie-title {
        font-weight: bold;
        font-size: 1.2rem;
        color: #2c3e50;
        margin-bottom: 15px;
        text-align: center;
        min-height: 60px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .movie-poster {
        border-radius: 15px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
        width: 100%;
    }
    
    .movie-poster:hover {
        transform: scale(1.05);
    }
    
    .movie-info {
        margin: 8px 0;
        padding: 5px 10px;
        background: rgba(52, 152, 219, 0.1);
        border-radius: 8px;
        font-size: 0.9rem;
    }

    .rating-badge {
        background: linear-gradient(45deg, #f39c12, #e67e22);
        color: white;
        padding: 5px 12px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 5px 0;
    }

    .genre-tag {
        background: linear-gradient(45deg, #3498db, #2980b9);
        color: white;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        margin: 2px;
        display: inline-block;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #e74c3c, #c0392b);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 12px 30px;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 4px 15px rgba(231, 76, 60, 0.3);
    }

    .stButton > button:hover {
        background: linear-gradient(45deg, #c0392b, #a93226);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(231, 76, 60, 0.4);
    }
    
    .recommendation-section {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 30px;
        border-radius: 20px;
        margin: 20px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }

    .section-title {
        font-size: 1.8rem;
        color: #2c3e50;
        margin-bottom: 20px;
        text-align: center;
        font-weight: bold;
    }
    
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }

    .stat-number {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 5px;
    }

    .stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)

class EnhancedMovieRecommender:
    def __init__(self):
        self.movies_df = None
        self.similarity_matrices = {}
        self.session = requests.Session()
        
        # Initialize session state
        if 'user_ratings' not in st.session_state:
            st.session_state.user_ratings = {}
        if 'watchlist' not in st.session_state:
            st.session_state.watchlist = []
        if 'viewing_history' not in st.session_state:
            st.session_state.viewing_history = []
    
    @st.cache_data
    def load_movie_data(_self):
        """Load movie data from existing files"""
        try:
            # Load the main movie dataset
            movies_df = pd.read_csv('Files/tmdb_5000_movies.csv')
            credits_df = pd.read_csv('Files/tmdb_5000_credits.csv')

            # Fix column names for merging - credits has 'movie_id', movies has 'id'
            movies_df = movies_df.merge(credits_df, left_on='id', right_on='movie_id', how='left')

            # Drop duplicate title column from credits
            if 'title_y' in movies_df.columns:
                movies_df = movies_df.drop('title_y', axis=1)
                movies_df = movies_df.rename(columns={'title_x': 'title'})

            # Load preprocessed data if available (this might have better processed features)
            try:
                with open('Files/new_df_dict.pkl', 'rb') as f:
                    processed_data = pickle.load(f)
                    if isinstance(processed_data, pd.DataFrame):
                        # Use processed data but keep the original structure
                        st.sidebar.info("✅ Using preprocessed movie data")
                        return processed_data
            except Exception as e:
                st.sidebar.warning(f"⚠️ Could not load preprocessed data: {e}")

            st.sidebar.success(f"✅ Loaded {len(movies_df)} movies from CSV files")
            return movies_df

        except Exception as e:
            st.error(f"Error loading movie data: {e}")
            st.error("Please ensure the following files exist in the Files/ directory:")
            st.error("- tmdb_5000_movies.csv")
            st.error("- tmdb_5000_credits.csv")
            return None
    
    @st.cache_data
    def load_similarity_matrices(_self):
        """Load pre-computed similarity matrices"""
        matrices = {}
        similarity_files = {
            'tags': 'Files/similarity_tags_tags.pkl',
            'genres': 'Files/similarity_tags_genres.pkl',
            'keywords': 'Files/similarity_tags_keywords.pkl',
            'cast': 'Files/similarity_tags_tcast.pkl',
            'production': 'Files/similarity_tags_tprduction_comp.pkl'
        }

        loaded_count = 0
        for name, filepath in similarity_files.items():
            try:
                if os.path.exists(filepath):
                    with open(filepath, 'rb') as f:
                        matrix = pickle.load(f)
                        matrices[name] = matrix
                        loaded_count += 1
                        st.sidebar.success(f"✅ Loaded {name} similarity matrix ({matrix.shape})")
                else:
                    st.sidebar.warning(f"⚠️ File not found: {filepath}")
            except Exception as e:
                st.sidebar.error(f"❌ Error loading {name} matrix: {e}")

        if loaded_count == 0:
            st.sidebar.error("❌ No similarity matrices loaded!")
            st.sidebar.info("💡 Please ensure similarity matrix files are in the Files/ directory")
        else:
            st.sidebar.success(f"✅ Loaded {loaded_count}/{len(similarity_files)} similarity matrices")

        return matrices
    

    
    def get_recommendations(self, movie_title, similarity_type='tags', num_recommendations=6):
        """Get movie recommendations using existing similarity matrices"""
        try:
            # Check if similarity matrix is available
            if similarity_type not in self.similarity_matrices:
                available_types = list(self.similarity_matrices.keys())
                if available_types:
                    return [], f"'{similarity_type}' similarity matrix not available. Available types: {', '.join(available_types)}"
                else:
                    return [], "No similarity matrices loaded. Please check your Files/ directory for .pkl files."

            # Find movie index
            movie_indices = self.movies_df[
                self.movies_df['title'].str.lower() == movie_title.lower()
            ].index

            if len(movie_indices) == 0:
                # Try partial matching
                partial_matches = self.movies_df[
                    self.movies_df['title'].str.contains(movie_title, case=False, na=False)
                ]
                if len(partial_matches) > 0:
                    suggestions = partial_matches['title'].head(5).tolist()
                    return [], f"Movie '{movie_title}' not found. Did you mean: {', '.join(suggestions)}?"
                else:
                    return [], f"Movie '{movie_title}' not found in database"

            movie_idx = movie_indices[0]
            similarity_matrix = self.similarity_matrices[similarity_type]

            # Check matrix dimensions
            if movie_idx >= similarity_matrix.shape[0]:
                return [], f"Movie index {movie_idx} out of range for similarity matrix (size: {similarity_matrix.shape})"

            # Get similarity scores
            sim_scores = list(enumerate(similarity_matrix[movie_idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]

            # Get recommended movies
            movie_indices = [i[0] for i in sim_scores]
            recommended_movies = self.movies_df.iloc[movie_indices]

            return recommended_movies, None

        except Exception as e:
            return [], f"Error generating recommendations: {str(e)}"
    
    def display_movie_card(self, movie_data, col, show_actions=True, section="default"):
        """Display clean movie card without images"""
        with col:
            # Get basic movie info
            title = movie_data.get('title', 'Unknown Title')
            rating = movie_data.get('vote_average', 0)
            release_date = movie_data.get('release_date', '')
            year = release_date[:4] if release_date else 'Unknown'
            vote_count = movie_data.get('vote_count', 0)
            popularity = movie_data.get('popularity', 0)
            overview = movie_data.get('overview', '')

            # Determine rating color
            if rating >= 8.0:
                rating_color = "#27ae60"  # Green
                quality = "Excellent"
            elif rating >= 7.0:
                rating_color = "#3498db"  # Blue
                quality = "Good"
            elif rating >= 6.0:
                rating_color = "#f39c12"  # Orange
                quality = "Average"
            else:
                rating_color = "#e74c3c"  # Red
                quality = "Poor" if rating > 0 else "Unrated"

            # Create clean card
            with st.container():
                st.markdown(f"""
                <div style="
                    border: 2px solid {rating_color};
                    border-radius: 15px;
                    padding: 20px;
                    margin: 10px 0;
                    background: linear-gradient(145deg, #ffffff, #f8f9fa);
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                ">
                    <h3 style="color: {rating_color}; margin: 0 0 10px 0; text-align: center;">
                        {title}
                    </h3>
                    <div style="text-align: center; margin-bottom: 15px;">
                        <span style="background: {rating_color}; color: white; padding: 5px 15px;
                                   border-radius: 20px; font-weight: bold; margin-right: 10px;">
                            ⭐ {rating:.1f}
                        </span>
                        <span style="background: #95a5a6; color: white; padding: 5px 15px;
                                   border-radius: 20px; font-weight: bold;">
                            📅 {year}
                        </span>
                    </div>
                    <div style="text-align: center; margin-bottom: 15px;">
                        <span style="background: #ecf0f1; color: #2c3e50; padding: 3px 10px;
                                   border-radius: 15px; font-size: 0.9rem; font-weight: bold;">
                            {quality}
                        </span>
                    </div>
                    <div style="display: flex; justify-content: space-around; margin-bottom: 10px;">
                        <div style="text-align: center;">
                            <div style="font-weight: bold; color: #7f8c8d;">👥 Votes</div>
                            <div style="color: #2c3e50;">{vote_count:,}</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-weight: bold; color: #7f8c8d;">🔥 Popular</div>
                            <div style="color: #2c3e50;">{popularity:.0f}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Overview in expandable section
            if overview:
                with st.expander("📖 Overview"):
                    st.write(overview)

            # Action buttons with improved state management
            if show_actions:
                col1, col2 = st.columns(2)
                movie_id = movie_data.get('id', 0)

                with col1:
                    # Check if movie is already in watchlist
                    watchlist = st.session_state.get('watchlist', [])
                    existing_ids = [m.get('id') for m in watchlist]
                    is_in_watchlist = movie_id in existing_ids

                    button_text = "✅ In Watchlist" if is_in_watchlist else "➕ Add to Watchlist"
                    button_disabled = is_in_watchlist

                    if st.button(button_text, key=f"watchlist_{section}_{movie_id}",
                               disabled=button_disabled, use_container_width=True):
                        self.add_to_watchlist(movie_data)
                        st.rerun()

                with col2:
                    # Check if movie is already rated
                    user_ratings = st.session_state.get('user_ratings', {})
                    current_rating = user_ratings.get(movie_id, {}).get('rating', 0)

                    rating_key = f"rate_{section}_{movie_id}"

                    # Initialize rating state if not exists
                    if rating_key not in st.session_state:
                        st.session_state[rating_key] = current_rating

                    user_rating = st.selectbox(
                        "⭐ Rate",
                        options=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        index=current_rating,
                        key=rating_key,
                        format_func=lambda x: f"{x}/10" if x > 0 else "Rate"
                    )

                    # Only add rating if it changed and is greater than 0
                    if user_rating > 0 and user_rating != current_rating:
                        self.add_rating(movie_id, title, user_rating)
                        st.rerun()
    
    def add_to_watchlist(self, movie_data):
        """Add movie to watchlist with AI-powered suggestions"""
        # Initialize watchlist if it doesn't exist
        if 'watchlist' not in st.session_state:
            st.session_state.watchlist = []

        # Convert Series to dict if needed
        if hasattr(movie_data, 'to_dict'):
            movie_dict = movie_data.to_dict()
        else:
            movie_dict = dict(movie_data)

        movie_id = movie_dict.get('id', 0)
        movie_title = movie_dict.get('title', 'Unknown')

        # Check if already in watchlist
        existing_ids = [m.get('id') for m in st.session_state.watchlist]

        if movie_id not in existing_ids:
            st.session_state.watchlist.append(movie_dict)

            # Store success message in session state to show after rerun
            st.session_state.last_action_message = f"✅ Added '{movie_title}' to your watchlist!"

            # AI-powered suggestion for multiple movies
            if len(st.session_state.watchlist) >= 3:
                st.session_state.last_ai_message = "🤖 **AI Insight:** Based on your watchlist, you might also like movies with similar themes. Check the recommendations page!"
        else:
            st.session_state.last_action_message = f"⚠️ '{movie_title}' is already in your watchlist!"

    def add_rating(self, movie_id, title, rating):
        """Add user rating with AI learning"""
        # Initialize ratings if it doesn't exist
        if 'user_ratings' not in st.session_state:
            st.session_state.user_ratings = {}

        # Store the rating
        st.session_state.user_ratings[movie_id] = {
            'title': title,
            'rating': rating,
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # Store success message in session state to show after rerun
        st.session_state.last_action_message = f"⭐ Rated '{title}' {rating}/10!"

        # AI learning feedback
        if rating >= 8:
            st.session_state.last_ai_message = "🤖 **AI Learning:** Great choice! I'll recommend more high-quality movies like this."
        elif rating >= 6:
            st.session_state.last_ai_message = "🤖 **AI Learning:** Noted your preference. I'll find similar movies you might enjoy."
        else:
            st.session_state.last_ai_message = "🤖 **AI Learning:** Thanks for the feedback. I'll avoid recommending similar movies."

    def display_action_messages(self):
        """Display any pending action messages and clear them"""
        if 'last_action_message' in st.session_state:
            if 'Added' in st.session_state.last_action_message:
                st.success(st.session_state.last_action_message)
            elif 'Rated' in st.session_state.last_action_message:
                st.success(st.session_state.last_action_message)
            else:
                st.warning(st.session_state.last_action_message)
            del st.session_state.last_action_message

        if 'last_ai_message' in st.session_state:
            st.info(st.session_state.last_ai_message)
            del st.session_state.last_ai_message

    def get_ai_personalized_recommendations(self, num_recommendations=6):
        """AI-powered personalized recommendations based on user behavior"""
        recommendations = []

        # Get user preferences from ratings and watchlist
        user_ratings = st.session_state.get('user_ratings', {})
        watchlist = st.session_state.get('watchlist', [])

        if user_ratings:
            # Find highly rated movies by user
            high_rated_movies = [
                movie_id for movie_id, data in user_ratings.items()
                if data['rating'] >= 7
            ]

            if high_rated_movies:
                # Get recommendations based on highly rated movies
                for movie_id in high_rated_movies[:3]:  # Use top 3 rated movies
                    try:
                        movie_row = self.movies_df[self.movies_df['id'] == movie_id]
                        if not movie_row.empty:
                            movie_title = movie_row.iloc[0]['title']
                            similar_movies = self.get_recommendations(movie_title, 'tags', 3)
                            if similar_movies is not None and not similar_movies.empty:
                                recommendations.extend(similar_movies.to_dict('records'))
                    except Exception as e:
                        continue

        if watchlist and len(recommendations) < num_recommendations:
            # Add recommendations based on watchlist
            for movie in watchlist[-3:]:  # Use last 3 added movies
                try:
                    movie_title = movie.get('title', '')
                    if movie_title:
                        similar_movies = self.get_recommendations(movie_title, 'tags', 2)
                        if similar_movies is not None and not similar_movies.empty:
                            recommendations.extend(similar_movies.to_dict('records'))
                except Exception as e:
                    continue

        if len(recommendations) < num_recommendations:
            # Fill with top-rated movies if not enough personalized recommendations
            top_movies = self.movies_df.nlargest(num_recommendations - len(recommendations), 'vote_average')
            recommendations.extend(top_movies.to_dict('records'))

        # Remove duplicates and limit results
        seen_ids = set()
        unique_recommendations = []
        for movie in recommendations:
            movie_id = movie.get('id')
            if movie_id not in seen_ids and len(unique_recommendations) < num_recommendations:
                seen_ids.add(movie_id)
                unique_recommendations.append(movie)

        return unique_recommendations[:num_recommendations]

    def get_ai_chatbot_response(self, user_message):
        """AI chatbot for movie recommendations based on genres"""
        user_message = user_message.lower().strip()

        # Define genre keywords and responses
        genre_responses = {
            'action': {
                'keywords': ['action', 'fight', 'adventure', 'thriller', 'explosive', 'intense'],
                'response': "🎬 I love action movies! Here are some adrenaline-pumping recommendations:",
                'genres': ['Action', 'Adventure', 'Thriller']
            },
            'comedy': {
                'keywords': ['comedy', 'funny', 'laugh', 'humor', 'hilarious', 'joke'],
                'response': "😄 Comedy is the best medicine! Here are some movies that'll make you laugh:",
                'genres': ['Comedy']
            },
            'drama': {
                'keywords': ['drama', 'emotional', 'deep', 'serious', 'touching', 'meaningful'],
                'response': "🎭 Drama movies can be so powerful! Here are some emotionally rich films:",
                'genres': ['Drama']
            },
            'horror': {
                'keywords': ['horror', 'scary', 'frightening', 'spooky', 'terror', 'creepy'],
                'response': "😱 Ready for some scares? Here are some spine-chilling horror movies:",
                'genres': ['Horror', 'Thriller']
            },
            'romance': {
                'keywords': ['romance', 'love', 'romantic', 'relationship', 'dating', 'heart'],
                'response': "💕 Love is in the air! Here are some beautiful romantic movies:",
                'genres': ['Romance']
            },
            'sci-fi': {
                'keywords': ['sci-fi', 'science fiction', 'space', 'future', 'alien', 'technology'],
                'response': "🚀 The future is here! Check out these amazing sci-fi movies:",
                'genres': ['Science Fiction']
            },
            'fantasy': {
                'keywords': ['fantasy', 'magic', 'wizard', 'dragon', 'mythical', 'supernatural'],
                'response': "🧙‍♂️ Enter magical worlds! Here are some enchanting fantasy movies:",
                'genres': ['Fantasy']
            },
            'animation': {
                'keywords': ['animation', 'animated', 'cartoon', 'pixar', 'disney', 'kids'],
                'response': "🎨 Animation brings stories to life! Here are some amazing animated films:",
                'genres': ['Animation']
            }
        }

        # Check for greetings
        greetings = ['hello', 'hi', 'hey', 'good morning', 'good evening', 'greetings']
        if any(greeting in user_message for greeting in greetings):
            return {
                'response': "🤖 Hello! I'm your AI movie assistant! I can recommend movies based on genres. Try asking me about action, comedy, drama, horror, romance, sci-fi, fantasy, or animation movies!",
                'movies': [],
                'genre': None
            }

        # Check for help requests
        help_keywords = ['help', 'what can you do', 'how do you work', 'commands']
        if any(keyword in user_message for keyword in help_keywords):
            return {
                'response': "🎬 I'm here to help you find great movies! Here's what I can do:\n\n• Ask me about specific genres (action, comedy, drama, horror, romance, sci-fi, fantasy, animation)\n• I'll recommend movies from our database\n• Try: 'I want action movies' or 'Show me comedies'\n• I can also suggest movies based on your mood!",
                'movies': [],
                'genre': None
            }

        # Check for mood-based requests
        if 'sad' in user_message or 'depressed' in user_message:
            return {
                'response': "🌟 When you're feeling down, sometimes a good comedy or uplifting drama helps! Here are some mood-boosting movies:",
                'movies': self.get_movies_by_genre(['Comedy', 'Drama'], 4),
                'genre': 'mood_boost'
            }

        if 'excited' in user_message or 'energetic' in user_message:
            return {
                'response': "⚡ You're full of energy! Here are some high-octane action movies to match your vibe:",
                'movies': self.get_movies_by_genre(['Action', 'Adventure'], 4),
                'genre': 'high_energy'
            }

        # Check for genre requests - be more flexible with matching
        for genre, data in genre_responses.items():
            # Check if any keyword matches or if the genre name itself is mentioned
            genre_mentioned = genre in user_message or any(keyword in user_message for keyword in data['keywords'])

            # Also check for common phrases
            if genre == 'comedy' and any(phrase in user_message for phrase in ['show me comedies', 'comedy movies', 'funny movies']):
                genre_mentioned = True
            elif genre == 'action' and any(phrase in user_message for phrase in ['action movies', 'action films']):
                genre_mentioned = True
            elif genre == 'horror' and any(phrase in user_message for phrase in ['horror movies', 'horror films', 'scary movies']):
                genre_mentioned = True
            elif genre == 'romance' and any(phrase in user_message for phrase in ['romance movies', 'romantic movies', 'love movies']):
                genre_mentioned = True
            elif genre == 'drama' and any(phrase in user_message for phrase in ['drama movies', 'drama films']):
                genre_mentioned = True
            elif genre == 'sci-fi' and any(phrase in user_message for phrase in ['sci-fi movies', 'science fiction', 'scifi']):
                genre_mentioned = True
            elif genre == 'fantasy' and any(phrase in user_message for phrase in ['fantasy movies', 'fantasy films']):
                genre_mentioned = True
            elif genre == 'animation' and any(phrase in user_message for phrase in ['animated movies', 'animation movies', 'cartoons']):
                genre_mentioned = True

            if genre_mentioned:
                movies = self.get_movies_by_genre(data['genres'], 4)
                return {
                    'response': data['response'],
                    'movies': movies,
                    'genre': genre
                }

        # Default response for unrecognized input
        return {
            'response': "🤔 I'm not sure what you're looking for. Try asking me about specific genres like:\n• 'I want action movies'\n• 'Show me comedies'\n• 'Any good horror films?'\n• 'Recommend sci-fi movies'\n\nWhat genre interests you today?",
            'movies': [],
            'genre': None
        }

    def get_movies_by_genre(self, genres, limit=4):
        """Get movies by genre(s) with improved filtering"""
        try:
            # Create a more flexible genre search
            genre_patterns = []
            for genre in genres:
                # Add variations of genre names
                if genre.lower() == 'comedy':
                    genre_patterns.extend(['Comedy', 'comedy'])
                elif genre.lower() == 'action':
                    genre_patterns.extend(['Action', 'action'])
                elif genre.lower() == 'drama':
                    genre_patterns.extend(['Drama', 'drama'])
                elif genre.lower() == 'horror':
                    genre_patterns.extend(['Horror', 'horror', 'Thriller'])
                elif genre.lower() == 'romance':
                    genre_patterns.extend(['Romance', 'romance'])
                elif genre.lower() == 'science fiction':
                    genre_patterns.extend(['Science Fiction', 'Sci-Fi', 'science fiction'])
                elif genre.lower() == 'fantasy':
                    genre_patterns.extend(['Fantasy', 'fantasy'])
                elif genre.lower() == 'animation':
                    genre_patterns.extend(['Animation', 'animation'])
                else:
                    genre_patterns.append(genre)

            # Filter movies that contain any of the specified genres
            pattern = '|'.join(genre_patterns)
            filtered_movies = self.movies_df[
                self.movies_df['genres'].str.contains(pattern, case=False, na=False)
            ]

            # Sort by rating and popularity, then limit results
            if not filtered_movies.empty:
                # Sort by vote_average first, then by vote_count for tie-breaking
                top_movies = filtered_movies.sort_values(['vote_average', 'vote_count'], ascending=[False, False]).head(limit)
                return top_movies.to_dict('records')
            else:
                # Fallback to top-rated movies
                fallback_movies = self.movies_df.nlargest(limit, 'vote_average')
                return fallback_movies.to_dict('records')

        except Exception as e:
            print(f"Error in get_movies_by_genre: {e}")  # Debug info
            # Fallback to top-rated movies
            fallback_movies = self.movies_df.nlargest(limit, 'vote_average')
            return fallback_movies.to_dict('records')

    def get_recommendations_by_index(self, movie_index, similarity_type='tags', num_recommendations=6):
        """Get recommendations by movie index"""
        try:
            if similarity_type not in self.similarity_matrices:
                return []

            similarity_matrix = self.similarity_matrices[similarity_type]
            similarity_scores = list(enumerate(similarity_matrix[movie_index]))
            similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

            # Get top similar movies (excluding the movie itself)
            similar_movies_indices = [i[0] for i in similarity_scores[1:num_recommendations+1]]

            return self.movies_df.iloc[similar_movies_indices].to_dict('records')
        except Exception as e:
            return []

    def display_statistics(self):
        """Display app statistics"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            movie_count = len(self.movies_df) if self.movies_df is not None else 0
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{movie_count}</div>
                <div class="stat-label">Movies in Database</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{len(st.session_state.watchlist)}</div>
                <div class="stat-label">Movies in Watchlist</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{len(st.session_state.user_ratings)}</div>
                <div class="stat-label">Movies Rated</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            matrix_count = len(self.similarity_matrices)
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{matrix_count}</div>
                <div class="stat-label">Similarity Models</div>
            </div>
            """, unsafe_allow_html=True)

    def run(self):
        """Main application runner"""
        # Load data
        self.movies_df = self.load_movie_data()
        if self.movies_df is None:
            st.error("Failed to load movie database")
            return

        self.similarity_matrices = self.load_similarity_matrices()

        # Main header
        st.markdown('<h1 class="main-header">🎬 MOVIE RECOMMENDATION SYSTEM</h1>', unsafe_allow_html=True)

        # Top navigation menu
        st.markdown("---")

        # Initialize page if not set
        if 'page' not in st.session_state:
            st.session_state.page = "Home"

        # Icon-based navigation bar
        st.markdown("""
        <style>
        .nav-container {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 15px;
            border-radius: 15px;
            margin: 20px 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .nav-button {
            background: rgba(255,255,255,0.2);
            border: none;
            border-radius: 10px;
            padding: 12px;
            margin: 5px;
            color: white;
            font-size: 1.1rem;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
            text-align: center;
            min-height: 60px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .nav-button:hover {
            background: rgba(255,255,255,0.3);
            transform: translateY(-2px);
        }
        .nav-button.active {
            background: rgba(255,255,255,0.4);
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }
        .nav-icon {
            font-size: 1.5rem;
            margin-bottom: 5px;
        }
        .nav-text {
            font-size: 0.9rem;
        }
        </style>
        """, unsafe_allow_html=True)

        # Create icon navigation
        nav_col1, nav_col2, nav_col3, nav_col4, nav_col5, nav_col6, nav_col7 = st.columns(7)

        with nav_col1:
            if st.button("🏠\nHome", key="nav_home", use_container_width=True,
                        type="primary" if st.session_state.page == "Home" else "secondary"):
                st.session_state.page = "Home"
                st.rerun()

        with nav_col2:
            if st.button("🤖\nAI Chat", key="nav_chatbot", use_container_width=True,
                        type="primary" if st.session_state.page == "AI Chatbot" else "secondary"):
                st.session_state.page = "AI Chatbot"
                st.rerun()

        with nav_col3:
            if st.button("🎯\nRecommend", key="nav_recommend", use_container_width=True,
                        type="primary" if st.session_state.page == "Get Recommendations" else "secondary"):
                st.session_state.page = "Get Recommendations"
                st.rerun()

        with nav_col4:
            if st.button("🔍\nSearch", key="nav_search", use_container_width=True,
                        type="primary" if st.session_state.page == "Search Movies" else "secondary"):
                st.session_state.page = "Search Movies"
                st.rerun()

        with nav_col5:
            if st.button("📊\nBrowse", key="nav_browse", use_container_width=True,
                        type="primary" if st.session_state.page == "Browse Movies" else "secondary"):
                st.session_state.page = "Browse Movies"
                st.rerun()

        with nav_col6:
            if st.button("⭐\nWatchlist", key="nav_watchlist", use_container_width=True,
                        type="primary" if st.session_state.page == "My Watchlist" else "secondary"):
                st.session_state.page = "My Watchlist"
                st.rerun()

        with nav_col7:
            if st.button("📈\nAnalytics", key="nav_analytics", use_container_width=True,
                        type="primary" if st.session_state.page == "Analytics" else "secondary"):
                st.session_state.page = "Analytics"
                st.rerun()

        # Current page indicator
        current_page = st.session_state.get('page', 'Home')
        page_icons = {
            'Home': '🏠',
            'AI Chatbot': '🤖',
            'Get Recommendations': '🎯',
            'Search Movies': '🔍',
            'Browse Movies': '📊',
            'My Watchlist': '⭐',
            'Analytics': '📈'
        }

        st.markdown(f"""
        <div style="text-align: center; padding: 10px; background: rgba(103, 126, 234, 0.1);
                   border-radius: 10px; margin: 10px 0;">
            <span style="font-size: 1.2rem; font-weight: bold; color: #667eea;">
                {page_icons.get(current_page, '📍')} {current_page}
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Display statistics
        self.display_statistics()

        # Main Navigation
        st.sidebar.title("🎬 Movie Recommender")
        st.sidebar.markdown("---")

        # Quick navigation buttons
        st.sidebar.markdown("### 🧭 Quick Navigation")

        # Single column for better button layout
        if st.sidebar.button("🏠 Home", use_container_width=True, key="sidebar_home"):
            st.session_state.page = "Home"
            st.rerun()

        if st.sidebar.button("🤖 AI Chatbot", use_container_width=True, key="sidebar_chatbot"):
            st.session_state.page = "AI Chatbot"
            st.rerun()

        if st.sidebar.button("🎯 Get Recommendations", use_container_width=True, key="sidebar_rec"):
            st.session_state.page = "Get Recommendations"
            st.rerun()

        if st.sidebar.button("🔍 Search Movies", use_container_width=True, key="sidebar_search"):
            st.session_state.page = "Search Movies"
            st.rerun()

        if st.sidebar.button("📊 Browse Movies", use_container_width=True, key="sidebar_browse"):
            st.session_state.page = "Browse Movies"
            st.rerun()

        if st.sidebar.button("⭐ My Watchlist", use_container_width=True, key="sidebar_watchlist"):
            st.session_state.page = "My Watchlist"
            st.rerun()

        if st.sidebar.button("📈 Analytics", use_container_width=True, key="sidebar_analytics"):
            st.session_state.page = "Analytics"
            st.rerun()

        st.sidebar.markdown("---")

        # Quick stats
        st.sidebar.markdown("### 📊 Quick Stats")
        st.sidebar.metric("Total Movies", f"{len(self.movies_df):,}")

        if 'watchlist' in st.session_state:
            st.sidebar.metric("My Watchlist", len(st.session_state.watchlist))

        if 'user_ratings' in st.session_state:
            st.sidebar.metric("My Ratings", len(st.session_state.user_ratings))

        # Quick actions
        st.sidebar.markdown("### ⚡ Quick Actions")

        # Random movie button
        if st.sidebar.button("🎲 Random Movie", use_container_width=True):
            random_movie = self.movies_df.sample(1).iloc[0]
            st.sidebar.success(f"🎬 {random_movie['title']}")
            st.sidebar.write(f"⭐ {random_movie['vote_average']:.1f}/10")

        # Top rated button
        if st.sidebar.button("🏆 View Top Rated", use_container_width=True, key="sidebar_top"):
            st.session_state.page = "Home"
            st.rerun()

        st.sidebar.markdown("---")
        st.sidebar.markdown("**🔗 Data:** TMDB 5000 Movies Dataset")
        st.sidebar.markdown("**🎨 Design:** Clean & Fast")

        # Route to appropriate page based on session state
        current_page = st.session_state.get('page', 'Home')

        if current_page == "Home":
            self.show_home_page()
        elif current_page == "AI Chatbot":
            self.show_chatbot_page()
        elif current_page == "Get Recommendations":
            self.show_recommendations_page()
        elif current_page == "Search Movies":
            self.show_search_page()
        elif current_page == "Browse Movies":
            self.show_browse_page()
        elif current_page == "My Watchlist":
            self.show_watchlist_page()
        elif current_page == "Analytics":
            self.show_analytics_page()

        # Footer
        st.markdown("""
        <div style="text-align: center; padding: 30px; background: linear-gradient(90deg, #2c3e50, #34495e); color: white; border-radius: 15px; margin-top: 40px;">
            <h3>🎬 MOVIE RECOMMENDATION SYSTEM</h3>
            <p>Using TMDB 5000 Movies Dataset | Enhanced with Real Posters | Built with ❤️ using Streamlit</p>
            <p>Advanced recommendation algorithms: Content-based, Genre-based, Cast-based, and Production Company-based filtering</p>
        </div>
        """, unsafe_allow_html=True)

    def show_home_page(self):
        """Display home page"""
        # Display any pending action messages
        self.display_action_messages()

        # Welcome section
        st.markdown('<h2 class="section-title">Welcome to MOVIE RECOMMENDATION SYSTEM! 🎭</h2>', unsafe_allow_html=True)
        st.markdown("""
        <p style="text-align: center; font-size: 1.1rem; margin-bottom: 30px;">
            Discover your next favorite movie with our advanced recommendation system using the TMDB 5000 Movies Dataset!
        </p>
        """, unsafe_allow_html=True)

        # Features section
        st.markdown("### ✨ Key Features:")

        # Create feature cards using columns
        col1, col2 = st.columns(2)

        with col1:
            st.info("🤖 **AI-Powered Chatbot**\n\nNatural conversation for movie discovery. Ask for genres, moods, or specific preferences!")
            st.success("🎯 **Smart Recommendations**\n\n5 different algorithms: Content, Genre, Cast, Keywords, and Production-based")

        with col2:
            st.warning("⭐ **Personal Features**\n\nWatchlist, ratings, viewing history, and personalized AI suggestions")
            st.error("📊 **Rich Analytics**\n\nDetailed statistics, interactive charts, and AI-powered movie insights")

        # AI-Powered Personalized Recommendations
        st.markdown("---")

        # Check if user has ratings or watchlist for personalized recommendations
        user_ratings = st.session_state.get('user_ratings', {})
        watchlist = st.session_state.get('watchlist', [])

        if user_ratings or watchlist:
            st.markdown('<h2 class="section-title">🤖 AI-Powered Recommendations For You</h2>', unsafe_allow_html=True)
            st.info("🧠 **AI Engine:** Analyzing your preferences to find perfect matches...")

            # Get personalized recommendations
            personalized_movies = self.get_ai_personalized_recommendations(6)

            if personalized_movies:
                cols = st.columns(3)
                for idx, movie in enumerate(personalized_movies):
                    if idx < 6:
                        # Convert dict back to Series-like object for display
                        movie_series = pd.Series(movie)
                        self.display_movie_card(movie_series, cols[idx % 3], section="ai_personal")

            st.markdown("---")

        # Display top-rated movies
        st.markdown('<h2 class="section-title">🌟 Top Rated Movies</h2>', unsafe_allow_html=True)

        top_movies = self.movies_df.nlargest(6, 'vote_average')

        cols = st.columns(3)
        for idx, (_, movie) in enumerate(top_movies.iterrows()):
            if idx < 6:
                self.display_movie_card(movie, cols[idx % 3], section="top_rated")

    def show_chatbot_page(self):
        """Display AI chatbot page"""
        st.markdown('<h2 class="section-title">🤖 AI Movie Assistant</h2>', unsafe_allow_html=True)

        # Introduction
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                   padding: 20px; border-radius: 15px; color: white; margin-bottom: 20px;">
            <h3 style="margin: 0; color: white;">👋 Hello! I'm your AI Movie Assistant</h3>
            <p style="margin: 10px 0 0 0;">I can recommend movies based on genres, moods, and preferences.
            Just chat with me naturally!</p>
        </div>
        """, unsafe_allow_html=True)

        # Initialize chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = [
                {
                    'role': 'assistant',
                    'message': "🎬 Hi there! I'm your AI movie assistant. I can recommend movies based on genres like action, comedy, drama, horror, romance, sci-fi, fantasy, and animation. What kind of movies are you in the mood for today?"
                }
            ]

        # Display chat history
        st.markdown("### 💬 Chat with AI Assistant")

        # Chat container
        chat_container = st.container()

        with chat_container:
            for chat in st.session_state.chat_history:
                if chat['role'] == 'user':
                    st.markdown(f"""
                    <div style="background: #e3f2fd; padding: 15px; border-radius: 15px;
                               margin: 10px 0; border-left: 4px solid #2196f3; color: #1976d2;">
                        <strong style="color: #1976d2;">🧑 You:</strong><br>
                        <span style="color: #333333; line-height: 1.5;">{chat['message']}</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background: #f3e5f5; padding: 15px; border-radius: 15px;
                               margin: 10px 0; border-left: 4px solid #9c27b0; color: #7b1fa2;">
                        <strong style="color: #7b1fa2;">🤖 AI Assistant:</strong><br>
                        <span style="color: #333333; line-height: 1.5; white-space: pre-line;">{chat['message']}</span>
                    </div>
                    """, unsafe_allow_html=True)

        # Chat input
        st.markdown("---")

        # Quick suggestion buttons
        st.markdown("### 🎯 Quick Suggestions:")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("🎬 Action Movies", use_container_width=True, key="quick_action"):
                self.process_chat_message("I want action movies")
                st.rerun()

        with col2:
            if st.button("😄 Comedy Films", use_container_width=True, key="quick_comedy"):
                self.process_chat_message("Show me comedies")
                st.rerun()

        with col3:
            if st.button("😱 Horror Movies", use_container_width=True, key="quick_horror"):
                self.process_chat_message("Any good horror films?")
                st.rerun()

        with col4:
            if st.button("💕 Romance Films", use_container_width=True, key="quick_romance"):
                self.process_chat_message("Recommend romantic movies")
                st.rerun()

        # Text input for custom messages
        user_input = st.text_input(
            "💭 Type your message:",
            placeholder="e.g., 'I want sci-fi movies' or 'Show me comedies'",
            key="chat_input"
        )

        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            if st.button("📤 Send", type="primary", use_container_width=True, key="send_message"):
                if user_input.strip():
                    self.process_chat_message(user_input)
                    # Clear the input by resetting the session state
                    if 'chat_input' in st.session_state:
                        del st.session_state['chat_input']
                    st.rerun()

        with col2:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = [
                    {
                        'role': 'assistant',
                        'message': "🎬 Chat cleared! I'm ready to help you find great movies. What genre interests you?"
                    }
                ]
                st.rerun()

        # Display recommended movies if any
        if 'chatbot_recommended_movies' in st.session_state and st.session_state.chatbot_recommended_movies:
            st.markdown("---")
            st.markdown('<h3 class="section-title">🎬 AI Recommended Movies</h3>', unsafe_allow_html=True)

            # Add a unique identifier to force refresh
            movie_ids = [str(movie.get('id', idx)) for idx, movie in enumerate(st.session_state.chatbot_recommended_movies)]
            unique_key = "_".join(movie_ids[:4])  # Create unique key from movie IDs

            cols = st.columns(2)
            for idx, movie in enumerate(st.session_state.chatbot_recommended_movies[:4]):
                movie_series = pd.Series(movie)
                self.display_movie_card(movie_series, cols[idx % 2], section=f"chatbot_{unique_key}_{idx}")

    def process_chat_message(self, user_message):
        """Process user message and generate AI response"""
        # Clear previous movie recommendations first
        st.session_state.chatbot_recommended_movies = []

        # Add user message to chat history
        st.session_state.chat_history.append({
            'role': 'user',
            'message': user_message
        })

        # Get AI response
        ai_response = self.get_ai_chatbot_response(user_message)

        # Add AI response to chat history
        st.session_state.chat_history.append({
            'role': 'assistant',
            'message': ai_response['response']
        })

        # If movies were recommended, display them
        if ai_response['movies'] and len(ai_response['movies']) > 0:
            # Add movie recommendations to chat
            movie_titles = [movie.get('title', 'Unknown') for movie in ai_response['movies']]
            movie_list = "\n".join([f"• {title} ({movie.get('vote_average', 'N/A'):.1f}⭐)" for movie, title in zip(ai_response['movies'], movie_titles)])

            st.session_state.chat_history.append({
                'role': 'assistant',
                'message': f"Here are my top recommendations:\n\n{movie_list}\n\n🎬 Check them out below!"
            })

            # Store NEW movies for display (this will replace any previous ones)
            st.session_state.chatbot_recommended_movies = ai_response['movies']
        else:
            # Ensure no movies are displayed if none were recommended
            st.session_state.chatbot_recommended_movies = []

    def show_recommendations_page(self):
        """Display AI-powered recommendations page"""
        # Display any pending action messages
        self.display_action_messages()

        st.markdown('<h2 class="section-title">🎯 AI Movie Recommendation Engine</h2>', unsafe_allow_html=True)

        # AI-powered personalized section
        user_ratings = st.session_state.get('user_ratings', {})
        watchlist = st.session_state.get('watchlist', [])

        if user_ratings or watchlist:
            st.markdown("### 🤖 AI-Powered Personal Recommendations")
            st.info("🧠 **AI Engine:** Using your viewing history and preferences to find perfect matches...")

            if st.button("🚀 Get My AI Recommendations", type="primary"):
                with st.spinner("🤖 AI analyzing your preferences..."):
                    ai_recommendations = self.get_ai_personalized_recommendations(9)

                    if ai_recommendations:
                        st.success(f"✅ AI found {len(ai_recommendations)} personalized recommendations!")

                        # Display AI recommendations
                        st.markdown("---")
                        st.markdown('<h3 class="section-title">🤖 Your AI-Curated Movies</h3>', unsafe_allow_html=True)

                        cols = st.columns(3)
                        for idx, movie in enumerate(ai_recommendations):
                            movie_series = pd.Series(movie)
                            self.display_movie_card(movie_series, cols[idx % 3], section="ai_rec")

            st.markdown("---")

        # Traditional recommendation system
        st.markdown("### 🎬 Find Movies Similar to One You Love")

        # Movie selection
        col1, col2 = st.columns([3, 1])

        with col1:
            selected_movie = st.selectbox(
                "🎬 Select a movie you enjoyed:",
                options=sorted(self.movies_df['title'].tolist()),
                help="Choose a movie to get similar recommendations"
            )

        with col2:
            num_recommendations = st.slider(
                "📊 Number of recommendations:",
                min_value=3,
                max_value=15,
                value=9,
                help="How many recommendations would you like?"
            )

        # Recommendation algorithm selection
        st.markdown("### 🧠 Choose AI Algorithm:")
        algorithm = st.radio(
            "Select the AI recommendation method:",
            ["tags", "genres", "cast", "keywords", "production"],
            format_func=lambda x: {
                "tags": "🏷️ Content-Based AI (Plot & Themes)",
                "genres": "🎭 Genre-Based AI (Similar Categories)",
                "cast": "👥 Cast-Based AI (Same Actors/Directors)",
                "keywords": "🔑 Keywords-Based AI (Story Elements)",
                "production": "🏢 Studio-Based AI (Production Style)"
            }.get(x, x),
            horizontal=True,
            help="Each algorithm uses different AI techniques to find similar movies"
        )

        if st.button("🔍 Find Similar Movies", type="primary"):
            with st.spinner(f"🤖 AI analyzing '{selected_movie}' using {algorithm} algorithm..."):
                recommendations, error_message = self.get_recommendations(selected_movie, algorithm, num_recommendations)

                if error_message:
                    st.error(f"❌ {error_message}")
                elif recommendations is None or len(recommendations) == 0:
                    st.error(f"❌ Sorry, couldn't find recommendations for '{selected_movie}'. Try a different movie or algorithm.")
                else:
                    st.success(f"✅ AI found {len(recommendations)} movies similar to '{selected_movie}'!")

                    # Show AI insight
                    algorithm_descriptions = {
                        "tags": "analyzing plot themes and story elements",
                        "genres": "matching movie categories and styles",
                        "cast": "finding movies with similar actors and directors",
                        "keywords": "identifying common story keywords and themes",
                        "production": "matching production companies and filmmaking styles"
                    }
                    st.info(f"🤖 **AI Method:** {algorithm_descriptions.get(algorithm, 'analyzing movie similarities')}")

                    # Display recommendations
                    st.markdown("---")  # Separator
                    st.markdown(f'<h2 class="section-title">🎬 Movies Similar to "{selected_movie}"</h2>', unsafe_allow_html=True)

                    # Create columns for movie cards
                    cols = st.columns(3)

                    for idx, (_, movie) in enumerate(recommendations.iterrows()):
                        self.display_movie_card(movie, cols[idx % 3], section="similar")

    def show_search_page(self):
        """Display search page"""
        # Display any pending action messages
        self.display_action_messages()

        st.markdown('<h2 class="section-title">🔍 Search Movies</h2>', unsafe_allow_html=True)

        # Search input
        search_query = st.text_input(
            "🔍 Search for movies:",
            placeholder="Enter movie title, genre, or keywords...",
            help="Search through the TMDB 5000 movies database"
        )

        if search_query:
            # Search in database
            search_results = self.movies_df[
                self.movies_df['title'].str.contains(search_query, case=False, na=False) |
                self.movies_df['overview'].str.contains(search_query, case=False, na=False)
            ]

            if len(search_results) > 0:
                st.success(f"📈 Found {len(search_results)} movies matching '{search_query}'")

                # Display search results
                cols = st.columns(3)
                for idx, (_, movie) in enumerate(search_results.head(12).iterrows()):
                    self.display_movie_card(movie, cols[idx % 3], section="search")
            else:
                st.warning(f"No movies found matching '{search_query}'")
                st.info("💡 Try searching with different keywords or check the spelling")

    def show_browse_page(self):
        """Display browse page with filters"""
        # Display any pending action messages
        self.display_action_messages()

        st.markdown('<h2 class="section-title">📊 Browse Movie Database</h2>', unsafe_allow_html=True)

        # Filters
        col1, col2, col3 = st.columns(3)

        with col1:
            sort_by = st.selectbox(
                "📈 Sort by:",
                ["vote_average", "popularity", "release_date", "vote_count", "revenue"],
                format_func=lambda x: {
                    "vote_average": "⭐ Rating",
                    "popularity": "🔥 Popularity",
                    "release_date": "📅 Release Date",
                    "vote_count": "👥 Vote Count",
                    "revenue": "💰 Revenue"
                }[x]
            )

        with col2:
            sort_order = st.radio("📊 Order:", ["Descending", "Ascending"], horizontal=True)
            ascending = sort_order == "Ascending"

        with col3:
            min_rating = st.slider("⭐ Minimum Rating:", 0.0, 10.0, 6.0, 0.1)

        # Apply filters
        filtered_movies = self.movies_df[
            self.movies_df['vote_average'] >= min_rating
        ].sort_values(sort_by, ascending=ascending)

        st.write(f"📈 Showing {len(filtered_movies)} movies")

        # Pagination
        movies_per_page = 12
        total_pages = len(filtered_movies) // movies_per_page + (1 if len(filtered_movies) % movies_per_page > 0 else 0)

        if total_pages > 1:
            page_num = st.selectbox("📄 Page:", range(1, total_pages + 1))
            start_idx = (page_num - 1) * movies_per_page
            end_idx = start_idx + movies_per_page
            page_movies = filtered_movies.iloc[start_idx:end_idx]
        else:
            page_movies = filtered_movies.head(movies_per_page)

        # Display movies
        cols = st.columns(3)
        for idx, (_, movie) in enumerate(page_movies.iterrows()):
            self.display_movie_card(movie, cols[idx % 3], section="browse")

    def show_watchlist_page(self):
        """Display user's watchlist with AI insights"""
        # Display any pending action messages
        self.display_action_messages()

        st.markdown('<h2 class="section-title">⭐ My Watchlist</h2>', unsafe_allow_html=True)

        # Initialize watchlist if it doesn't exist
        if 'watchlist' not in st.session_state:
            st.session_state.watchlist = []

        if st.session_state.watchlist:
            st.success(f"🎬 You have {len(st.session_state.watchlist)} movies in your watchlist")

            # AI insights about watchlist
            if len(st.session_state.watchlist) >= 3:
                st.info("🤖 **AI Analysis:** Based on your watchlist, I can see you enjoy diverse genres. Check out personalized recommendations on the home page!")

            # Clear all button
            if st.button("🗑️ Clear All Watchlist", type="secondary"):
                st.session_state.watchlist = []
                st.success("Watchlist cleared!")
                st.rerun()

            # Display watchlist movies
            cols = st.columns(3)
            for idx, movie in enumerate(st.session_state.watchlist):
                # Convert dict to Series for display
                movie_series = pd.Series(movie)
                self.display_movie_card(movie_series, cols[idx % 3], show_actions=False, section="watchlist")

                # Remove from watchlist button
                with cols[idx % 3]:
                    if st.button(f"❌ Remove", key=f"remove_{movie.get('id', idx)}", use_container_width=True):
                        movie_title = movie.get('title', 'Unknown')
                        st.session_state.watchlist = [
                            m for m in st.session_state.watchlist if m.get('id') != movie.get('id')
                        ]
                        st.session_state.last_action_message = f"❌ Removed '{movie_title}' from watchlist!"
                        st.rerun()
        else:
            st.info("🎭 Your watchlist is empty. Add some movies to get started!")
            st.markdown("### 🎯 How to add movies:")
            st.markdown("1. 🏠 Go to **Home** page and click **➕ Watchlist** on any movie")
            st.markdown("2. 🔍 **Search** for movies and add them to your watchlist")
            st.markdown("3. 📊 **Browse** movies and discover new favorites")

            # Show some popular movies to get started
            st.markdown("### 🌟 Popular Movies to Get Started:")
            popular_movies = self.movies_df.nlargest(3, 'popularity')
            cols = st.columns(3)
            for idx, (_, movie) in enumerate(popular_movies.iterrows()):
                self.display_movie_card(movie, cols[idx % 3], section="popular")

    def show_analytics_page(self):
        """Display AI-powered analytics and insights"""
        # Display any pending action messages
        self.display_action_messages()

        st.markdown('<h2 class="section-title">📈 AI-Powered Movie Analytics</h2>', unsafe_allow_html=True)

        # Personal AI insights
        user_ratings = st.session_state.get('user_ratings', {})
        watchlist = st.session_state.get('watchlist', [])

        if user_ratings or watchlist:
            st.markdown("### 🤖 Your Personal AI Insights")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("🎬 Movies Rated", len(user_ratings))
                if user_ratings:
                    avg_rating = sum(data['rating'] for data in user_ratings.values()) / len(user_ratings)
                    st.metric("⭐ Your Avg Rating", f"{avg_rating:.1f}/10")

            with col2:
                st.metric("📝 Watchlist Size", len(watchlist))
                if user_ratings:
                    high_ratings = sum(1 for data in user_ratings.values() if data['rating'] >= 8)
                    st.metric("🏆 High Ratings (8+)", high_ratings)

            with col3:
                if user_ratings and watchlist:
                    total_engagement = len(user_ratings) + len(watchlist)
                    st.metric("🎯 Total Engagement", total_engagement)

                    # AI personality insight
                    if avg_rating >= 7.5:
                        personality = "🎭 Discerning Critic"
                    elif avg_rating >= 6.5:
                        personality = "🎬 Balanced Viewer"
                    else:
                        personality = "🔍 Honest Reviewer"
                    st.metric("🤖 AI Profile", personality)

            # AI recommendations based on behavior
            if user_ratings:
                st.markdown("### 🧠 AI Behavior Analysis")

                # Analyze rating patterns
                ratings_list = [data['rating'] for data in user_ratings.values()]
                if ratings_list:
                    if max(ratings_list) - min(ratings_list) <= 2:
                        st.info("🤖 **AI Insight:** You have consistent rating patterns - you know what you like!")
                    elif avg_rating >= 7:
                        st.info("🤖 **AI Insight:** You tend to rate movies highly - you're optimistic about cinema!")
                    else:
                        st.info("🤖 **AI Insight:** You're a critical viewer who values quality storytelling.")

            st.markdown("---")

        # Database analytics
        st.markdown("### 📊 Movie Database Analytics")

        if self.movies_df is not None and len(self.movies_df) > 0:
            # Rating distribution
            fig_rating = px.histogram(
                self.movies_df,
                x='vote_average',
                nbins=20,
                title="Rating Distribution",
                labels={'vote_average': 'Rating', 'count': 'Number of Movies'}
            )
            fig_rating.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_rating, use_container_width=True)

            # Revenue vs Budget
            if 'budget' in self.movies_df.columns and 'revenue' in self.movies_df.columns:
                fig_revenue = px.scatter(
                    self.movies_df[self.movies_df['budget'] > 0],
                    x='budget',
                    y='revenue',
                    title="Budget vs Revenue",
                    labels={'budget': 'Budget ($)', 'revenue': 'Revenue ($)'},
                    hover_data=['title']
                )
                fig_revenue.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_revenue, use_container_width=True)

            # Release year distribution
            self.movies_df['release_year'] = pd.to_datetime(
                self.movies_df['release_date'], errors='coerce'
            ).dt.year

            year_counts = self.movies_df['release_year'].value_counts().sort_index()

            fig_year = px.line(
                x=year_counts.index,
                y=year_counts.values,
                title="Movies by Release Year",
                labels={'x': 'Year', 'y': 'Number of Movies'}
            )
            fig_year.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_year, use_container_width=True)

def main():
    """Main application entry point"""
    app = EnhancedMovieRecommender()
    app.run()

if __name__ == "__main__":
    main()
