# 🎬 Movie Recommendation System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)](https://streamlit.io)
[![TMDB](https://img.shields.io/badge/Data-TMDB%205000-yellow.svg)](https://www.themoviedb.org/)

> **Discover your next favorite movie with our AI-powered recommendation system!**

An advanced, feature-rich movie recommendation system built with Python and Streamlit, featuring multiple recommendation algorithms, AI-powered chatbot, personalized user profiles, and comprehensive analytics.

## 🌟 Key Features

### 🤖 AI-Powered Recommendations
- **Smart Chatbot**: Natural language conversation for movie discovery
- **Multiple Algorithms**: 5 different recommendation engines
  - Content-based filtering (tags, genres, keywords)
  - Cast-based recommendations
  - Production company-based suggestions
  - Hybrid collaborative filtering
  - Popularity-based recommendations

### 👤 Personalized Experience
- **User Profiles**: Personal watchlist and rating system
- **Viewing History**: Track your movie journey
- **AI Learning**: System learns from your preferences
- **Custom Recommendations**: Tailored suggestions based on your taste

### 📊 Rich Analytics & Insights
- **Interactive Charts**: Visualize movie trends and statistics
- **Genre Analysis**: Explore movie distribution by genres
- **Rating Insights**: Analyze movie ratings and popularity
- **Personal Statistics**: Track your viewing patterns

### 🎨 Modern UI/UX
- **Responsive Design**: Beautiful, mobile-friendly interface
- **Real-time Updates**: Dynamic content loading
- **Interactive Elements**: Hover effects and smooth animations
- **Clean Layout**: Intuitive navigation and user experience

## 🏗️ Project Architecture

```
Movie Recommendation System/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── Files/                          # Data and model files
│   ├── tmdb_5000_movies.csv       # Movie dataset
│   ├── tmdb_5000_credits.csv      # Credits dataset
│   ├── similarity_*.pkl           # Pre-computed similarity matrices
│   └── *.pkl                      # Processed data files
├── services/                       # Core business logic
│   ├── recommendation_engine.py   # Hybrid recommendation algorithms
│   ├── tmdb_api.py                # TMDB API integration
│   └── user_preferences.py        # User profile management
├── processing/                     # Data processing modules
│   ├── preprocess.py              # Data preprocessing pipeline
│   └── display.py                 # UI display utilities
├── test_user_data/                # Sample user data
│   ├── user_ratings.json         # User rating history
│   ├── viewing_history.json      # Viewing history
│   └── watchlist.json            # User watchlist
└── models/                        # ML model cache
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Internet connection (for TMDB API)

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/movie-recommendation-system.git
   cd movie-recommendation-system
   ```

2. **Create Virtual Environment**
   ```bash
   # Windows
   python -m venv movie-env
   movie-env\Scripts\activate

   # macOS/Linux
   python3 -m venv movie-env
   source movie-env/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Required NLTK Data** (First time only)
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   ```

5. **Run the Application**
   ```bash
   streamlit run app.py
   ```

6. **Access the App**
   Open your browser and navigate to `http://localhost:8501`

## 📊 Dataset Information

This project uses the **TMDB 5000 Movie Dataset** which includes:

- **5,000 movies** with comprehensive metadata
- **Movie details**: Title, overview, genres, release date, runtime
- **Cast & Crew**: Actor names, character names, director information
- **Production**: Production companies, countries, languages
- **Ratings**: User ratings, vote counts, popularity scores
- **Keywords**: Movie tags and descriptive keywords

### Data Sources
- Primary: [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- API: [The Movie Database (TMDB) API](https://www.themoviedb.org/documentation/api)

## 🔧 Technical Implementation

### Machine Learning Algorithms

1. **Content-Based Filtering**
   - TF-IDF vectorization of movie features
   - Cosine similarity computation
   - Feature engineering from genres, keywords, cast

2. **Collaborative Filtering**
   - User-item interaction matrix
   - Matrix factorization techniques
   - Neighborhood-based methods

3. **Hybrid Approach**
   - Weighted combination of multiple algorithms
   - Dynamic weight adjustment based on data availability
   - Fallback mechanisms for cold start problems

### Key Technologies

- **Frontend**: Streamlit with custom CSS/HTML
- **Backend**: Python with pandas, scikit-learn
- **ML Libraries**: scikit-learn, numpy, pandas
- **Visualization**: Plotly, matplotlib
- **API Integration**: TMDB API for real-time data
- **Caching**: Streamlit caching for performance optimization

## 🎯 Usage Guide

### 1. Home Page
- View top-rated movies
- Get AI-powered personalized recommendations
- Quick access to all features

### 2. AI Chatbot
- Natural conversation: "I want action movies"
- Mood-based recommendations: "I'm feeling sad"
- Genre exploration: "Show me sci-fi films"

### 3. Get Recommendations
- Search for any movie
- Choose recommendation algorithm
- Get similar movies instantly

### 4. Search & Browse
- Advanced search functionality
- Filter by genre, year, rating
- Sort by various criteria

### 5. Personal Features
- Build your watchlist
- Rate movies (1-10 scale)
- Track viewing history
- View personalized analytics

## 📈 Performance Features

- **Caching**: Intelligent caching for faster load times
- **Lazy Loading**: On-demand data loading
- **Optimized Algorithms**: Efficient similarity computations
- **Session Management**: Persistent user state
- **Error Handling**: Graceful fallbacks and error recovery

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/

# Format code
black app.py services/ processing/

# Lint code
flake8 app.py services/ processing/
```



## 🙏 Acknowledgments

- **TMDB**: For providing the comprehensive movie dataset
- **Streamlit**: For the amazing web framework
- **scikit-learn**: For machine learning algorithms
- **Open Source Community**: For the incredible tools and libraries

## 📞 Support

If you encounter any issues or have questions:

1. **Check the Issues**: Look for existing solutions
2. **Create an Issue**: Report bugs or request features
3. **Documentation**: Refer to code comments and docstrings
4. **Community**: Join discussions in the repository

---

**⭐ Star this repository if you found it helpful!**

*Built with ❤️ using Python and Streamlit*