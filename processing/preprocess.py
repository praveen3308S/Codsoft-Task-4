import string
import pickle
import pandas as pd
import ast
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import threading
from contextlib import contextmanager

# Configuration
API_TIMEOUT = 15  # seconds - increased timeout for getting images
CONNECT_TIMEOUT = 10  # seconds - increased connection timeout
READ_TIMEOUT = 15  # seconds - increased read timeout
MAX_RETRIES = 2  # 2 retries for better success rate
RETRY_DELAY = 1  # seconds - reasonable delay
OFFLINE_MODE = False  # Set to True to disable all API calls and use placeholders

# Simple cache to avoid repeated API calls
_poster_cache = {}
_person_cache = {}
_api_available = True  # Track if API is available
_api_check_time = 0  # Last time we checked API availability
_api_check_interval = 300  # Check API availability every 5 minutes

# Create a session with proper timeout and retry configuration
def create_session():
    session = requests.Session()
    
    # Configure retry strategy - disable automatic retries to handle them manually
    retry_strategy = Retry(
        total=0,  # We'll handle retries manually
        connect=0,
        read=0,
        status=0,
        backoff_factor=0
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

# Global session instance
_session = create_session()

def check_api_availability():
    """Check if TMDb API is available - more lenient approach"""
    global _api_available, _api_check_time
    
    current_time = time.time()
    if current_time - _api_check_time < _api_check_interval:
        return _api_available
    
    try:
        # Try a simple API call to check availability with longer timeout
        response = _session.get(
            'https://api.themoviedb.org/3/configuration?api_key=6177b4297dff132d300422e0343471fb',
            timeout=(5, 10)  # More generous timeout for availability check
        )
        _api_available = response.status_code == 200
        if _api_available:
            print("✓ API is available - will fetch real movie posters")
        else:
            print("⚠ API returned error status - using placeholder images")
    except Exception as e:
        print(f"⚠ API check failed ({type(e).__name__}) - using placeholder images")
        _api_available = False
    
    _api_check_time = current_time
    return _api_available

# Object for porterStemmer
ps = PorterStemmer()
nltk.download('stopwords')
import streamlit as st


def get_genres(obj):
    lista = ast.literal_eval(obj)
    l1 = []
    for i in lista:
        l1.append(i['name'])
    return l1


def get_cast(obj):
    a = ast.literal_eval(obj)
    l_ = []
    len_ = len(a)
    for i in range(0, 10):
        if i < len_:
            l_.append(a[i]['name'])
    return l_


def get_crew(obj):
    l1 = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            l1.append(i['name'])
            break
    return l1


def read_csv_to_df():
    #  Reading both the csv files
    credit_ = pd.read_csv(r'Files/tmdb_5000_credits.csv')
    movies = pd.read_csv(r'Files/tmdb_5000_movies.csv')

    # Merging the dataframes
    movies = movies.merge(credit_, on='title')

    movies2 = movies
    movies2.drop(['homepage', 'tagline'], axis=1, inplace=True)
    movies2 = movies2[['movie_id', 'title', 'budget', 'overview', 'popularity', 'release_date', 'revenue', 'runtime',
                       'spoken_languages', 'status', 'vote_average', 'vote_count']]

    #  Extracting important and relevant features
    movies = movies[
        ['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew', 'production_companies', 'release_date']]
    movies.dropna(inplace=True)

    # df[df['column_name'] == some_condition]['target_column'] = new_value
    # df.loc[df['column_name'] == some_condition, 'target_column'] = new_value

    #  Applying functions to convert from list to only items.
    movies['genres'] = movies['genres'].apply(get_genres)
    movies['keywords'] = movies['keywords'].apply(get_genres)
    movies['top_cast'] = movies['cast'].apply(get_cast)
    movies['director'] = movies['crew'].apply(get_crew)
    movies['prduction_comp'] = movies['production_companies'].apply(get_genres)

    #  Removing spaces from between the lines
    movies['overview'] = movies['overview'].apply(lambda x: x.split())
    movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['tcast'] = movies['top_cast'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['tcrew'] = movies['director'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['tprduction_comp'] = movies['prduction_comp'].apply(lambda x: [i.replace(" ", "") for i in x])

    # Creating a tags where we have all the words together for analysis
    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['tcast'] + movies['tcrew']

    #  Creating new dataframe for the analysis part only.
    new_df = movies[['movie_id', 'title', 'tags', 'genres', 'keywords', 'tcast', 'tcrew', 'tprduction_comp']]

    # new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
    new_df['genres'] = new_df['genres'].apply(lambda x: " ".join(x))
    # new_df['keywords'] = new_df['keywords'].apply(lambda x: " ".join(x))
    new_df['tcast'] = new_df['tcast'].apply(lambda x: " ".join(x))
    new_df['tprduction_comp'] = new_df['tprduction_comp'].apply(lambda x: " ".join(x))

    new_df['tcast'] = new_df['tcast'].apply(lambda x: x.lower())
    new_df['genres'] = new_df['genres'].apply(lambda x: x.lower())
    new_df['tprduction_comp'] = new_df['tprduction_comp'].apply(lambda x: x.lower())

    #  Applying stemming on tags and tags and keywords
    new_df['tags'] = new_df['tags'].apply(stemming_stopwords)
    new_df['keywords'] = new_df['keywords'].apply(stemming_stopwords)

    return movies, new_df, movies2


def stemming_stopwords(li):
    ans = []

    # ps = PorterStemmer()

    for i in li:
        ans.append(ps.stem(i))

    # Removing Stopwords
    stop_words = set(stopwords.words('english'))
    filtered_sentence = []
    for w in ans:
        w = w.lower()
        if w not in stop_words:
            filtered_sentence.append(w)

    str_ = ''
    for i in filtered_sentence:
        if len(i) > 2:
            str_ = str_ + i + ' '

    # Removing Punctuations
    punc = string.punctuation
    str_.translate(str_.maketrans('', '', punc))
    return str_


def fetch_posters(movie_id, max_retries=MAX_RETRIES):
    """Fetch movie poster with robust timeout handling and caching"""
    # Check cache first
    if movie_id in _poster_cache:
        return _poster_cache[movie_id]
    
    # Create a more professional looking placeholder
    default_image = f"https://via.placeholder.com/780x1170/34495e/ecf0f1?text=🎬+Movie+Poster%0A%0AID:+{movie_id}%0A%0ARecommended+Film%0A%0APoster+Loading..."
    
    # Skip API call if in offline mode
    if OFFLINE_MODE:
        print(f"Offline mode: Using default image for movie_id {movie_id}")
        _poster_cache[movie_id] = default_image
        return default_image
    
    url = 'https://api.themoviedb.org/3/movie/{}?api_key=6177b4297dff132d300422e0343471fb'.format(movie_id)
    
    for attempt in range(max_retries):
        try:
            # Use tuple for timeout: (connect_timeout, read_timeout)
            response = _session.get(url, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
            response.raise_for_status()
            data = response.json()
            
            if 'poster_path' in data and data['poster_path']:
                poster_url = "https://image.tmdb.org/t/p/w780/" + data['poster_path']
                _poster_cache[movie_id] = poster_url
                return poster_url
            else:
                _poster_cache[movie_id] = default_image
                return default_image
                
        except (requests.exceptions.ConnectTimeout, requests.exceptions.Timeout) as e:
            print(f"Timeout for movie_id {movie_id} (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY)
                continue
        except requests.exceptions.ConnectionError as e:
            print(f"Connection error for movie_id {movie_id} (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY * 2)
                continue
        except requests.exceptions.RequestException as e:
            print(f"Request error for movie_id {movie_id}: {type(e).__name__}")
            break
        except (KeyError, TypeError, ValueError) as e:
            print(f"Data parsing error for movie_id {movie_id}: {e}")
            break
        except Exception as e:
            print(f"Unexpected error for movie_id {movie_id}: {type(e).__name__}")
            break
    
    # Cache and return default image if all attempts failed
    _poster_cache[movie_id] = default_image
    return default_image


def recommend(new_df, movie, pickle_file_path):
    """Recommend movies with improved error handling and progress tracking"""
    try:
        with open(pickle_file_path, 'rb') as pickle_file:
            similarity_tags = pickle.load(pickle_file)

        movie_idx = new_df[new_df['title'] == movie].index[0]

        # Getting the top 25 movies from the list which are most similar
        movie_list = sorted(list(enumerate(similarity_tags[movie_idx])), reverse=True, key=lambda x: x[1])[1:26]

        rec_movie_list = []
        rec_poster_list = []

        print(f"Fetching posters for {len(movie_list)} recommended movies...")
        
        for idx, i in enumerate(movie_list):
            try:
                movie_title = new_df.iloc[i[0]]['title']
                movie_id = new_df.iloc[i[0]]['movie_id']
                
                rec_movie_list.append(movie_title)
                poster_url = fetch_posters(movie_id)
                rec_poster_list.append(poster_url)
                
                # Progress indicator
                if (idx + 1) % 5 == 0:
                    print(f"Fetched {idx + 1}/{len(movie_list)} posters...")
                    
            except Exception as e:
                print(f"Error processing movie at index {i[0]}: {e}")
                # Add default values to maintain list consistency
                rec_movie_list.append("Unknown Movie")
                rec_poster_list.append("https://media.istockphoto.com/vectors/error-icon-vector-illustration-vector-id922024224?k=6&m=922024224&s=612x612&w=0&h=LXl8Ul7bria6auAXKIjlvb6hRHkAodTqyqBeA6K7R54=")

        return rec_movie_list, rec_poster_list
        
    except Exception as e:
        print(f"Error in recommend function: {e}")
        # Return empty lists if there's a critical error
        return [], []


def vectorise(new_df, col_name):
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vec_tags = cv.fit_transform(new_df[col_name]).toarray()
    sim_bt = cosine_similarity(vec_tags)
    return sim_bt


def fetch_person_details(id_, max_retries=MAX_RETRIES):
    """Fetch person details with robust timeout handling and caching"""
    # Check cache first
    if id_ in _person_cache:
        return _person_cache[id_]
    
    default_image = "https://media.istockphoto.com/vectors/error-icon-vector-illustration-vector-id922024224?k=6&m" \
                   "=922024224&s=612x612&w=0&h=LXl8Ul7bria6auAXKIjlvb6hRHkAodTqyqBeA6K7R54="
    
    url = 'https://api.themoviedb.org/3/person/{}?api_key=6177b4297dff132d300422e0343471fb'.format(id_)
    
    for attempt in range(max_retries):
        try:
            # Use tuple for timeout: (connect_timeout, read_timeout)
            response = _session.get(url, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
            response.raise_for_status()
            data = response.json()
            
            if 'profile_path' in data and data['profile_path']:
                profile_url = 'https://image.tmdb.org/t/p/w220_and_h330_face' + data['profile_path']
            else:
                profile_url = default_image

            biography = data.get('biography', " ") if data.get('biography') else " "
            
            result = (profile_url, biography)
            _person_cache[id_] = result
            return result
            
        except (requests.exceptions.ConnectTimeout, requests.exceptions.Timeout) as e:
            print(f"Timeout for person_id {id_} (attempt {attempt + 1}/{max_retries}): {type(e).__name__}")
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY)
                continue
        except requests.exceptions.ConnectionError as e:
            print(f"Connection error for person_id {id_} (attempt {attempt + 1}/{max_retries}): {str(e)[:100]}...")
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY * 2)
                continue
        except requests.exceptions.RequestException as e:
            print(f"Request error for person_id {id_}: {type(e).__name__} - {str(e)[:100]}...")
            break
        except (KeyError, TypeError, ValueError) as e:
            print(f"Data parsing error for person_id {id_}: {e}")
            break
        except Exception as e:
            print(f"Unexpected error for person_id {id_}: {type(e).__name__} - {str(e)[:100]}...")
            break
    
    # Cache and return default result if all attempts failed
    result = (default_image, "")
    _person_cache[id_] = result
    return result


def get_details(selected_movie_name):
    # Loading both the dataframes for fast reading
    pickle_file_path = r'Files/movies_dict.pkl'
    with open(pickle_file_path, 'rb') as pickle_file:
        loaded_dict = pickle.load(pickle_file)

    movies = pd.DataFrame.from_dict(loaded_dict)

    pickle_file_path = r'Files/movies2_dict.pkl'
    with open(pickle_file_path, 'rb') as pickle_file:
        loaded_dict_2 = pickle.load(pickle_file)

    movies2 = pd.DataFrame.from_dict(loaded_dict_2)

    # Extracting series of data to be displayed
    a = pd.DataFrame(movies2[movies2['title'] == selected_movie_name])
    b = pd.DataFrame(movies[movies['title'] == selected_movie_name])

    # Extracting necessary details
    budget = a.iloc[0, 2]
    overview = a.iloc[0, 3]
    release_date = a.iloc[:, 5].iloc[0]
    revenue = a.iloc[:, 6].iloc[0]
    runtime = a.iloc[:, 7].iloc[0]
    available_lang = ast.literal_eval(a.iloc[0, 8])
    vote_rating = a.iloc[:, 10].iloc[0]
    vote_count = a.iloc[:, 11].iloc[0]
    movie_id = a.iloc[:, 0].iloc[0]
    cast = b.iloc[:, 9].iloc[0]
    director = b.iloc[:, 10].iloc[0]
    genres = b.iloc[:, 3].iloc[0]
    this_poster = fetch_posters(movie_id)
    cast_per = b.iloc[:, 5].iloc[0]
    a = ast.literal_eval(cast_per)
    cast_id = []
    for i in a:
        cast_id.append(i['id'])
    lang = []
    for i in available_lang:
        lang.append(i['name'])

    # Adding to a list for easy export
    info = [this_poster, budget, genres, overview, release_date, revenue, runtime, available_lang, vote_rating,
            vote_count, movie_id, cast, director, lang, cast_id]

    return info
