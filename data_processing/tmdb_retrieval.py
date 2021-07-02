import requests
import pandas as pd
import numpy as np


# Get list of keywords for a given movie via TMDB.
def getKeywords(tmdb_id, tmdb_key):
    r_str = 'https://api.themoviedb.org/3/movie/{}/keywords?api_key={}'.format(tmdb_id,tmdb_key)
    
    r = requests.get(r_str)
    keywords = [wd['name'] for wd in r.json()['keywords']]
    
    return keywords



# Get cast and crew info for a given movie via TMDB
def getCastandCrew(tmdb_id,tmdb_key):
    '''
    Cast is recorded as a simple list of names.
    Crew is recorded as name, job pairs.
    '''
    
    r_str = 'https://api.themoviedb.org/3/movie/{}/credits?api_key={}&language=en-US'.format(tmdb_id,tmdb_key)
    
    r = requests.get(r_str)
    
    try:
        cast_dicts = r.json()['cast']
        cast_list = [d['name'] for d in cast_dicts]
    except:
        cast_list = None
    
    try:
        crew_raw = r.json()['crew']
        crew_dict = {d['job']:d['name'] for d in crew_raw}
    except:
        crew_dict = None
    
    return cast_list, crew_dict




# Convert an IMDB ID to a TMDB ID.
def IMDBtoTMDB(imdb_id, tmdb_key):
    r_str = 'https://api.themoviedb.org/3/find/{}?api_key={}&language=en-US&external_source=imdb_id'.format(imdb_id,tmdb_key)
    
    r = requests.get(r_str)
    
    try:
        tmdb_id = r.json()['movie_results'][0]['id']
    except:
        return None
    
    return tmdb_id