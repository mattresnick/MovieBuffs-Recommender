import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup



def findAllMovies(start_num, stop_num, key, savefile='id_list.csv'):
    
    '''
    Finds list of valid IMDB movie IDs by checking all possible numbers.
    
    Parameters
        - start_num (integer): Beginning number to start ID search from.
        - stop_num (integer): Number to end ID search on. Max = 7548383.
        - key (string): OMDb API key.
        - savefile (string): CSV file to save ID list to.
        
    Returns
        - None
    '''
    
    for i in range(start_num, stop_num+1):
        print('\r{:20}'.format(i),end='')
        
        m_id = 'tt'+str(i).zfill(7)
        
        r = requests.get('http://www.omdbapi.com/?apikey={}&i={}'.format(key,m_id))
        
        '''
        langstr = r.json()['Language']
        langlist = langstr.split(', ')
        '''
        
        try:
            is_movie = r.json()['Type']=='movie'
        except:
            is_movie=False
        
        if is_movie:
            df=pd.DataFrame(np.array([m_id]))
            df.to_csv(savefile, mode='a', header=False, index=False)



def findWikiMovies(save_name=''):
    
    '''
    Finds list of movie titles by scraping wikipedia list pages.
    
    Parameters
        - save_name (string): CSV file to save ID list to. If left empty, the
          function will instead return the list. 
    
    Returns
        - List of titles (or None, if a save_name is passed).
    '''
    
    suffixes = ['numbers','A','B','C','D','E','F',
                'G','H','I','J-K','L','M','N-O',
                'P','Q-R','S','T','U-W','X-Z']
    
    all_titles=[]
    for s in suffixes:
        r = requests.get('https://en.wikipedia.org/wiki/List_of_films:_{}'.format(s))
        
        
        soup = BeautifulSoup(r.content, 'html.parser')
        
        lines = soup.select("i")
        titles = [l.text for l in lines]
        
        if len(save_name)>0:
            df=pd.DataFrame(np.array(titles))
            df.to_csv('title_list.csv', mode='a', header=False, index=False)
        else:
            all_titles.extend(titles)
    
    if len(all_titles)>0:
        return all_titles





def OMDbRequest(key, ID=None, title=None, full_plot=False):
    '''
    Single OMDb JSON request. Provide only one of ID or title. 
    
    Some notes for using the json dictionary returned: 
        - It may be necessary to check that r.json()['Type']=='movie' before 
        storing the response, since it is possible for movie/TV series to have 
        the same name.
        
        - If r.json()['Response']=='False' on a title request, we should save
        the title that caused it. The OMDbSearchRequest function may find it later.
    
    Parameters
        - key (string): OMDb API key.
        - ID (string): Exact IMDb movie ID.
        - title (string): Movie title.
        - full_plot (boolean): If true, return verbose plot synopsis.
    
    Returns
        - JSON dictionary of a single title (or, dictionary of failed response).
    
    '''
    if ID is not None:
        r_str = 'http://www.omdbapi.com/?apikey={}&i={}'.format(key,ID)
    else:
        if not isinstance(title, str):
            title = str(title)
        rtitle = title.replace(' ','+')
        r_str = 'http://www.omdbapi.com/?apikey={}&t={}'.format(key,rtitle)
    
    if full_plot: r_str = r_str+'&plot=full'
    
    r = requests.get(r_str)
    
    return r.json()




def OMDbSearchRequest(key, ID_list, title=None, full_plot=False):
    
    '''
    Tries to obtain results for titles that didn't get exact matches. It is 
    possible and even likely that more than one result will be found with this
    method. This is okay since only movies will be returned.
    
    Parameters
        - key (string): OMDb API key.
        - ID_list (list or array): All IMDb IDs for which good responses were found.
          This list needs to be checked against to avoid adding duplicates.
        - title (string): Movie title.
        - full_plot (boolean): If true, return verbose plot synopsis.
    
    Returns
        - List of JSON dictionaries for each film found (or None, if no such
          movies were found).
    '''
    
    rtitle = title.replace(' ','+')
    r_str = 'http://www.omdbapi.com/?apikey={}&s={}&type=movie'.format(key,rtitle)
    
    if full_plot: r_str = r_str+'&plot=full'
    
    r = requests.get(r_str)
    r_json = r.json()
    
    if r_json['Response']=='False': return
    
    dict_list = []
    for item in r_json['Search']:
        ID = item['imdbID'] 
        if ID not in ID_list:
            new_response = OMDbRequest(key=key, ID=ID, full_plot=full_plot)
            if new_response['Response']=='True':
                dict_list.append(new_response)
    
    if len(dict_list)>0: return dict_list
    
    return




def OMDbSearchReturnBest(key, title=None):
    '''
    Only returns the title from the search which is most popular (by 
    number of votes on IMDb).
    
    Parameters
        - key (string): OMDb API key.
        - title (string): Movie title.
    
    Returns
        - IMDb ID of the best result.
    '''
    
    rtitle = title.replace(' ','+')
    r_str = 'http://www.omdbapi.com/?apikey={}&s={}&type=movie'.format(key,rtitle)
    
    
    r = requests.get(r_str)
    r_json = r.json()
    
    if r_json['Response']=='False': return
    
    sort_list = [[i['imdbID'], i['imdbVotes']] for i in r_json['Search']]
    sort_list.sort(reverse=True, key=lambda x: x[1])
    
    if len(sort_list)>0: 
        return sort_list[0][0]
    
    return