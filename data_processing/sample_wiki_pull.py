import omdb_retrieval as o
import pandas as pd
import numpy as np

def write_out(title, ID, plot, save_name):
    df=pd.DataFrame(np.array([title, ID, plot]))
    df = df.T
    df.to_csv(save_name, mode='a', header=False, index=False)




with open('apikey.txt','r') as f: key=f.read()
with open('tmdb_key.txt','r') as f: tkey=f.read()





o.findWikiMovies('title_list.csv')


save_file = 'short_plot_data.csv'
df = pd.DataFrame(columns=['Title','ID','Plot'])
df.to_csv(save_file, index=False)


title_list = pd.read_csv('title_list.csv').to_numpy().ravel()
N = len(title_list)


fail_list = []
for i,title in enumerate(title_list):
    print('\r{:30}'.format('Main list progress: ' + str(i+1) + '/' + str(N)),end='')
    
    req = o.OMDbRequest(key=key, title=title)
    if req['Response']=='False':
        fail_list.append(title)
    elif req['Type']=='movie':
        write_out(title, req['imdbID'], req['Plot'], save_name=save_file)



print ()


N = len(fail_list)
ID_list = pd.read_csv('short_plot_data.csv')['ID'].to_numpy().ravel()

for i,title in enumerate(fail_list):
    print('\r{:30}'.format('Fail list progress: ' + str(i+1) + '/' + str(N)),end='')
    
    reqs = o.OMDbSearchRequest(key=key, ID_list=ID_list, title=title)
    if reqs is not None:
        for req in reqs:
            write_out(req['Title'], req['imdbID'], req['Plot'], save_name=save_file)


print ()
