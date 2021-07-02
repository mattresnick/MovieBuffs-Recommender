import gensim as g
import requests
import numpy as np
import scipy
from sklearn.decomposition import PCA

def formatForModel(f_str):
    '''
    Parameters:
        - f_str (string): String of raw plot sentence(s).

    Returns:
        - list of strings, no punctuation, no uppercase.

    '''
    
    f_str = f_str.translate(str.maketrans('', '', r"""!"#$%&'()*+,-.:;=?@[\]^_`{|}~""")).lower()
    f_str = f_str.replace(u'\u201C', '')
    f_str = f_str.replace(u'\u201D', '')
    f_str = f_str.replace(u'\u2018', '')
    f_str = f_str.replace(u'\u2019', '')
    f_str = f_str.replace('\'', '')
    
    
    f_str = list(filter(None, f_str.split())) 
    
    return f_str


def getPlotCentroid(plot, model):
    '''
    Parameters:
        - plot (list): Formatted list of plot words or keywords.
        - model (dict): word:embedding dictionary.

    Returns:
        - centroid in word embedding space.
    '''
    
    
    vecs = []
    for word in plot:
        try: vecs.append(model[word])
        except: pass
    return np.mean(vecs,axis=0)
        

def keywordFiltering(plot):
    '''
    Parameters:
        - plot (list): Formatted list of plot words.

    Returns:
        - List of plot keywords.
    '''
    # By usage or fitted model?
    return plot



def PCAProjection(centroids,n_comp):
    '''
    Parameters:
        - centroids (array-like): matrix of plot centroids.
        - n_comp (integer): number of dimensions to remain.

    Returns:
        - Centroids projected to a lower-dimensional space for clustering.
    '''
    centroids = np.array(centroids)
    
    pca = PCA(n_comp)
    projected_centroids = pca.fit_transform(centroids)
    
    return projected_centroids











model= g.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)


with open('apikey.txt','r') as f: key=f.read()

titles = ['blade+runner','blade+runner+2049','the+godfather']

s_centroids = []
for t in titles:
    r = requests.get('http://www.omdbapi.com/?apikey={}&t={}'.format(key,t))
    plot = formatForModel(r.json()['Plot'])
    plot = keywordFiltering(plot)
    s_centroids.append(getPlotCentroid(plot,model))



print ('Short plot:')
print ('Distance between Blade Runner (1982) and the sequel:',
       scipy.spatial.distance.cosine(s_centroids[0], s_centroids[1]))
print ('Distance between Blade Runner and The Godfather:',
       scipy.spatial.distance.cosine(s_centroids[0], s_centroids[2]))
print()



s_proj = PCAProjection(s_centroids,n_comp=3)
print ('Projected short plot:')
print ('Distance between Blade Runner (1982) and the sequel:',
       scipy.spatial.distance.cosine(s_proj[0], s_proj[1]))
print ('Distance between Blade Runner and The Godfather:',
       scipy.spatial.distance.cosine(s_proj[0], s_proj[2]))
print()





l_centroids = []
for t in titles:
    r = requests.get('http://www.omdbapi.com/?apikey={}&t={}&plot=full'.format(key,t))
    plot = formatForModel(r.json()['Plot'])
    plot = keywordFiltering(plot)
    l_centroids.append(getPlotCentroid(plot,model))


print ('Long plot:')
print ('Distance between Blade Runner (1982) and the sequel:',
       scipy.spatial.distance.cosine(l_centroids[0], l_centroids[1]))
print ('Distance between Blade Runner and The Godfather:',
       scipy.spatial.distance.cosine(l_centroids[0], l_centroids[2]))
print()



l_proj = PCAProjection(l_centroids,n_comp=3)
print ('Projected long plot:')
print ('Distance between Blade Runner (1982) and the sequel:',
       scipy.spatial.distance.cosine(l_proj[0], l_proj[1]))
print ('Distance between Blade Runner and The Godfather:',
       scipy.spatial.distance.cosine(l_proj[0], l_proj[2]))
print()
