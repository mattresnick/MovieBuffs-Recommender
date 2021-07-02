import numpy as np
import pandas as pd
import gensim as g


model=g.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)


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


# Convert formatted plot to word2vec embeddings.
def getPlotVecs(plot, model):
    
    vecs = []
    for word in plot:
        try: vecs.append(model[word])
        except: pass
    return vecs


# Function for converting plots to embeddings from a DataFrame column.
def toVecRep(row, cent=False):
    global model
    plot = row['Plot']
    plot_list = formatForModel(plot)
    
    vecs = getPlotVecs(plot_list, model)
    
    if cent:
        return np.mean(vecs,axis=0)
    
    return vecs
    


# Function for converting plots to embeddings from a DataFrame column, 
# with special formatting/delimeters.
def toVecRepStr(row, cent=False):
    global model
    plot = row['Plot']
    plot_list = formatForModel(plot)
    
    vecs = ''
    for word in plot_list:
        try: 
            vec_word = model[word]
            
            add_word = ''
            for val in vec_word:
                add_word = add_word+str(val)+','
            
            
            vecs = vecs+str(add_word[:-1])+'|'
        
        # Skip over words not found in dictionary.
        except: pass 
    
    return vecs[:-1]






# Clean up original data by removing N/A synopses.
to_be_cleaned = pd.read_csv('short_plot_data.csv')
to_be_cleaned = to_be_cleaned.dropna()
to_be_cleaned.to_csv('short_plot_data_clean.csv', index=False)

'''
# Create single composite data file with all clean data and converted plots.
df = pd.read_csv('short_plot_data_clean.csv')
df['plot_vec_rep'] = df.apply(lambda row: toVecRep(row), axis=1)
df['plot_centroid'] = df.apply(lambda row: toVecRep(row, True), axis=1)
df.to_csv('all_clean_data.csv', index=False)
'''

# Create special data file with only converted plots (no words, no centroids).
clean_data = pd.read_csv('short_plot_data_clean.csv')
new_df = pd.DataFrame(columns=['plot_vec_rep'])
new_df['plot_vec_rep'] = clean_data.apply(lambda row: toVecRepStr(row), axis=1)
new_df.to_csv('plot_vecs.csv', index=False)



