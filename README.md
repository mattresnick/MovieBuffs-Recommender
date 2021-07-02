This is the isolated code of a larger project. For the privacy of my colleagues and the security of their own work, this contains only the pieces I completed myself. 

# Recommender System

![alt text](diagrams/true_representation.png?raw=true)

Logical pipeline:

![alt text](diagrams/main_diagram.png?raw=true)

Detailed autoencoder view (somewhat outdated, as the encoder is now comprised of a bidirectional LSTM, 
and neither the encoder or decoder has a dense layer on the end):

![alt text](diagrams/inset_diagram.png?raw=true)

The idea for the recommender system is 1) objectivity, 2) variable granularity, and 3) film-centric.
Specifically, a user should get a recommendation by first providing a film by name and specifying what components
of the recommendation are important to them (e.g plot, director, cast, etc.). This means that two users can and 
should get identical recommendations depending on the input they provide. The goal is that the system will 
recommend movies not based on subjective preference or mood but rather by similarity in as many objective
characteristics as possible. 

The primary component (and currently only data we have a metric for) is plot-based recommendations. The steps of 
the approach are detailed below and in the diagram at the top.



## Deep Dimensionality Reduction
 - map the words of the plots into an embedding space via word2vec
 - using these embedding vectors, train a sequential autoencoder with LSTM units to produce a dimensionality
   reducing encoder of arbitrary size
 - project the encoded representations down further or use as-is for nearest neighbor recommendation as with method 1.

The problem with this method is of course significant complexity (training time, compute time needed for
inference, architecture design). But it comes with the benefit of only needing to be trained once (or, at least 
not very often) as titles get added to the database. That is, training must be offline, but inference is online
and thus adding new titles or a search component is possible.

Folding in additional info such as director or cast would require the creation of an objective distance metric between 
people (and I would use a graph-based method to do this), but then this metric can be appended to the embedding 
vectors and weighted to reflect the user's preference. So they could indicate they want the same director, similar 
directors, don't care, or dissimilar directors and the weight in the embeddings would be adjusted accordingly. Other 
information, like visual components, are a little bit more complicated but could be added in as well.




