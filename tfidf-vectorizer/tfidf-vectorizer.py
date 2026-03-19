import numpy as np
from collections import Counter
import math

def tfidf_vectorizer(documents):
    """
    Build TF-IDF matrix from a list of text documents.
    Returns tuple of (tfidf_matrix, vocabulary).
    """
    # Write code here
    vocabulary = set() # (total t,) : (1, total t)
    unique_terms_count_per_doc = []
    num_docs = len(documents)
    count_per_term_per_doc = []
    
    for doc in documents:
        unique_terms = set(doc.split())
        vocabulary.update(unique_terms)
        unique_terms_count_per_doc.append(len(doc.split()))

    vocabulary = sorted(vocabulary)
    print(vocabulary)

    for doc in documents:
        doc_word_list = doc.lower().split()
        count_per_term_per_doc.append(
            [doc_word_list.count(term) for term in vocabulary]
        )
        
    count_per_term_per_doc = np.array(count_per_term_per_doc)
    print(count_per_term_per_doc)    
    
    unique_terms_count_per_doc = np.array(unique_terms_count_per_doc)
    print("total terms in d:\n", unique_terms_count_per_doc.reshape(-1, 1))

    mask = count_per_term_per_doc >= 1
    print("valid mask:", mask)
    
    df = np.sum(mask, axis=0)
    print("df:", df)
    
    idf = np.log(num_docs / df)
    print("idf:", idf)
    
    tf = count_per_term_per_doc / unique_terms_count_per_doc.reshape(-1, 1)
    print("tf:\n", tf)
    
    tf_idf = tf * idf.reshape(1, -1)
    print("tf-idf:\n", tf_idf)
    
    return tf_idf, vocabulary        