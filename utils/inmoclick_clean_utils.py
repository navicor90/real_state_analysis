import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def fix_clean_total_area(df_original):
    df = df_original.copy()
    df = df[df['totalArea']!='disable']
    # Removing string chars in totalArea
    df['totalArea_fixed'] = df.totalArea.str.replace('mts2','').str.replace('m2','').str.replace('m','')
    df.totalArea_fixed = df.totalArea_fixed.str.strip()
    def cast_to_comma(value):
        if len(value)>= 3 and value[-3]==".":
            return value.replace(".",',')
        else:
            return value
    df.totalArea_fixed = df.totalArea_fixed.apply(lambda x: cast_to_comma(x))
    df.totalArea_fixed = df.totalArea_fixed.str.replace('.', '')
    df.totalArea_fixed = df.totalArea_fixed.str.replace(',', '.')
    df.totalArea_fixed = pd.to_numeric(df.totalArea_fixed, errors='coerce')
    return df


def fix_clean_floor_area(df_original):
    df = df_original.copy()
    df = df[df['floorArea']!='disable']
    df['floorArea_fixed'] = df.floorArea.str.replace('mts2','').str.replace('m2','').str.replace('m','')
    df.floorArea_fixed = df.floorArea_fixed.str.strip()
    def cast_to_comma(value):
        if isinstance(value,str) and len(value)>= 3 and value[-3]==".":
            return value.replace(".",',')
        else:
            return value
    df.floorArea_fixed = df.floorArea_fixed.apply(lambda x: cast_to_comma(x))
    df.floorArea_fixed = df.floorArea_fixed.str.replace('.', '')
    df.floorArea_fixed = df.floorArea_fixed.str.replace(',', '.')
    df.floorArea_fixed = pd.to_numeric(df.floorArea_fixed, errors='coerce')
    return df


def bucketizer(df, bucket_limits, field, new_field):
    prev_limit = None
    ranges = []
    for i,limit in enumerate(bucket_limits):
        if i==0:
            df.loc[df[field] <= limit, new_field] = 0
            ranges = ["0.."+str(limit)]
        elif i==len(bucket_limits)-1:
            df.loc[np.logical_and(df[field] > prev_limit,df[field] <= limit), new_field] = prev_limit
            ranges.append(str(prev_limit)+".."+str(limit))
            
            df.loc[df[field] > limit, new_field] = limit
            ranges.append(str(limit)+"..")
        else:
            df.loc[np.logical_and(df[field] > prev_limit,df[field] <= limit), new_field] = prev_limit
            ranges.append(str(prev_limit)+".."+str(limit))
        prev_limit = limit
    return ranges


def str_serie_to_vector(pd_serie, stop_words=[]):
    """ Convert a list of str to vectors using tfidf alghorithm
            pd_serie: a pandas serie with strings to be converted to vector
            stop_words: list of words to be avoided in the conversion
    """
    neighborhoods = list(pd_serie.fillna("").str.lower().str.replace('\W', ' '))
    documents = []
    for d in neighborhoods:
        for sw in stop_words:
            d = d.replace(sw,' ')
        documents.append(d)
    tfidf = TfidfVectorizer().fit(documents)
    return tfidf.transform(neighborhoods)


def vectors_sparse_similarity(matrix1,matrix2):
    """ This function receive two list of vectors(represented as matrix), and compare
        all vectors in matrix1 with all vectors in matrix2, evaluating its similarity.
            matrix1: csr_matrix - list of vectors, each row will be used as a vector
            matrix2: csr_matrix - list of vectors with will be compared matrix1, each row will be
            used as a vector.
    """
    def matrix_similarity(r):
        return cosine_similarity(r, matrix2).flatten()
    return np.apply_along_axis(matrix_similarity, 1, matrix1.todense())

def similar_prices(prop1,prop2,span:int):
    some_price_is_undefined = "Consultar" in prop1.price or "Consultar" in prop2.price
    return some_price_is_undefined or prop2.amount-span < prop1.amount <prop2.amount+span

def similar_total_area(prop1,prop2,span:int):
    return prop2.totalArea_fixed-span < prop1.totalArea_fixed < prop2.totalArea_fixed+span

def similar_floor_area(prop1,prop2,span:int):
    return prop2.floorArea_fixed-span < prop1.floorArea_fixed < prop2.floorArea_fixed+span


def are_similar_properties(prop1,prop2):
    same_prop_type = prop1.property_type==prop2.property_type
    if not same_prop_type:
        return False
    are_similar = False
    if prop1.property_type=="PropertyType.APARTMENT":
        are_similar = ( prop1.bedrooms == prop2.bedrooms and
                        prop1.district == prop2.district and
                        prop1.bathrooms == prop2.bathrooms and 
                        similar_floor_area(prop1, prop2, 10) and
                        similar_total_area(prop1, prop2, 20) and
                        similar_prices(prop1, prop2, 40000))
        
    elif prop1.property_type=="PropertyType.HOUSE":
        are_similar = ( prop1.bedrooms == prop2.bedrooms and
                        prop1.district == prop2.district and
                        prop1.bathrooms == prop2.bathrooms and 
                        similar_floor_area(prop1, prop2, 10) and
                        similar_total_area(prop1, prop2, 20) and
                        similar_prices(prop1, prop2, 40000))
        
    elif prop1.property_type=="PropertyType.LAND":
        are_similar = ( prop1.district == prop2.district and
                        prop1.has_water == prop2.has_water and 
                        prop1.has_electricity == prop2.has_electricity and 
                        prop1.has_gas == prop2.has_gas and 
                        similar_total_area(prop1, prop2, 20) and
                        similar_prices(prop1, prop2, 40000))
        
    else:
        raise Exception(f"Unsupported property type:"+prop1.property_type)
        
    return are_similar
            
        