import os
import glob
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


ID_COL = 'movie_id'
DATA_COLUMNS = [ID_COL,'year','movie_name','genre','description']
MIN_DESC_LEN = 20
PROCESSED_DATA_PATH = './processed_data.csv'
NLTK_DATA_PATH = os.path.join(os.path.curdir, './nltk_data')
nltk.data.path.append(NLTK_DATA_PATH)

# Downloads NLTK data by default into NLTK_DATA_PATH
NLTK_DATA_PACKS = ['punkt_tab', 'wordnet']
for dp in NLTK_DATA_PACKS: 
    nltk.download(dp, NLTK_DATA_PATH, quiet=True)

lemmatizer = WordNetLemmatizer()


def clean_and_save_data(original_data_path, new_csv, num_rows=500, seed=2025): 
    """Cleans and saves randomly trimmed rows of data. 

    Parameters
    ----------
    original_data_path : str
        Location of original data folder path. 
    new_data_path : int
        Name of csv file to save data. 
    num_rows : int
        Number of rows to save. 
    seed : int
        Random selection seed. 
    """
    
    if not os.path.isdir(original_data_path): 
        raise Exception(f'{original_data_path} is not a directory.')

    files = glob.glob(os.path.join(original_data_path, '*.csv'))
    
    # Concatenate each dataframe from csv files
    df = pd.concat(
            # Drop unused columns and rows with n/a values
            [pd.read_csv(f, usecols=DATA_COLUMNS).dropna() for f in files],
            ignore_index=True
        )
    
    # Organize data
    df = df.drop_duplicates(subset=[ID_COL])

    # Drop rows with short descriptions
    df = df[df['description'].apply(len) >= MIN_DESC_LEN]

    # Drop not-so-pretty rows and columns
    df = df[df['description'].apply(lambda s: 'See full summary' not in s)]
    df = df[df['year'].apply(lambda s: str(s).isnumeric())]
    df = df.drop(columns=ID_COL)

    # Randomly sample num_rows rows
    df = df.sample(num_rows, random_state=seed)
    
    df = df.reset_index(drop=True)

    # Save
    df.to_csv(new_csv, index=False)


def load_csv(csv_path): 
    """Load the data in the csv file."""
    df = pd.read_csv(csv_path)
    return df


def lemmatize_sentence(s): 
    """Lemmatize a single sentence."""

    # Split string into word tokens
    s = word_tokenize(s)

    for i in range(len(s)): 
        # Convert words with alpabets into lower case and lemmatize
        if s[i].isalpha(): 
            s[i] = lemmatizer.lemmatize(s[i].lower())
    
    # Combine list of words into single string
    return ' '.join(s)


def tfidf(sentences): 
    """Use TF-IDF on list of strings. 
    
    Parameters
    ----------
    sentences : List of lemmatized sentences to TF-IDF.

    Returns
    -------
    ndarray of vectorized data. 
    """
    tfidf = TfidfVectorizer()
    result = tfidf.fit_transform(sentences)

    return result


# Run to get processed_data.csv
if __name__=="__main__": 
    clean_and_save_data('./data', PROCESSED_DATA_PATH)
