import utils
from data import *
from data import PROCESSED_DATA_PATH as DP
from sklearn.metrics.pairwise import cosine_similarity
import textwrap

args = utils.get_args()
print(f'Query: {args.desc}\n')

df = load_csv(DP)
# Using movie title, description of movie, and genre information for TF-IDF. 
df['Details'] = df.apply(lambda row: f'{row['movie_name']}: {row['description']} ({row['genre']})', axis=1)

# Lemmatize details of movies and attach lemmatized query
lem_details = list(df['Details'].apply(lemmatize_sentence))
lem_query = lemmatize_sentence(args.desc)
sentences = lem_details + [lem_query]

# TF-IDF on the lemmatized data
matrix = tfidf(sentences)

# Compute cosine similarity between query and dataset
similarities = cosine_similarity(matrix[:-1], matrix[-1:]).flatten()
sorted_idx = similarities.argsort()[::-1]

top_idx = sorted_idx[:args.count]
top_similarities = similarities[top_idx]

wrapper = textwrap.TextWrapper(initial_indent='     ', subsequent_indent='     ', width=112)
print('Recommended Movies')
print('------------------')
for i, (sim, idx) in enumerate(zip(top_similarities, top_idx)): 
    top_row = df.iloc[idx]
    movie = f'{top_row['movie_name']} ({top_row['year']})'
    print(f'Rank {i+1: <4} {movie: <50} Genre: {top_row['genre']: <30} Score: {sim:.4f}')
    print(wrapper.fill(top_row['description']))