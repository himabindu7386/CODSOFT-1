import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
data = {
    'Movie A': [5, 4, 0, 1],
    'Movie B': [4, 0, 0, 1],
    'Movie C': [1, 1, 0, 5],
    'Movie D': [0, 0, 5, 4]
}

df = pd.DataFrame(data, index=['User1','User2','User3','User4'])
df
similarity = cosine_similarity(df)
sim_df = pd.DataFrame(similarity, index=df.index, columns=df.index)
sim_df
user = 'User1'
similar_users = sim_df[user].sort_values(ascending=False)[1:]

recommendations = df.loc[similar_users.index].mean().sort_values(ascending=False)
recommendations
