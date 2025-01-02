# %%
import numpy as np;
import pandas as pd;

# %%
books = pd.read_csv('Books.csv')
users = pd.read_csv('Users.csv')
ratings = pd.read_csv('Ratings.csv')

# %%
books

# %%
ratings

# %%
users

# %%
print(books.shape)
print(ratings.shape)
print(users.shape)

# %%
books.isnull().sum()

# %%
users.isnull().sum()

# %%
users.duplicated().sum()

# %% [markdown]
# ## Popularity Based Recommender System

# %%
ratings_with_name = ratings.merge(books,on='ISBN')

# %%
ratings_with_name[~ratings_with_name['Book-Rating'].apply(lambda x: isinstance(x, (int, float)))].shape

# %%
ratings_with_name['Book-Rating'] = pd.to_numeric(ratings_with_name['Book-Rating'], errors='coerce')

# %%
ratings_with_name

# %%
num_rating_df = ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_rating_df.rename(columns={'Book-Rating' : 'num_ratings'},inplace=True)


# %%
ratings_with_name.groupby('Book-Title')['Book-Rating'].count().reset_index()

# %%
num_rating_df

# %%
avg_rating_df = ratings_with_name.groupby('Book-Title')['Book-Rating'].mean().reset_index()
avg_rating_df.rename(columns={'Book-Rating' : 'avg_ratings'},inplace=True)

# %%
avg_rating_df

# %%
popularity_df = num_rating_df.merge(avg_rating_df,on='Book-Title')
popularity_df

# %%
popularity_df=popularity_df[popularity_df['num_ratings']>=250].sort_values('avg_ratings',ascending=False).head(50)

# %%
popularity_df = popularity_df.merge(books,on='Book-Title').drop_duplicates('Book-Title')[['Book-Title','Book-Author','Year-Of-Publication','num_ratings','avg_ratings','Image-URL-M']]

# %%
popularity_df

# %% [markdown]
# ## Collaboration 

# %%
ratings_with_name

# %%
import pickle

# %%
pickle.dump(popularity_df,open('popular.pkl','wb'))

# %%
x = ratings_with_name.groupby('User-ID')['Book-Rating'].count() > 200
print(x)
padhe_likhe_users = x[x].index

# %%
padhe_likhe_users.shape

# %%
ratings_with_name['User-ID'].isin(padhe_likhe_users)

# %%
ratings_with_name[ratings_with_name['User-ID'].isin(padhe_likhe_users)]

# %%
filtered_rating = ratings_with_name[ratings_with_name['User-ID'].isin(padhe_likhe_users)]

# %%
filtered_rating

# %%
y = filtered_rating.groupby('Book-Title')['Book-Rating'].count() >= 50
famous_books = y[y].index

# %%
final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]

# %%
pt = final_ratings.pivot_table(index='Book-Title',columns='User-ID',values='Book-Rating')

# %%
pt.fillna(0,inplace=True)

# %%
pt

# %%
from sklearn.metrics.pairwise import cosine_similarity

# %%
similarity_score = cosine_similarity(pt)
print(similarity_score[0])

# %%
def recommend_function(book_name):
    #index fetch
    index = np.where(pt.index==book_name)[0][0]
    similar_items = sorted(list(enumerate(similarity_score[index])),key=lambda x:x[1],reverse=True)[1:6]
    data = []
    for i in similar_items:

        item = []
        result = (books['Book-Title']) == pt.index[i[0]]
    
        temp_df = books[result]
        item.append(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.append(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.append(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
        # print(item)
        data.append(item)
    

    return data;




# %%
recommend_function('1984')

# %%
pickle.dump(pt,open('pt.pkl','wb'))
pickle.dump(books,open('books.pkl','wb'))
pickle.dump(similarity_score,open('similarity_score.pkl','wb'))

# %%



