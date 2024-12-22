import torch
import pandas as pd

import random
import os
from sklearn.preprocessing import MultiLabelBinarizer

class ml_25m():
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.users_data, self.items_data, self.ratings_data= self.load_ml_25m(os.path.join(self.args.data_dir, self.args.data_name))
        args.num_users = self.users_data['user_id'].max()
        args.num_items = self.items_data['movie_id'].max()
        self.all_item_ids = self.items_data['movie_id'].tolist()

        args.dim_user_discrete_data = []
        
        args.dim_item_discrete_data = len(self.items_data['genres'][0])
        
    def load_ml_25m(self, data_path):
        rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
        dtype_spec = {'user_id': int, 'movie_id': int, 'rating': float, 'timestamp': int}
        ratings = pd.read_csv(data_path + '/ratings.csv', sep=',', header=None, names=rnames, engine='python', dtype=dtype_spec, skiprows=1)

        users = ratings.drop_duplicates(subset=['user_id'])
        
        mnames = ['movie_id', 'title', 'genres']
        movies = pd.read_csv(data_path + '/movies.csv', sep=',', header=None, names=mnames, encoding="unicode_escape", engine='python', skiprows=1)

        movies['genres'] = movies['genres'].str.split('|')
        mlb = MultiLabelBinarizer()
        genre_labels = mlb.fit_transform(movies['genres'])
        movies['genres'] = genre_labels.tolist()

        # Normalize timestamps
        ratings['timestamp'] = ratings['timestamp'] - ratings['timestamp'].min()
        ratings.sort_values(by=['user_id', 'timestamp'])
        return users, movies, ratings
    
    def get_user_embedding(self, user_id):
        df = self.users_data[self.users_data['user_id'] == user_id]
        if df.empty:
            raise ValueError(f"No data found for user_id: {user_id}")
        #id = df['user_id'].values[0]
        #textual_data = self.user_textual_data.loc[self.user_textual_data['user_id'] == id, 'textual_data'].values[0]
        #user_discrete_data = df.iloc[0, 1:].values.tolist()
        return None

    def get_item_data(self, item_id):
        df = self.items_data[self.items_data['movie_id'] == item_id]
        if df.empty:
            raise ValueError(f"No data found for movie_id: {item_id}")
        id = df['movie_id'].values[0]
        genres = df['genres'].values[0]
        return genres

    def get_user_item_data(self, user_id, ratings_data=None):
        if ratings_data is None:
            ratings_data = self.ratings_data

        user_ratings = ratings_data[ratings_data['user_id'] == user_id]

        item_ids = user_ratings['movie_id'].tolist()

        item_discrete_data = []
        for movie_id in item_ids:
            genre= self.get_item_data(movie_id)
            item_discrete_data.append(genre)

        ratings = user_ratings['rating'].tolist()
        timestamps = user_ratings['timestamp'].tolist()

        return item_ids, item_discrete_data, ratings, timestamps

    def get_neg_user_item_data(self, user_id, train = 'train'):
        user_ratings = self.ratings_data[self.ratings_data['user_id'] == user_id]

        item_ids = user_ratings['movie_id'].tolist()

        neg = list(set(self.all_item_ids) - set(item_ids))

        if train == 'train':
            num_neg_sample = len(item_ids)
        else:
            num_neg_sample = self.args.num_neg_sample
            
        if len(neg) < num_neg_sample:
            neg_item_ids = random.choices(neg, k=num_neg_sample)
        else:
            neg_item_ids = random.sample(neg, num_neg_sample)

        neg_item_discrete_data = []
        for movie_id in neg_item_ids:
            genre= self.get_item_data(movie_id)
            neg_item_discrete_data.append(genre)

        return neg_item_ids, neg_item_discrete_data

class ml_20m():
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.users_data, self.items_data, self.ratings_data = self.load_ml_20m(os.path.join(self.args.data_dir, self.args.data_name))
        args.num_users = self.users_data['user_id'].max()
        args.num_items = self.items_data['movie_id'].max()
        self.all_item_ids = self.items_data['movie_id'].tolist()

        args.dim_user_discrete_data = []
        
        args.dim_item_discrete_data = len(self.items_data['genres'][0])
        
    def load_ml_20m(self, data_path):
        rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
        dtype_spec = {'user_id': int, 'movie_id': int, 'rating': float, 'timestamp': int}
        ratings = pd.read_csv(data_path + '/ratings.csv', sep=',', header=None, names=rnames, engine='python', dtype=dtype_spec, skiprows=1)

        users = ratings.drop_duplicates(subset=['user_id'])
        
        mnames = ['movie_id', 'title', 'genres']
        movies = pd.read_csv(data_path + '/movies.csv', sep=',', header=None, names=mnames, encoding="unicode_escape", engine='python', skiprows=1)
            
        movies['genres'] = movies['genres'].str.split('|')
        mlb = MultiLabelBinarizer()
        genre_labels = mlb.fit_transform(movies['genres'])
        movies['genres'] = genre_labels.tolist()

        # Normalize timestamps
        ratings['timestamp'] = ratings['timestamp'] - ratings['timestamp'].min()
        ratings.sort_values(by=['user_id', 'timestamp'])
        return users, movies, ratings
    
    def get_user_embedding(self, user_id):
        df = self.users_data[self.users_data['user_id'] == user_id]
        if df.empty:
            raise ValueError(f"No data found for user_id: {user_id}")
        #id = df['user_id'].values[0]
        #textual_data = self.user_textual_data.loc[self.user_textual_data['user_id'] == id, 'textual_data'].values[0]
        #user_discrete_data = df.iloc[0, 1:].values.tolist()
        return None

    def get_item_data(self, item_id):
        df = self.items_data[self.items_data['movie_id'] == item_id]
        if df.empty:
            raise ValueError(f"No data found for movie_id: {item_id}")
        id = df['movie_id'].values[0]
        genres = df['genres'].values[0]
        return genres

    def get_user_item_data(self, user_id, ratings_data=None):
        if ratings_data is None:
            ratings_data = self.ratings_data

        user_ratings = ratings_data[ratings_data['user_id'] == user_id]

        item_ids = user_ratings['movie_id'].tolist()

        item_discrete_data = []
        for movie_id in item_ids:
            genre= self.get_item_data(movie_id)
            item_discrete_data.append(genre)

        ratings = user_ratings['rating'].tolist()
        timestamps = user_ratings['timestamp'].tolist()

        return item_ids, item_discrete_data, ratings, timestamps

    def get_neg_user_item_data(self, user_id, train = 'train'):
        user_ratings = self.ratings_data[self.ratings_data['user_id'] == user_id]

        item_ids = user_ratings['movie_id'].tolist()

        neg = list(set(self.all_item_ids) - set(item_ids))

        if train == 'train':
            num_neg_sample = len(item_ids)
        else:
            num_neg_sample = self.args.num_neg_sample
            
        if len(neg) < num_neg_sample:
            neg_item_ids = random.choices(neg, k=num_neg_sample)
        else:
            neg_item_ids = random.sample(neg, num_neg_sample)

        neg_item_discrete_data = []
        for movie_id in neg_item_ids:
            genre= self.get_item_data(movie_id)
            neg_item_discrete_data.append(genre)

        return neg_item_ids, neg_item_discrete_data
        

class ml_1m():
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.users_data, self.items_data, self.ratings_data= self.load_ml_1m(os.path.join(self.args.data_dir, self.args.data_name))
        args.num_users = self.users_data['user_id'].max()
        args.num_items = self.items_data['movie_id'].max()
        self.all_item_ids = self.items_data['movie_id'].tolist()

        args.dim_user_discrete_data = [self.users_data['gender'].max()+1, self.users_data['age'].max()+1, self.users_data['occupation'].max()+1, self.users_data['zip'].max()+1]
        
        args.dim_item_discrete_data = len(self.items_data['genres'][0])
        
    def load_ml_1m(self, data_path):
        unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
        users = pd.read_csv(data_path + '/users.dat', sep='::', header=None, names=unames, engine='python')

        rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
        ratings = pd.read_csv(data_path + '/ratings.dat', sep='::', header=None, names=rnames, engine='python')
        
        mnames = ['movie_id', 'title', 'genres']
        movies = pd.read_csv(data_path + '/movies.dat', sep='::', header=None, names=mnames, encoding="unicode_escape", engine='python')

        users['zip'] = pd.factorize(users['zip'])[0]
        users['gender'] = pd.factorize(users['gender'])[0]
        users['age'] = pd.factorize(users['age'])[0]

        movies['genres'] = movies['genres'].str.split('|')
        mlb = MultiLabelBinarizer()
        genre_labels = mlb.fit_transform(movies['genres'])
        movies['genres'] = genre_labels.tolist()

        # Normalize timestamps
        ratings['timestamp'] = ratings['timestamp'] - ratings['timestamp'].min()
        ratings.sort_values(by=['user_id', 'timestamp'])
        return users, movies, ratings
    
    def get_user_embedding(self, user_id):
        df = self.users_data[self.users_data['user_id'] == user_id]

        if df.empty:
            raise ValueError(f"No data found for user_id: {user_id}")
        id = df['user_id'].values[0]
        user_discrete_data = df.iloc[0, 1:].values.tolist()

        return user_discrete_data

    def get_item_data(self, item_id):
        df = self.items_data[self.items_data['movie_id'] == item_id]
        if df.empty:
            raise ValueError(f"No data found for movie_id: {item_id}")
        id = df['movie_id'].values[0]
        genres = df['genres'].values[0]
        return genres

    def get_user_item_data(self, user_id, ratings_data=None):
        if ratings_data is None:
            ratings_data = self.ratings_data

        user_ratings = ratings_data[ratings_data['user_id'] == user_id]

        item_ids = user_ratings['movie_id'].tolist()

        item_discrete_data = []
        for movie_id in item_ids:
            genre= self.get_item_data(movie_id)
            item_discrete_data.append(genre)

        ratings = user_ratings['rating'].tolist()
        timestamps = user_ratings['timestamp'].tolist()

        return item_ids, item_discrete_data, ratings, timestamps

    def get_neg_user_item_data(self, user_id, train = 'train'):
        user_ratings = self.ratings_data[self.ratings_data['user_id'] == user_id]

        item_ids = user_ratings['movie_id'].tolist()

        neg = list(set(self.all_item_ids) - set(item_ids))

        if train == 'train':
            num_neg_sample = len(item_ids)
        else:
            num_neg_sample = self.args.num_neg_sample
            
        if len(neg) < num_neg_sample:
            neg_item_ids = random.choices(neg, k=num_neg_sample)
        else:
            neg_item_ids = random.sample(neg, num_neg_sample)

        neg_item_discrete_data = []
        for movie_id in neg_item_ids:
            genre= self.get_item_data(movie_id)
            neg_item_discrete_data.append(genre)

        return neg_item_ids, neg_item_discrete_data
    
class ml_100k():
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.users_data, self.items_data, self.ratings_data= self.load_ml_100k(os.path.join(self.args.data_dir, self.args.data_name))
        args.num_users = self.users_data['user_id'].max()
        args.num_items = self.items_data['movie_id'].max()
        self.all_item_ids = self.items_data['movie_id'].tolist()

        args.dim_user_discrete_data = [self.users_data['gender'].max()+1, self.users_data['age'].max()+1, self.users_data['occupation'].max()+1, self.users_data['zip'].max()+1]
        
        args.dim_item_discrete_data = len(self.items_data['genres'][0])
        
    def load_ml_100k(self, data_path):

        user_column_names = ['user_id', 'age', 'gender', 'occupation', 'zip']
        users = pd.read_csv(data_path + '/u.user', sep='|', header=None, names=user_column_names, encoding='latin1')

        rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
        ratings = pd.read_csv(data_path + '/u.data', sep='\t', header=None, names=rnames, encoding='latin1')

        movie_column_names = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL',
                'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
                'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        items = pd.read_csv(data_path + '/u.item', sep='|', header=None, names=movie_column_names, encoding='latin1')
        genre_columns = movie_column_names[5:] 
        items['genres'] = items[genre_columns].values.tolist()
        movies = items[['movie_id', 'title', 'genres']]


        users['zip'] = pd.factorize(users['zip'])[0]
        users['gender'] = pd.factorize(users['gender'])[0]
        users['age'] = pd.factorize(users['age'])[0]
        users['occupation'] = pd.factorize(users['occupation'])[0]

        # Normalize timestamps
        ratings['timestamp'] = ratings['timestamp'] - ratings['timestamp'].min()
        ratings.sort_values(by=['user_id', 'timestamp'])
        return users, movies, ratings
    
    def get_user_embedding(self, user_id):
        df = self.users_data[self.users_data['user_id'] == user_id]
        if df.empty:
            raise ValueError(f"No data found for user_id: {user_id}")
        #id = df['user_id'].values[0]
        #textual_data = self.user_textual_data.loc[self.user_textual_data['user_id'] == id, 'textual_data'].values[0]
        user_discrete_data = df.iloc[0, 1:].values.tolist()
        return user_discrete_data

    def get_item_data(self, item_id):
        df = self.items_data[self.items_data['movie_id'] == item_id]
        if df.empty:
            raise ValueError(f"No data found for movie_id: {item_id}")
        id = df['movie_id'].values[0]
        genres = df['genres'].values[0]
        return genres

    def get_user_item_data(self, user_id, ratings_data=None):
        if ratings_data is None:
            ratings_data = self.ratings_data

        user_ratings = ratings_data[ratings_data['user_id'] == user_id]

        item_ids = user_ratings['movie_id'].tolist()

        item_discrete_data = []
        for movie_id in item_ids:
            genre= self.get_item_data(movie_id)
            item_discrete_data.append(genre)

        ratings = user_ratings['rating'].tolist()
        timestamps = user_ratings['timestamp'].tolist()

        return item_ids, item_discrete_data, ratings, timestamps

    def get_neg_user_item_data(self, user_id, train = 'train'):
        user_ratings = self.ratings_data[self.ratings_data['user_id'] == user_id]

        item_ids = user_ratings['movie_id'].tolist()

        neg = list(set(self.all_item_ids) - set(item_ids))

        if train == 'train':
            num_neg_sample = len(item_ids)
        else:
            num_neg_sample = self.args.num_neg_sample
            
        if len(neg) < num_neg_sample:
            neg_item_ids = random.choices(neg, k=num_neg_sample)
        else:
            neg_item_ids = random.sample(neg, num_neg_sample)

        neg_item_discrete_data = []
        for movie_id in neg_item_ids:
            genre= self.get_item_data(movie_id)
            neg_item_discrete_data.append(genre)

        return neg_item_ids, neg_item_discrete_data