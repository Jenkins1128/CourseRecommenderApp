import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

models = ("Course Similarity",
          "User Profile",
          "Clustering",
          "Clustering with PCA",
          "KNN",
          "NMF",
          "Neural Network",
          "Regression with Embedding Features",
          "Classification with Embedding Features")


def load_ratings():
    return pd.read_csv("data/ratings.csv")


def load_course_sims():
    return pd.read_csv("data/sim.csv")


def load_courses():
    df = pd.read_csv("data/course_processed.csv")
    df['TITLE'] = df['TITLE'].str.title()
    return df


def load_bow():
    return pd.read_csv("data/courses_bows.csv")


def add_new_ratings(new_courses):
    res_dict = {}
    if len(new_courses) > 0:
        # Create a new user id, max id + 1
        ratings_df = load_ratings()
        new_id = ratings_df['user'].max() + 1
        users = [new_id] * len(new_courses)
        ratings = [3.0] * len(new_courses)
        res_dict['user'] = users
        res_dict['item'] = new_courses
        res_dict['rating'] = ratings
        new_df = pd.DataFrame(res_dict)
        updated_ratings = pd.concat([ratings_df, new_df])
        updated_ratings.to_csv("data/ratings.csv", index=False)
        return new_id


# Create course id to index and index to id mappings
def get_doc_dicts():
    bow_df = load_bow()
    grouped_df = bow_df.groupby(['doc_index', 'doc_id']).max().reset_index(drop=False)
    idx_id_dict = grouped_df[['doc_id']].to_dict()['doc_id']
    id_idx_dict = {v: k for k, v in idx_id_dict.items()}
    del grouped_df
    return idx_id_dict, id_idx_dict


def course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix):
    all_courses = set(idx_id_dict.values())
    unselected_course_ids = all_courses.difference(enrolled_course_ids)
    # Create a dictionary to store your recommendation results
    res = {}
    # First find all enrolled courses for user
    for enrolled_course in enrolled_course_ids:
        for unselect_course in unselected_course_ids:
            if enrolled_course in id_idx_dict and unselect_course in id_idx_dict:
                idx1 = id_idx_dict[enrolled_course]
                idx2 = id_idx_dict[unselect_course]
                sim = sim_matrix[idx1][idx2]
                if unselect_course not in res:
                    res[unselect_course] = sim
                else:
                    if sim >= res[unselect_course]:
                        res[unselect_course] = sim
    res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}
    return res





# Model training
def train(model_name, params):
    # TODO: Add model training code here
    print('params:', params)

    if model_name == models[0]: #Course Similarity
            # Load the Bag of Words (BoW) dataset
            bow_df = load_bow()

            # Create a similarity matrix
            bow_matrix_np = bow_df.pivot(index='doc_index', columns='token', values='bow').fillna(0).to_numpy()
            sim_matrix = cosine_similarity(bow_matrix_np)

            # Apply similarity threshold
            sim_threshold = params.get('sim_threshold', 50) / 100.0  # Default to 50% if not provided
            sim_matrix[sim_matrix < sim_threshold] = 0

            # Save the filtered similarity matrix
            filtered_sim_df = pd.DataFrame(sim_matrix)
            filtered_sim_df.to_csv("data/sim.csv", index=False)
    elif model_name == models[1]: # User Profile
         # Load and process user ratings data
        ratings_df = pd.read_csv('data/ratings.csv')
        # Load course genres data
        course_genres_df = pd.read_csv('data/course_genres.csv')
        # Get unique users
        unique_users = ratings_df['user'].unique()
        # Create empty dataframe for user profiles
        profiles = []
        # For each user
        for user_id in unique_users:
            # Get user's rated courses
            user_ratings = ratings_df[ratings_df['user'] == user_id]
            # Get course genres for rated courses
            user_course_genres = course_genres_df[course_genres_df['COURSE_ID'].isin(user_ratings['item'])]
            # Calculate weighted genre scores based on ratings
            profile = pd.DataFrame()
            profile['user'] = [user_id]
            # Calculate genre scores by multiplying ratings with course genres
            for genre in course_genres_df.columns[2:]:
                genre_scores = []
                for idx, row in user_ratings.iterrows():
                    course_id = row['item']
                    rating = row['rating']
                    course_genre = course_genres_df[course_genres_df['COURSE_ID'] == course_id][genre].values[0]
                    genre_scores.append(rating * course_genre)
                profile[genre] = [sum(genre_scores)]
            profiles.append(profile)

        # Combine all user profiles
        profile_df = pd.concat(profiles, ignore_index=True)
        # Save user profiles
        profile_df.to_csv('data/profile_genres.csv', index=False)


# Prediction
def predict(model_name, user_ids, params):
    sim_threshold = 0.6
    if "sim_threshold" in params:
        sim_threshold = params["sim_threshold"] / 100.0
    idx_id_dict, id_idx_dict = get_doc_dicts()
    sim_matrix = load_course_sims().to_numpy()
    users = []
    courses = []
    scores = []
    res_dict = {}

    for user_id in user_ids:
        # Course Similarity model
        if model_name == models[0]:
            ratings_df = load_ratings()
            user_ratings = ratings_df[ratings_df['user'] == user_id]
            enrolled_course_ids = user_ratings['item'].to_list()
            res = course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix)
            for key, score in res.items():
                if score >= sim_threshold:
                    users.append(user_id)
                    courses.append(key)
                    scores.append(score)
        # User Profile model prediction
        elif model_name == models[1]:
            # Load necessary data
            ratings_df = load_ratings()
            course_genres_df = pd.read_csv('data/course_genres.csv')
            profile_df = pd.read_csv('data/profile_genres.csv')

            # Get user profile
            user_profile = profile_df[profile_df['user'] == user_id]
            if not user_profile.empty:
                # Get user vector (excluding user ID column)
                user_vector = user_profile.iloc[0, 1:].values

                # Get enrolled courses for the user
                user_ratings = ratings_df[ratings_df['user'] == user_id]
                enrolled_courses = user_ratings['item'].to_list()

                # Get all courses and find unknown courses
                all_courses = set(course_genres_df['COURSE_ID'].values)
                unknown_courses = all_courses.difference(enrolled_courses)

                # Filter course genres for unknown courses
                unknown_course_genres = course_genres_df[course_genres_df['COURSE_ID'].isin(unknown_courses)]
                unknown_course_ids = unknown_course_genres['COURSE_ID'].values

                # Calculate recommendation scores using dot product
                recommendation_scores = np.dot(unknown_course_genres.iloc[:, 2:].values, user_vector)

                # Add recommendations above threshold
                for i in range(len(unknown_course_ids)):
                    score = recommendation_scores[i]
                    print(f'score: {score}, sim_threshold: {sim_threshold}')
                    if score >= sim_threshold:
                        users.append(user_id)
                        courses.append(unknown_course_ids[i])
                        scores.append(score)
            else:
                print(f'User {user_id} has no profile data.')

    res_dict['USER'] = users[:params['top_courses']] if 'top_courses' in params else users
    res_dict['COURSE_ID'] = courses[:params['top_courses']] if 'top_courses' in params else courses
    res_dict['SCORE'] = scores[:params['top_courses']] if 'top_courses' in params else scores
    res_df = pd.DataFrame(res_dict, columns=['USER', 'COURSE_ID', 'SCORE'])
    return res_df
