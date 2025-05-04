import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

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

def combine_cluster_labels(user_ids, labels):
    # Convert labels to a DataFrame
    labels_df = pd.DataFrame(labels)
    # Merge user_ids DataFrame with labels DataFrame based on index
    cluster_df = pd.merge(user_ids, labels_df, left_index=True, right_index=True)
    # Rename columns to 'user' and 'cluster'
    cluster_df.columns = ['user', 'cluster']
    return cluster_df





# Model training
def train(model_name, test_user_id, params):
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

        # Get user's rated courses
        user_ratings = ratings_df[ratings_df['user'] == test_user_id]
        # Calculate weighted genre scores based on ratings
        test_user_profile_df = pd.DataFrame()
        test_user_profile_df['user'] = [test_user_id]
        # Calculate genre scores by multiplying ratings with course genres
        for genre in course_genres_df.columns[2:]:
            genre_scores = []
            for _, row in user_ratings.iterrows():
                course_id = row['item']
                rating = row['rating']
                course_genre = course_genres_df[course_genres_df['COURSE_ID'] == course_id][genre].values[0]
                genre_scores.append(rating * course_genre)
            test_user_profile_df[genre] = [sum(genre_scores)]

        # Save user profiles
        test_user_profile_df.to_csv('data/test_user_profile.csv', index=False)
    elif model_name == models[2]: # Clustering
        k = 13
        if 'cluster_no' in params:
            k = params['cluster_no']
         # Load and process user ratings data
        ratings_df = pd.read_csv('data/ratings.csv')
        # Load course genres data
        course_genres_df = pd.read_csv('data/course_genres.csv')
        # Load user profiles
        user_profiles_df = pd.read_csv('data/user_profiles.csv')

        # Get test user profile
        user_ratings = ratings_df[ratings_df['user'] == test_user_id]
        # Calculate weighted genre scores based on ratings
        test_user_profile_df = pd.DataFrame()
        test_user_profile_df['user'] = [test_user_id]
        # Calculate genre scores by multiplying ratings with course genres
        for genre in course_genres_df.columns[2:]:
            genre_scores = []
            for _, row in user_ratings.iterrows():
                course_id = row['item']
                rating = row['rating']
                course_genre = course_genres_df[course_genres_df['COURSE_ID'] == course_id][genre].values[0]
                genre_scores.append(rating * course_genre)
            test_user_profile_df[genre] = [sum(genre_scores)]

        # Append the test user profile to the user profiles DataFrame
        user_profiles_df = pd.concat([user_profiles_df, test_user_profile_df], ignore_index=True)

        feature_names = list(user_profiles_df.columns[1:])

        # Use StandardScaler to make each feature with mean 0, standard deviation 1
        # Instantiating a StandardScaler object
        scaler = StandardScaler()

        # Standardizing the selected features (feature_names) in the user_profile_df DataFrame
        user_profiles_df[feature_names] = scaler.fit_transform(user_profiles_df[feature_names])

        features = user_profiles_df.loc[:, user_profiles_df.columns != 'user']
        user_ids = user_profiles_df.loc[:, user_profiles_df.columns == 'user']

        rs = 42
        k_means_model = KMeans(n_clusters=k, random_state=rs).fit(features)
        cluster_labels = k_means_model.labels_

        combined_df = combine_cluster_labels(user_ids, cluster_labels)
        # Save the cluster labels to a CSV file
        combined_df.to_csv('data/user_clusters.csv', index=False)
    elif model_name == models[3]: # Clustering with PCA
        k = 13
        if 'cluster_no' in params:
            k = params['cluster_no']
         # Load and process user ratings data
        ratings_df = pd.read_csv('data/ratings.csv')
        # Load course genres data
        course_genres_df = pd.read_csv('data/course_genres.csv')
        # Load user profiles
        user_profiles_df = pd.read_csv('data/user_profiles.csv')

        # Get test user profile
        user_ratings = ratings_df[ratings_df['user'] == test_user_id]
        # Calculate weighted genre scores based on ratings
        test_user_profile_df = pd.DataFrame()
        test_user_profile_df['user'] = [test_user_id]
        # Calculate genre scores by multiplying ratings with course genres
        for genre in course_genres_df.columns[2:]:
            genre_scores = []
            for _, row in user_ratings.iterrows():
                course_id = row['item']
                rating = row['rating']
                course_genre = course_genres_df[course_genres_df['COURSE_ID'] == course_id][genre].values[0]
                genre_scores.append(rating * course_genre)
            test_user_profile_df[genre] = [sum(genre_scores)]

        # Append the test user profile to the user profiles DataFrame
        user_profiles_df = pd.concat([user_profiles_df, test_user_profile_df], ignore_index=True)

        feature_names = list(user_profiles_df.columns[1:])

        # Use StandardScaler to make each feature with mean 0, standard deviation 1
        # Instantiating a StandardScaler object
        scaler = StandardScaler()

        # Standardizing the selected features (feature_names) in the user_profile_df DataFrame
        user_profiles_df[feature_names] = scaler.fit_transform(user_profiles_df[feature_names])

        features = user_profiles_df.loc[:, user_profiles_df.columns != 'user']
        user_ids = user_profiles_df.loc[:, user_profiles_df.columns == 'user']

        # Apply PCA
        pca = PCA(n_components=9)
        components = pca.fit_transform(features)

        rs = 42
        k_means_model = KMeans(n_clusters=k, random_state=rs).fit(components)
        cluster_labels = k_means_model.labels_

        combined_df = combine_cluster_labels(user_ids, cluster_labels)
        # Save the cluster labels to a CSV file
        combined_df.to_csv('data/user_clusters.csv', index=False)

# Prediction
def predict(model_name, test_user_id, params):
    sim_threshold = 0.6
    score_threshold = 2
    if "sim_threshold" in params:
        sim_threshold = params["sim_threshold"] / 100.0
    if 'score_threshold' in params:
        score_threshold = params['score_threshold']
    idx_id_dict, id_idx_dict = get_doc_dicts()
    sim_matrix = load_course_sims().to_numpy()
    users = []
    courses = []
    scores = []
    res_dict = {}

    # Course Similarity model
    if model_name == models[0]:
        ratings_df = load_ratings()
        user_ratings = ratings_df[ratings_df['user'] == test_user_id]
        enrolled_course_ids = user_ratings['item'].to_list()
        res = course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix)
        for key, score in res.items():
            if score >= sim_threshold:
                users.append(test_user_id)
                courses.append(key)
                scores.append(score)
    # User Profile model prediction
    elif model_name == models[1]:
        # Load necessary data
        ratings_df = load_ratings()
        course_genres_df = pd.read_csv('data/course_genres.csv')
        profile_df = pd.read_csv('data/test_user_profile.csv')

        # Get user profile
        user_profile = profile_df[profile_df['user'] == test_user_id]
        if not user_profile.empty:
            # Get user vector (excluding user ID column)
            user_vector = user_profile.iloc[0, 1:].values

            # Get enrolled courses for the user
            user_ratings = ratings_df[ratings_df['user'] == test_user_id]
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
                print(f'score: {score}, score: {score_threshold}')
                if score >= score_threshold:
                    users.append(test_user_id)
                    courses.append(unknown_course_ids[i])
                    scores.append(score)
        else:
            print(f'User {test_user_id} has no profile data.')
    elif model_name == models[2]: # Clustering
        enrollments_threshold = params.get('enrollments_threshold', 10)
        ratings_df = load_ratings()
        test_users_df = ratings_df[['user', 'item']]
        # Load user clusters
        user_clusters_df = pd.read_csv('data/user_clusters.csv')

        test_users_labelled = pd.merge(test_users_df, user_clusters_df, left_on='user', right_on='user')

        # Extracting the 'item' and 'cluster' columns from the test_users_labelled DataFrame
        courses_cluster = test_users_labelled[['item', 'cluster']]

        # Adding a new column 'count' with a value of 1 for each row in the courses_cluster DataFrame
        courses_cluster['count'] = [1] * len(courses_cluster)

        # Grouping the DataFrame by 'cluster' and 'item', aggregating the 'count' column with the sum function,
        # and resetting the index to make the result more readable
        courses_cluster_grouped = courses_cluster.groupby(['cluster','item']).agg(enrollments=('count','sum')).reset_index()
        # Recommend unseen courses based on the popular courses in test users cluster
        recommendations = {}

        ## - For each user, first finds its cluster label
        for user_id in test_users_labelled['user'].unique():
            user_subset = test_users_labelled[test_users_labelled['user'] == user_id]
            ## - First get all courses belonging to the same cluster and figure out what are the popular ones (such as course enrollments beyond a threshold like 100)
            all_courses = courses_cluster_grouped[courses_cluster_grouped['cluster'] == user_subset['cluster'].iloc[0]]
            popular_courses = all_courses[all_courses['enrollments'] >= enrollments_threshold]['item'].values.tolist()
            popular_courses = set(popular_courses)
            ## - Get the user's current enrolled courses
            users_enrolled_courses = test_users_labelled[test_users_labelled['user'] == user_id]['item'].values.tolist()
            users_enrolled_courses = set(users_enrolled_courses)
            ## - Check if there are any courses on the popular course list which are new/unseen to the user.
            unseen_courses = popular_courses.difference(users_enrolled_courses)
            ## If yes, make those unseen and popular courses as recommendation results for the user
            recommendations[f'{user_id}'] = list(unseen_courses)


        if f'{test_user_id}' in recommendations:
            print(f"Recommendations for user {test_user_id}: {recommendations[f'{test_user_id}']}")
        else:
            print(f"No recommendations found for user {test_user_id}.")
        if f'{test_user_id}' in recommendations and len(recommendations[f'{test_user_id}']) > 0:
            # Add the recommendations to the result lists
            for course in recommendations[f'{test_user_id}']:
                users.append(test_user_id)
                courses.append(course)
                scores.append(None)
    elif model_name == models[3]: # Clustering with PCA
        enrollments_threshold = params.get('enrollments_threshold', 10)
        ratings_df = load_ratings()
        test_users_df = ratings_df[['user', 'item']]
        # Load user clusters
        user_clusters_df = pd.read_csv('data/user_clusters.csv')

        test_users_labelled = pd.merge(test_users_df, user_clusters_df, left_on='user', right_on='user')

        # Extracting the 'item' and 'cluster' columns from the test_users_labelled DataFrame
        courses_cluster = test_users_labelled[['item', 'cluster']]

        # Adding a new column 'count' with a value of 1 for each row in the courses_cluster DataFrame
        courses_cluster['count'] = [1] * len(courses_cluster)

        # Grouping the DataFrame by 'cluster' and 'item', aggregating the 'count' column with the sum function,
        # and resetting the index to make the result more readable
        courses_cluster_grouped = courses_cluster.groupby(['cluster','item']).agg(enrollments=('count','sum')).reset_index()
        # Recommend unseen courses based on the popular courses in test users cluster
        recommendations = {}

        ## - For each user, first finds its cluster label
        for user_id in test_users_labelled['user'].unique():
            user_subset = test_users_labelled[test_users_labelled['user'] == user_id]
            ## - First get all courses belonging to the same cluster and figure out what are the popular ones (such as course enrollments beyond a threshold like 100)
            all_courses = courses_cluster_grouped[courses_cluster_grouped['cluster'] == user_subset['cluster'].iloc[0]]
            popular_courses = all_courses[all_courses['enrollments'] >= enrollments_threshold]['item'].values.tolist()
            popular_courses = set(popular_courses)
            ## - Get the user's current enrolled courses
            users_enrolled_courses = test_users_labelled[test_users_labelled['user'] == user_id]['item'].values.tolist()
            users_enrolled_courses = set(users_enrolled_courses)
            ## - Check if there are any courses on the popular course list which are new/unseen to the user.
            unseen_courses = popular_courses.difference(users_enrolled_courses)
            ## If yes, make those unseen and popular courses as recommendation results for the user
            recommendations[f'{user_id}'] = list(unseen_courses)


        if f'{test_user_id}' in recommendations:
            print(f"Recommendations W PCA for user {test_user_id}: {recommendations[f'{test_user_id}']}")
        else:
            print(f"No recommendations found for user {test_user_id}.")
        if f'{test_user_id}' in recommendations and len(recommendations[f'{test_user_id}']) > 0:
            # Add the recommendations to the result lists
            for course in recommendations[f'{test_user_id}']:
                users.append(test_user_id)
                courses.append(course)
                scores.append(None)


    res_dict['USER'] = users[:params['top_courses']] if 'top_courses' in params else users
    res_dict['COURSE_ID'] = courses[:params['top_courses']] if 'top_courses' in params else courses
    res_dict['SCORE'] = scores[:params['top_courses']] if 'top_courses' in params else scores
    res_df = pd.DataFrame(res_dict, columns=['USER', 'COURSE_ID', 'SCORE'])
    return res_df
