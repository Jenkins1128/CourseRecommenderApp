import streamlit as st
import pandas as pd
import time
import backend as backend

from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid import GridUpdateMode, DataReturnMode

# Basic webpage setup
st.set_page_config(
   page_title="Course Recommender System",
   layout="wide",
   initial_sidebar_state="expanded",
)


# ------- Functions ------
# Load datasets
@st.cache_data
def load_ratings():
    return backend.load_ratings()


@st.cache_data
def load_course_sims():
    return backend.load_course_sims()


@st.cache_data
def load_courses():
    return backend.load_courses()


@st.cache_data
def load_bow():
    return backend.load_bow()


# Initialize the app by first loading datasets
def init__recommender_app():

    with st.spinner('Loading datasets...'):
        ratings_df = load_ratings()
        sim_df = load_course_sims()
        course_df = load_courses()
        course_bow_df = load_bow()

    # Select courses
    st.success('Datasets loaded successfully...')

    st.markdown("""---""")
    st.subheader("Select courses that you have audited or completed: ")

    # Build an interactive table for `course_df`
    gb = GridOptionsBuilder.from_dataframe(course_df)
    gb.configure_default_column(enablePivot=True, enableValue=True, enableRowGroup=True)
    gb.configure_selection(selection_mode="multiple", use_checkbox=True)
    gb.configure_side_bar()
    grid_options = gb.build()

    # Create a grid response
    response = AgGrid(
        course_df,
        gridOptions=grid_options,
        enable_enterprise_modules=True,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        fit_columns_on_grid_load=False,
    )


    results = pd.DataFrame(response["selected_rows"], columns=['COURSE_ID', 'TITLE', 'DESCRIPTION'])
    results = results[['COURSE_ID', 'TITLE']]
    st.subheader("Your courses: ")
    st.table(results)
    return results


def train(model_name, test_user_id, params):

    if model_name == backend.models[0]:
        # Start training course similarity model
        with st.spinner('Training Course Similarity Model...'):
            time.sleep(0.5)
            backend.train(model_name, test_user_id, params)
        st.success('Course Similarity Model trained successfully!')
    elif model_name == backend.models[1]:
        with st.spinner('Training User Profile Model...'):
            time.sleep(0.5)
            backend.train(model_name, test_user_id, params)
        st.success('User Profile Model trained successfully!')
    elif model_name == backend.models[2]:
        with st.spinner('Training Clustering Model...'):
            time.sleep(0.5)
            backend.train(model_name, test_user_id, params)
        st.success('Clustering Model trained successfully!')
    elif model_name == backend.models[3]:
        with st.spinner('Training Clustering Model with PCA...'):
            time.sleep(0.5)
            backend.train(model_name, test_user_id, params)
        st.success('Clustering Model with PCA trained successfully!')
    elif model_name == backend.models[4]:
        with st.spinner('Training KNN Model...'):
            time.sleep(0.5)
            backend.train(model_name, test_user_id, params)
        st.success('KNN Model trained successfully!')
    elif model_name == backend.models[5]:
        with st.spinner('Training NMF Model...'):
            time.sleep(0.5)
            backend.train(model_name, test_user_id, params)
        st.success('NMF Model trained successfully!')
    elif model_name == backend.models[6]: # Neural Network
        with st.spinner('Training Neural Network Model...'):
            time.sleep(0.5)
            backend.train(model_name, test_user_id, params)
        st.success('Neural Network Model trained successfully!')

    else:
        st.warning('Training logic for this model is not implemented yet.')


def predict(model_name, test_user_id, params):
    res = None
    # Start making predictions based on model name, test user ids, and parameters
    with st.spinner('Generating course recommendations: '):
        time.sleep(0.5)
        res = backend.predict(model_name, test_user_id, params)
    st.success('Recommendations generated!')
    return res


# ------ UI ------
# Sidebar
st.sidebar.title('Personalized Learning Recommender')
# Initialize the app
selected_courses_df = init__recommender_app()

# Model selection selectbox
st.sidebar.subheader('1. Select recommendation models')
model_selection = st.sidebar.selectbox(
    "Select model:",
    backend.models
)

# Hyper-parameters for each model
params = {}
st.sidebar.subheader('2. Tune Hyper-parameters: ')
# Course similarity model
if model_selection == backend.models[0]:
    # Add a slide bar for selecting top courses
    top_courses = st.sidebar.slider('Top courses',
                                    min_value=0, max_value=100,
                                    value=10, step=1)
    # Add a slide bar for choosing similarity threshold
    course_sim_threshold = st.sidebar.slider('Course Similarity Threshold %',
                                             min_value=0, max_value=100,
                                             value=50, step=10)
    params['top_courses'] = top_courses
    params['sim_threshold'] = course_sim_threshold
# User profile model
elif model_selection == backend.models[1]:
    # Add a slide bar for selecting top courses
    top_courses = st.sidebar.slider('Top courses',
                                    min_value=0, max_value=100,
                                    value=10, step=1)
    profile_score_threshold = st.sidebar.slider('User Profile Score Threshold',
                                              min_value=0, max_value=15,
                                              value=2, step=1)
    params['top_courses'] = top_courses
    params['profile_score_threshold'] = profile_score_threshold
# Clustering model
elif model_selection == backend.models[2]:
    cluster_no = st.sidebar.slider('Number of Clusters',
                                   min_value=0, max_value=50,
                                   value=10, step=1)
    enrollments_threshold = st.sidebar.slider('User Enrollments Threshold',
                                              min_value=0, max_value=100,
                                              value=10, step=2)
    params['cluster_no'] = cluster_no
    params['enrollments_threshold'] = enrollments_threshold
# Clustering model with PCA
elif model_selection == backend.models[3]:
    cluster_no = st.sidebar.slider('Number of Clusters',
                                   min_value=0, max_value=50,
                                   value=10, step=1)
    enrollments_threshold = st.sidebar.slider('User Enrollments Threshold',
                                              min_value=0, max_value=100,
                                              value=10, step=2)
    params['cluster_no'] = cluster_no
    params['enrollments_threshold'] = enrollments_threshold
# KNN model
elif model_selection == backend.models[4]:
    # Add a slide bar for selecting top courses
    top_courses = st.sidebar.slider('Top courses',
                                    min_value=0, max_value=100,
                                    value=10, step=1)
    knn_score_threshold = st.sidebar.slider('KNN Rating Mode Threshold',
                                            min_value=0, max_value=3,
                                            value=2, step=1)
    params['top_courses'] = top_courses
    params['knn_score_threshold'] = knn_score_threshold
# NMF Model
elif model_selection == backend.models[5]:
    # Add a slide bar for selecting top courses
    top_courses = st.sidebar.slider('Top courses',
                                    min_value=0, max_value=100,
                                    value=10, step=1)
    nmf_score_threshold = st.sidebar.slider('NMF Rating Mode Threshold',
                                            min_value=0, max_value=3,
                                            value=2, step=1)
    params['top_courses'] = top_courses
    params['nmf_score_threshold'] = nmf_score_threshold
elif model_selection == backend.models[6]:
    # Add a slide bar for selecting top courses
    top_courses = st.sidebar.slider('Top courses',
                                    min_value=0, max_value=100,
                                    value=10, step=1)
    nn_score_threshold = st.sidebar.slider('Neural Network Score Threshold',
                                            min_value=0.0, max_value=1.0,
                                            value=0.01, step=.01)
    params['top_courses'] = top_courses
    params['nn_score_threshold'] = nn_score_threshold
else:
    pass


# Training
st.sidebar.subheader('3. Training: ')
training_button = st.sidebar.button("Train Model")
training_text = st.sidebar.text('')
# Start training process
if training_button:
    new_test_user_id = backend.add_new_ratings(selected_courses_df['COURSE_ID'].values)
    train(model_selection, new_test_user_id, params)


# Prediction
st.sidebar.subheader('4. Prediction')
# Start prediction process
pred_button = st.sidebar.button("Recommend New Courses")
if pred_button and selected_courses_df.shape[0] > 0:
    # Create a new id for current user session
    test_user_id = backend.load_ratings()['user'].max()
    res_df = predict(model_selection, test_user_id, params)
    res_df = res_df[['COURSE_ID', 'SCORE']]
    course_df = load_courses()
    res_df = pd.merge(res_df, course_df, on=["COURSE_ID"]).drop(['COURSE_ID', 'DESCRIPTION', 'SCORE'], axis=1) if res_df['SCORE'].isnull().all() else pd.merge(res_df, course_df, on=["COURSE_ID"]).drop(['COURSE_ID', 'DESCRIPTION'], axis=1)
    st.table(res_df)
