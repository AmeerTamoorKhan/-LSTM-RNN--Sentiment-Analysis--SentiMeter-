import streamlit as st
import tensorflow as tf
import pickle
import numpy as np


def load_model():
    # Load the Tokenizer
    with open('utilities/tokenizer.pickle', 'rb') as file:
        tokinizer = pickle.load(file)

    # Load the Model
    model = tf.keras.models.load_model('utilities/sentimentModel.h5')

    return tokinizer, model


def sentiment_analyzer(comment, tokinizer, model):
    # Test sentences
    text = np.array([comment])
    predict = tokinizer.texts_to_sequences(np.array(text))
    predict = tf.keras.preprocessing.sequence.pad_sequences(predict, maxlen=50)

    # Predict the Sentiment
    # 0 = Negative
    # 1 = Positive

    prediction = np.around(model.predict(np.array(predict)))
    print(prediction)
    return prediction


st.set_page_config(page_title='Sentimeter', page_icon='😃')

cols_title = st.beta_columns((1, 2, 1, 1, 1))
cols_title[0].image('images/title.png', width=100, use_column_width=True)
cols_title[1].title('Sentimeter')


options = ['About', 'Sentimeter']
radio = st.sidebar.radio('Select', options)
st.sidebar.markdown(
    '''
    <h3>Created By: Ameer Tamoor Khan</h3>
    <h4>Github : <a href="https://github.com/AmeerTamoorKhan" target="_blank">Click Here </a></h4> 
    <h4>Email: drop-in@atkhan.info</h4>
    ''', unsafe_allow_html=True
)


def about():
    st.header('Working Demonstration:')
    st.video('images/sentimeter.mp4')
    st.header('How It Works:')
    st.markdown('''
    <p>A simple project but helped me a lot to dive further into Machine Learning.</p>
    <p>Sentimeter is designed to analyze the semantic of a sentence to conclude whether it has a positive sentiment or 
    negative. The model includes LSTM (Long Short Term Memory), which is an RNN (Recurrent Neural Network). The model 
    is trained with 1.5 million tweets and thanks to Kaggle, which is a go-to platform when it comes to big-data and ML.
    And also thanks to Kaggle for providing free TPU service because even GPU is a time-consuming option with such 
    big-data.</p>️
    <p>The concept is simple if the machine senses the positive sentiment in the sentence it will thumbs-up, and for 
    negative sentiment thumbs-down. </p>
    <p>The program has good accuracy to sense the semantic of the sentence as the last two sentences in the video could 
    be challenging for the program to understand, but it made the right guess.</p>
    <ol>
    <li>Food was good, and not bad at all.</li>
    <li>Food was bad, and not good at all.</li>
    </ol>
    Achieved Accuracy: 82.67% \n
    Parameter Trained: More than 2.3 million \n
    <strong>#SentiMeter #LSTM #RNN #MachineLearning</strong>
    ''', unsafe_allow_html=True)


if radio == options[0]:
    about()
elif radio == options[1]:
    tokinizer, model = load_model()
    st.header('Enter Sentence:')
    comment = st.text_input('e.g., I am feeling great or I am feeling bad.')
    cols = st.beta_columns((1, 1, 1, 1, 1))
    enter = cols[0].button('Analyze')
    reset = cols[1].button('Reset')
    if enter:
        sentiment = sentiment_analyzer(comment, tokinizer, model)
        cols_image = st.beta_columns(3)
        if sentiment:
            cols_image[1].image('images/happy.png')
        else:
            cols_image[1].image('images/sad.png')
    elif reset:
        st.empty()