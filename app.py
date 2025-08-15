import pandas as pd
import streamlit as st
import pickle
teams=['Sunrisers Hyderabad',
         'Mumbai Indians',
         'Royal Challengers Bangalore',
         'Kolkata Knight Riders',
         'Kings XI Punjab',
         'Chennai Super Kings',
         'Rajasthan Royals',
         'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
       'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
       'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
       'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
       'Sharjah','Mohali', 'Bengaluru']

pipe=pickle.load(open('pipe.pkl','rb'))

st.title('IPL Prediction')

col1 , col2=st.columns(2)

with col1:
    batting_team = st.selectbox('Select the Batting Team',teams)
with col2:
     bowling_team = st.selectbox('Select the Bowling Team',teams)

selected_city= st.selectbox('Select Host City',(cities))

target=st.number_input('Target')

col3, col4, col5=st.columns(3)

with col3:
   score= st.number_input('Score')
with col4:
    overs=st.number_input('Overs Completed')
with col5:
    wickets=st.number_input('Wickets')

if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wicket_left = 10 - wickets
    crr = score / overs if overs > 0 else 0  # Avoid division by zero
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0  # Avoid div by zero

    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'balls_left': [balls_left],
        'runs_left': [runs_left],
        'wickets': [wicket_left],  # Use wicket_left, not wickets
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })
    st.table(input_df)
    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]
    st.write(batting_team + "=" + str(round(win*100)) + "%")
    st.write(bowling_team + "=" + str(round(loss*100)) + "%")

    
