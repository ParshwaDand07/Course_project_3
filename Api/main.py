import streamlit as st
import numpy as np
import pandas as pd 
import pickle
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Cricket", page_icon="🏏")

pages = ["Score Prediction", "ODI Match Winner", "11 Player selector","Total Sixes in this Tournament"]

# Add a selectbox in the sidebar for navigation
selected_page = st.sidebar.selectbox("Navigate to:", pages)

batting_teams = ["Afghanistan", "Australia", "Bangladesh", "England", "India", "Netherlands", "New Zealand", "Pakistan", "South Africa", "Sri Lanka"]
bowling_teams = batting_teams 
stadium_dict={'Narendra Modi Stadium, Ahmedabad': 1,
        'Rajiv Gandhi International Stadium, Uppal, Hyderabad': 2,
        'Himachal Pradesh Cricket Association Stadium, Dharamsala': 3,
        'Arun Jaitley Stadium, Delhi': 4,
        'MA Chidambaram Stadium, Chepauk, Chennai': 5,
        'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow': 6,
        'Maharashtra Cricket Association Stadium, Pune': 7,
        'M Chinnaswamy Stadium, Bengaluru': 8,
        'Wankhede Stadium, Mumbai': 9,
        'Eden Gardens, Kolkata': 10}

venues = list(stadium_dict.keys())

if selected_page == "Score Prediction":
    with open('trained_model.pkl', 'rb') as model_file:
            Final = pickle.load(model_file)
            
 
    # Create a dictionary with keys based on batting and bowling teams
    team_dict = {
            f"batting_team_{team}": 0 for team in batting_teams
        }
    team_dict.update({
            f"bowling_team_{team}": 0 for team in bowling_teams
        })

    # Setup title page
    st.header("Score Prediction")
    st.sidebar.header("Make Prediction")


    # Add the default value to the options temporarily
    batting_teams = ["Select batting Team"] + batting_teams
    bowling_teams = ["Select bowling Team"] + bowling_teams
    venues = ["Select a venue"] + venues


    # Add a dropdown to the Streamlit app
    batting_team = st.sidebar.selectbox("Select batting team:", batting_teams)
    bowling_team = st.sidebar.selectbox("Select bowling team:", bowling_teams)
    venue = st.sidebar.selectbox("Select venue:", venues)
    make_pred = st.sidebar.button("Predict")

    if batting_team == bowling_team:
        st.sidebar.warning("Batting team and bowling team cannot be the same. Please select different teams.")

        

    if make_pred and batting_team != bowling_team:
        if (batting_team == 'Select batting Team') or (bowling_team == 'Select bowling Team') or (venue == 'Select a venue'):
            st.sidebar.warning("Please select all options correctly")
        else:
            user_batting_team = batting_team
            user_bowling_team = bowling_team
            venue = venue

            df = pd.DataFrame.from_dict(team_dict, orient='index', columns=['Value'])
            df= df.transpose()

            column_name = f"batting_team_{user_batting_team}"

            if column_name in df.columns:

                df[column_name] = 1

            else:
                print(f"Column '{column_name}' not found in the DataFrame.")
                
            column_name = f"bowling_team_{user_bowling_team}"

            if column_name in df.columns:

                df[column_name] = 1

            else:
                print(f"Column '{column_name}' not found in the DataFrame.")

            df['venue'] = stadium_dict[venue]
            df['ball'] = 0.0
            df['cumulative_runs'] = 0
            df['wickets'] = 0

            

            score=list()
            start = 0.0
            stop = 50.0
            step = 10.0

            for i in range(int(start * 10), int(stop * 10) + 1, int(step * 10)):
                i /= 10.0
                df['ball'] = i
                score.append (Final.predict(df).round(0).astype(int))
                
            lower = min(score)
            st.subheader(f"{user_batting_team} will score between {int(lower-10)} to {int(lower+10)} ")


elif selected_page == "ODI Match Winner":
    def final():
        winner = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(33,)), 
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

        winner.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 
        return winner
    
    with open('pipe.pkl', 'rb') as model_file:
        pipe = pickle.load(model_file)

    st.header('ODI Match Winner')
    st.sidebar.header("Make Prediction")


    # Add the default value to the options temporarily
    batting_teams = ["Select batting Team"] + batting_teams
    bowling_teams = ["Select bowling Team"] + bowling_teams

    venues = ["Select a venue"] + venues


    # Add a dropdown to the Streamlit app
    batting_team = st.sidebar.selectbox("Select batting team:", batting_teams)
    bowling_team = st.sidebar.selectbox("Select bowling team:", bowling_teams)
    runs_left = st.sidebar.text_input("Enter runs needed to chase:")
    wickets_left = st.sidebar.text_input("Enter Current wickets left:")
    ball_left = st.sidebar.text_input("Enter number of ball left:")
    Current_runs = st.sidebar.text_input("Enter current run:")
    target = st.sidebar.text_input("Enter target run:")
    venue = st.sidebar.selectbox("Select venue:", venues)
    make_pred = st.sidebar.button("Predict")

    if batting_team == bowling_team:
        st.sidebar.warning("Batting team and bowling team cannot be the same. Please select different teams.")

        

    if make_pred and batting_team != bowling_team:
        if (batting_team == 'Select batting Team') or (bowling_team == 'Select bowling Team') or (venue == 'Select a venue'):
            st.sidebar.warning("Please select all options correctly")
        else:
            data = {
                'batting_team': [batting_team],
                'bowling_team': [bowling_team],
                'venue': [venue],
                'runs_left': [int(runs_left)],
                'balls_left': [int(ball_left)],
                'wickets_left': [int(wickets_left)],
                'total':[int(target)]
                }

            final_cricket = pd.DataFrame(data)
            if(final_cricket['balls_left']==300).any():
                final_cricket['crr']=0
            else:
                final_cricket['crr'] = int(Current_runs) * 6 / (300 - final_cricket['balls_left'])
            final_cricket['rrr'] = final_cricket['runs_left'] * 6 / final_cricket['balls_left']
            
            result= pipe.predict_proba(final_cricket)
            st.subheader('Winning percentage')
            st.subheader(f"{batting_team} : {(result[1]*100):.2f}% ")
            st.subheader(f"{bowling_team} : {(result[0]*100):.2f}% ")
    
elif selected_page == "11 Player selector":
    
    st.header(" 11 Player selector")
    with open('player.pkl', 'rb') as file:
        df = pickle.load(file)

    with open('player_model.pkl', 'rb') as model_file:
        play_xi = pickle.load(model_file)
        
        
    team_1 = st.sidebar.selectbox("Select team 1:", batting_teams)
    team_2 = st.sidebar.selectbox("Select team 2:", bowling_teams)
    make_pred = st.sidebar.button("Predict")
    

   

    if team_1 == team_2:
        st.sidebar.warning("Batting team and bowling team cannot be the same. Please select different teams.")

    if make_pred and team_1 != team_2:
        pred_df=df.copy()
        pred_df=pred_df.drop(['Player', 'Country'],axis=1)
        probabilities = play_xi.predict(pred_df)  
        df.loc[pred_df.index, 'PredictedProbabilities'] = probabilities
    
        
        team_1_ = df[df['Country'] == team_1].nlargest(11, 'PredictedProbabilities')['Player']
        team_2_ = df[df['Country'] == team_2].nlargest(11, 'PredictedProbabilities')['Player']  
          
        team_1_reset = team_1_.reset_index(drop=True)
        team_1_reset.index += 1  
        
        team_2_reset = team_2_.reset_index(drop=True)
        team_2_reset.index += 1 
        
        st.subheader(team_1)
        st.table(team_1_reset)
        st.subheader(team_2)
        st.table(team_2_reset)
        
elif selected_page == "Total Sixes in this Tournament":
    
    st.header(" Total Sixes in this Tournament")
    
    with open('six_df.pkl', 'rb') as file:
        df = pickle.load(file)
    with open('data1.pkl', 'rb') as file:
        df_output = pickle.load(file)
    look_back=30
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_output)
    with open('six.pkl', 'rb') as model_file:
        six = pickle.load(model_file)
    
    forecast_steps = 3600
    current_sequence = df[-1]
    predicted_values = []
    
    
    
    for i in range(forecast_steps):
        current_sequence_reshaped = np.reshape(current_sequence, (1, look_back, 1))
        
        next_value = six.predict(current_sequence_reshaped, verbose=0)
        
        predicted_values.append(next_value[0, 0])
        
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = next_value
        
    predicted_values = np.array(predicted_values).reshape(-1, 1)
    predicted_values = scaler.inverse_transform(predicted_values)
    
        
    st.subheader((f'Total Sixes in this ICC World Cup 2023 : {int(np.round(predicted_values)[-1])}'))    