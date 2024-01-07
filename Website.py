#!/usr/bin/env python
# coding: utf-8

# In[1]:


# First let's import the packages we will use in this project
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
plt.style.use('ggplot')
from matplotlib.pyplot import figure

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (12,8)

pd.options.mode.chained_assignment = None

import scipy as stats
import time
import html5lib
import lxml
from bs4 import BeautifulSoup
import joblib
from sklearn.linear_model import Ridge  # M


team_dict = {'ATL': 'Atlanta Hawks',
'BOS':'Boston Celtics',
'CHO': 'Charlotte Hornets',
'CHI': 'Chicago Bulls',
'CLE': 'Cleveland Cavaliers',
'DAL': 'Dallas Mavericks',
'DEN': 'Denver Nuggets',
'DET': 'Detroit Pistons',
'GSW': 'Golden State Warriors',
'HOU': 'Houston Rockets',
'IND': 'Indiana Pacers',
'LAC': 'Los Angeles Clippers',
'LAL': 'Los Angeles Lakers',
'MEM': 'Memphis Grizzlies',
'MIA': 'Miami Heat',
'MIL': 'Milwaukee Bucks',
'MIN': 'Minnesota Timberwolves',
'NOP': 'New Orleans Pelicans',
'NYK': 'New York Knicks',
'BRK': 'Brooklyn Nets',
'OKC': 'Oklahoma City Thunder',
'ORL': 'Orlando Magic',
'PHI': 'Philadelphia 76ers',
'PHO': 'Phoenix Suns',
'POR': 'Portland Trail Blazers',
'SAC': 'Sacramento Kings',
'SAS': 'San Antonio Spurs',          
'TOR': 'Toronto Raptors',
'UTA': 'Utah Jazz',
'WAS': 'Washington Wizards',
'SEA': 'Seattle SuperSonics',
'NOK': 'New Orleans/Oklahoma City Hornets',
'NOH': 'New Orleans Hornets',
'CHA': 'Charlotte Bobcats',            
'SDC': 'San Diego Clippers',
'NJN': 'New Jersey Nets',         
'KCK': 'Kansas City Kings',
'WSB': 'Washington Bullets',
'VAN': 'Vancouver Grizzlies ',
'CHH': 'Charlotte Hornets',
'TOT': 'Traded'
            }



#Advanced Stats

adv = pd.read_html('https://www.basketball-reference.com/leagues/NBA_2024_advanced.html', match='Advanced')[0]
drop_RK = adv[adv['Rk'] == 'Rk']
adv = adv.drop(drop_RK.index)
advanced_list = ['Age', 'G', 'MP', 'PER', 'TS%', '3PAr',
           'FTr', 'ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%',
           'OWS', 'DWS', 'WS', 'WS/48', 'OBPM', 'DBPM', 'BPM', 'VORP']
adv[advanced_list] = adv[advanced_list].astype('float64')
adv = adv.set_index('Rk')
#     df= df.drop(df[df['Tm'] == 'TOT'].index, axis=0)
adv['Team'] = adv['Tm'].map(team_dict)
adv['Player'] = adv['Player'].map(lambda title:title.rstrip('*'))
adv= adv.drop(adv[['Unnamed: 19', 'Unnamed: 24']], axis=1)
adv['Season'] = 'https://www.basketball-reference.com/leagues/NBA_2024_advanced.html'.split('_')[1][0:4]
adv['ID'] = adv['Season'] + ' ' + adv['Player'] + ' ' + adv['Team']
adv['Team ID'] = adv['Season'] + ' ' + adv['Team']
adv.to_csv(f'C:\\Users\\Owner\\Documents\\Python Scripts\\NBA MVP Model files\\Advanced stats\\2024 Advanced Stats.csv')


#normalizing it
norm_adv = adv.copy()
norm_adv[['G', 'MP', 'PER', 'TS%', '3PAr',
       'FTr', 'ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%',
       'OWS', 'DWS', 'WS', 'WS/48', 'OBPM', 'DBPM', 'BPM', 'VORP']] = (norm_adv[['G', 'MP', 'PER', 'TS%', '3PAr',
       'FTr', 'ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%',
       'OWS', 'DWS', 'WS', 'WS/48', 'OBPM', 'DBPM', 'BPM', 'VORP']]- norm_adv[['G', 'MP', 'PER', 'TS%', '3PAr',
       'FTr', 'ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%',
       'OWS', 'DWS', 'WS', 'WS/48', 'OBPM', 'DBPM', 'BPM', 'VORP']].mean()) / norm_adv[['G', 'MP', 'PER', 'TS%', '3PAr',
       'FTr', 'ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%',
       'OWS', 'DWS', 'WS', 'WS/48', 'OBPM', 'DBPM', 'BPM', 'VORP']].std()
norm_adv.to_csv(f'C:\\Users\\Owner\\Documents\\Python Scripts\\NBA MVP Model files\\Advanced stats\\2024 Advanced Stats Normalized.csv')

#Per Game

per = pd.read_html('https://www.basketball-reference.com/leagues/NBA_2024_per_game.html', match='Player Per Game')[0]
per = per.fillna(0)
drop_RK2 = per[per['Rk'] == 'Rk']
per = per.drop(drop_RK2.index)
per_cols = ['Age', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%',
       '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%',
       'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']

per[per_cols] = per[per_cols].astype('float64')
per = per.set_index('Rk')
#     df= df.drop(df[df['Tm'] == 'TOT'].index, axis=0)
per['Player'] = per['Player'].map(lambda title:title.rstrip('*'))
per['Season'] = 'https://www.basketball-reference.com/leagues/NBA_2024_per_game.html'.split('_')[1][0:4]
per['GameScore'] = per['PTS'] + 0.4 * per['FG'] - 0.7 * per["FGA"] - 0.4*(per['FTA'] - per["FT"]) + 0.7 * per["ORB"] + 0.3 * per["DRB"] + per["STL"] + 0.7 * per["AST"] + 0.7 * per["BLK"] - 0.4 * per["PF"] - per["TOV"]
per['Team'] = per['Tm'].map(team_dict)
per['ID'] = per['Season'] + ' ' + per['Player'] + ' ' + per['Team']
per['Team ID'] = per['Season'] + ' ' + per['Team']
per.to_csv(f'C:\\Users\\Owner\\Documents\\Python Scripts\\NBA MVP Model files\\Player Per Game\\2024 Per Game Stats.csv')

#normalizing it
norm_per = per.copy()
norm_per[['G', 'GS', 'MP', 'FG', 'FGA', 'FG%',
       '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%',
       'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS','GameScore']] = (norm_per[['G', 'GS', 'MP', 'FG', 'FGA', 'FG%',
       '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%',
       'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS','GameScore']] - norm_per[['G', 'GS', 'MP', 'FG', 'FGA', 'FG%',
       '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%',
       'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS','GameScore']].mean()) / norm_per[['G', 'GS', 'MP', 'FG', 'FGA', 'FG%',
       '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%',
       'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS','GameScore']].std()
norm_per.to_csv(f'C:\\Users\\Owner\\Documents\\Python Scripts\\NBA MVP Model files\\Player Per Game\\2024 Per Game Stats Normalized.csv')

#Team Advanced Stats

team = pd.read_html('https://www.basketball-reference.com/leagues/NBA_2024.html', match='Advanced Stats')[0]
team.columns = [col[1] for col in team.columns]
team = team.drop(['Unnamed: 22_level_1', 'Unnamed: 27_level_1'], axis=1)
team = team.drop('Unnamed: 17_level_1', axis=1)
team = team.set_index('Rk')
#Dropping FT/FGA columns
team = team.drop(team.columns[23],axis=1)
#Dropping League Average Row
team = team.drop(team[team['Team'] == 'League Average'].index)
team['Team'] = team['Team'].map(lambda title:title.rstrip('*'))
team['Season'] = 'https://www.basketball-reference.com/leagues/NBA_2024.html'.split('_')[1][0:4]
team['Team ID'] = team['Season'] + ' ' + team['Team']
team.to_csv(f'C:\\Users\\Owner\\Documents\\Python Scripts\\NBA MVP Model files\\Team Advanced stats\\2024 Team Advanced Stats.csv')

#normalizing it

norm_team = team.copy()
# Subset of columns to transform
cols = ['W', 'L', 'PW', 'PL', 'MOV', 'SOS', 'SRS', 'ORtg',
       'DRtg', 'NRtg', 'Pace', 'FTr', '3PAr', 'TS%']

def standardize(column):
    return (column - column.mean()) / column.std()

# Standardize column 'A' using the apply function
norm_team[cols] = norm_team[cols].apply(standardize)
norm_team.to_csv(f'C:\\Users\\Owner\\Documents\\Python Scripts\\NBA MVP Model files\\Team Advanced stats\\2024 Team Advanced Stats Normalized.csv')

norm_team.rename(columns = {"ORtg": "Team ORtg", "DRtg": "Team DRtg", "NRtg": "Team NRtg", "Pace": "Team Pace", "FTr": "Team FTr", "3PAr": "Team 3PAr","TS%": "Team TS%" }, inplace=True)
team.rename(columns = {"ORtg": "Team ORtg", "DRtg": "Team DRtg", "NRtg": "Team NRtg", "Pace": "Team Pace", "FTr": "Team FTr", "3PAr": "Team 3PAr","TS%": "Team TS%" }, inplace=True)







#Combining Dataframe
import glob
import os
import pandas as pd

# the path to your csv file directory
mycsvdir = 'C:\\Users\\Owner\\Documents\\Python Scripts\\NBA MVP Model files\\Player Per Game\\'



# get all the csv files in that directory (assuming they have the extension .csv)
csvfiles = glob.glob(os.path.join(mycsvdir, '*Normalized.csv'))

# loop through the files and read them in with pandas
dataframes = []  # a list to hold all the individual pandas DataFrames
for n in range(0,45):
    df = pd.read_csv(csvfiles[n])
    dataframes.append(df)

    
# concatenate them all together
all_per_norm = pd.concat(dataframes, ignore_index=True)

# print out to a new csv file
all_per_norm.to_csv('all_per_norm.csv')

all_per_norm = all_per_norm.drop('Unnamed: 0', axis=1)
all_per_norm = all_per_norm.drop('Rk', axis=1)



curr_per_adv_norm = pd.merge(norm_per, norm_adv[['PER', 'TS%', '3PAr',
       'FTr', 'ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%',
       'OWS', 'DWS', 'WS', 'WS/48', 'OBPM', 'DBPM', 'BPM', 'VORP','ID']],'left', on=['ID'])
curr_per_adv_norm.to_csv('curr_per_adv_norm.csv')



curr_per_adv_team_norm = pd.merge(curr_per_adv_norm, norm_team[['SOS', 'SRS', 'Team ORtg',
       'Team DRtg', 'Team NRtg', 'Team Pace', 'Team FTr', 'Team 3PAr',
       'Team TS%','Team ID']],'left', on=['Team ID'])
curr_per_adv_team_norm.to_csv('curr_per_adv_team_norm.csv')

all_data = pd.read_csv('all_per_adv_team_norm_mvps_dpoy.csv')

#ROOKIES
import pandas as pd

# Assuming you have a DataFrame called 'df' with a column 'Person' and a new column 'UHOH'
# Also, assuming that you have a 'Year' column indicating the year

# Sort the DataFrame by 'Person' and 'Year'
all_data.sort_values(by=['Player', 'Season'], inplace=True)

# Initialize a new column 'UHOH' with 0 for all rows
all_data['Rookie'] = 0

# Iterate through the DataFrame to check if a person appears in the previous year
for index, row in all_data.iterrows():
    person = row['Player']
    
    # Find the previous year's data for the same person
    previous_years_row = all_data[(all_data['Player'] == person) & (all_data['Season'] < row['Season'])]
    
    # If the person did not appear in the previous year, set 'UHOH' to 1
    if previous_years_row.empty:
        all_data.at[index, 'Rookie'] = 1
        
        
import pandas as pd

# Assuming you have a DataFrame called 'df' with columns 'Person', 'PreviousYearFigure', and 'UHOH'
# Also, assuming that the figure to check against is stored in a variable called 'certain_figure'

certain_figure = 1  # Replace this with the figure you're checking against

# Sort the DataFrame by 'Person' and 'Year' (assuming you have a 'Year' column)
all_data.sort_values(by=['Player', 'Season'], inplace=True)

# Initialize a new column 'UHOH' with 'No' for all rows
all_data['All D Team'] = 0

# Iterate through the DataFrame to check if the person met the certain figure in any previous year
for index, row in all_data.iterrows():
    person = row['Player']
    
    # Check if the person met the certain figure in any previous year
    previous_years_met_condition = (all_data['Player'] == person) & (all_data['Season'] == row['Season']) & (all_data['MVP Winner'] >= certain_figure)
    
    # If met in any previous year, set 'UHOH' to 'Yes' for subsequent years
    if previous_years_met_condition.any():
        person_indices = (all_data['Player'] == person) & (all_data['Season'] >= row['Season'])
        all_data.loc[person_indices, 'All D Team'] = 1

        
        
        
# Initialize a new column 'UHOH' with 'No' for all rows
all_data['Won DPOY Before'] = 0

# Iterate through the DataFrame to check if the person met the certain figure in any previous year
for index, row in all_data.iterrows():
    person = row['Player']
    
    # Check if the person met the certain figure in any previous year
    previous_years_met_condition = (all_data['Player'] == person) & (all_data['Season'] == row['Season']) & (all_data['DPOY Winner'] >= certain_figure)
    
    # If met in any previous year, set 'UHOH' to 'Yes' for subsequent years
    if previous_years_met_condition.any():
        person_indices = (all_data['Player'] == person) & (all_data['Season'] >= row['Season'])
        all_data.loc[person_indices, 'Won DPOY Before'] = 1


        
        
# Initialize a new column 'UHOH' with 'No' for all rows
all_data['All D Team'] = 0

# Iterate through the DataFrame to check if the person met the certain figure in any previous year
for index, row in all_data.iterrows():
    person = row['Player']
    
    # Check if the person met the certain figure in any previous year
    previous_years_met_condition = (all_data['Player'] == person) & (all_data['Season'] == row['Season']) & (all_data['# Tm'] >= certain_figure)
    
    # If met in any previous year, set 'UHOH' to 'Yes' for subsequent years
    if previous_years_met_condition.any():
        person_indices = (all_data['Player'] == person) & (all_data['Season'] >= row['Season'])
        all_data.loc[person_indices, 'All D Team'] = 1
        
                
all_datahmm = all_data[all_data['Season'] == 2024]
current_data = pd.merge(curr_per_adv_team_norm, all_datahmm[['MVP Winner', 'DPOY Winner',
       '# Tm', 'ROTY Winner','Player', 'Won DPOY Before',"All D Team", "Rookie"]],'left', on=['Player'])
current_data.to_csv('current_data.csv')

mvp_cols = ['FG', 'FGA', 'FG%', '3P',
       '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB',
       'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS','AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'WS', 'BPM', 'VORP', 'SOS', 'SRS', 'Team ORtg', 'Team DRtg',
       'Team NRtg',  'Team TS%', 'GameScore']
dpoy_cols = ['ORB',
       'DRB', 'TRB', 'STL', 'BLK', 'PF', 'PTS','STL%', 'BLK%', 'DWS', 'BPM', 'VORP', 'SOS', 'SRS', 'Team DRtg',
       'Team NRtg', "All D Team"]
roty_cols = ["MP",'FG', 'FGA', 'FG%', '3P',
       '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB',
       'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS','GameScore', 'USG%', 'WS', 'BPM',"ROTY Winner"]

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from joblib import dump,load
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
import scipy as sp

loaded_model = load('best_regression_model.joblib')

current = current_data[mvp_cols]
current = current.drop_duplicates()
current = current.fillna(current.mean())
current["MVP Prediction"] = loaded_model.predict(current)
current.sort_values('MVP Prediction', ascending=False)
current.sort_index(inplace=True)

player_info = current_data[['ID', 'Player', 'Pos', 'Team', 'Team ID', 'Tm']]
player_info.sort_index(inplace=True)

current[['ID', 'Player', 'Pos', 'Team', 'Team ID', 'Tm']] = player_info[['ID', 'Player', 'Pos', 'Team', 'Team ID', 'Tm']]  

current['MVP Probability'] = current['MVP Prediction'] / current['MVP Prediction'].sum() * 100

loaded_model = load('prac_dpoy_model.joblib')

dpoy = current_data[dpoy_cols] 
dpoy = dpoy.drop_duplicates()
dpoy = dpoy.fillna(dpoy.mean())
dpoy["DPOY Prediction"] = loaded_model.predict(dpoy)
dpoy.sort_values('DPOY Prediction', ascending=False)
dpoy.sort_index(inplace=True)

player_info = current_data[['ID', 'Player', 'Pos', 'Team', 'Team ID', 'Tm']]
player_info.sort_index(inplace=True)

dpoy[['ID', 'Player', 'Pos', 'Team', 'Team ID', 'Tm']] = player_info[['ID', 'Player', 'Pos', 'Team', 'Team ID', 'Tm']]  

dpoy['DPOY Probability'] = dpoy['DPOY Prediction'] / dpoy['DPOY Prediction'].sum() * 100



loaded_model = load('prac_roty_model.joblib')
new_rookies = current_data[current_data['Rookie']==1]


rooks = new_rookies[["MP",'FG', 'FGA', 'FG%', '3P',
       '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB',
       'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'GameScore' ,'USG%', 'WS', 'BPM']] 
rooks = (rooks - rooks.mean()) / rooks.std()

rooks = rooks.drop_duplicates()
rooks["ROTY Prediction"] = loaded_model.predict(rooks)
rooks.sort_values('ROTY Prediction', ascending=False)
rooks.sort_index(inplace=True)

player_info = new_rookies[['ID', 'Player', 'Pos', 'Team', 'Team ID', 'Tm']]
player_info.sort_index(inplace=True)

rooks[['ID', 'Player', 'Pos', 'Team', 'Team ID', 'Tm']] = player_info[['ID', 'Player', 'Pos', 'Team', 'Team ID', 'Tm']]  

rooks['ROTY Probability'] = rooks['ROTY Prediction'] / rooks['ROTY Prediction'].sum() * 100

show_data = pd.merge(current_data, current[['MVP Prediction', 'ID']],'left', on=['ID'])
show_data = pd.merge(show_data, dpoy[['DPOY Prediction', 'ID']],'left', on=['ID'])
show_data = pd.merge(show_data, rooks[['ROTY Prediction', 'ID']],'left', on=['ID'])
show_data = pd.merge(show_data, per,'left', on=['ID'])
show_data['Team ID'] = show_data['Team ID_y']
show_data = pd.merge(show_data, team,'left', on=['Team ID'])

show_data.to_csv('show_data.csv')


mvp_leaders = show_data.sort_values('MVP Prediction', ascending=False).head(20)[["MVP Prediction",'Player_y', 'Pos_y', 'Team ID','W','L','Team NRtg_y', 'Age_y', "PTS_y", "TRB_y", "AST_y", "STL_y", "BLK_y", "TOV_y"]]
mvp_leaders.rename(columns = {"Player_y": "Player", "Pos_y": "Pos", "Age_y": "Age", "PTS_y": "PTS", "TRB_y": "TRB", "AST_y": "AST","STL_y": "STL","BLK_y": "BLK","TOV_y": "TOV","Team NRtg_y": "Team NRtg" }, inplace=True)
mvp_leaders.to_csv('C:\\Users\\Owner\\Downloads\\mvp_leaders.csv')

dpoy_leaders = show_data.sort_values('DPOY Prediction', ascending=False).head(20)[["DPOY Prediction",'Player_y', 'Pos_y', 'Team ID','W','L','Team DRtg_y', 'Age_y', "PTS_y", "TRB_y", "AST_y", "STL_y", "BLK_y", "TOV_y"]]
dpoy_leaders.rename(columns = {"Player_y": "Player", "Pos_y": "Pos", "Age_y": "Age", "PTS_y": "PTS", "TRB_y": "TRB", "AST_y": "AST","STL_y": "STL","BLK_y": "BLK","TOV_y": "TOV","Team DRtg_y": "Team DRtg" }, inplace=True)
dpoy_leaders.to_csv('C:\\Users\\Owner\\Downloads\\dpoy_leaders.csv')



roty_leaders = show_data.sort_values('ROTY Prediction', ascending=False).head(20)[["ROTY Prediction",'Player_y', 'Pos_y', 'Team ID','W','L','Team NRtg_y', 'Age_y', "PTS_y", "TRB_y", "AST_y", "STL_y", "BLK_y", "TOV_y"]]
roty_leaders.rename(columns = {"Player_y": "Player", "Pos_y": "Pos", "Age_y": "Age", "PTS_y": "PTS", "TRB_y": "TRB", "AST_y": "AST","STL_y": "STL","BLK_y": "BLK","TOV_y": "TOV","Team NRtg_y": "Team NRtg" }, inplace=True)
roty_leaders.to_csv('C:\\Users\\Owner\\Downloads\\roty_leaders.csv')



# In[ ]:




