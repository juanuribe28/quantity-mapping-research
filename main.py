import pandas as pd
import readCSV as csv
import retrieveSSH as ssh
import string_matching as strMatch

# --- Retrieve files from SSH

host = '10.5.97.223'
username = 'juribe'
password = 'juribe'

ssh.retrieve_file(host, username, password, '/home/mkorpusi/food_units.csv', 'data/food_units.csv')
retrieved_files = ssh.retrive_dir(host, username, password, '/home/mkorpusi/quant_data', 'data', exceptions=('Speech', 'test.csv'))

# --- Organize data

data_by_meal = {}
exceptions = ['Answer.quantity3_res']
column_names =['Answer.diary', 'Answer.quantity', 'Input.nbd', 'Input.foodName', 'Input.quant']

for directory in retrieved_files:
    data_by_meal_df = pd.DataFrame()
    for file in retrieved_files[directory]:
        temp_data = csv.read_and_organize('data/{}/{}'.format(directory, file), column_names, exceptions=exceptions)
        data_by_meal_df = pd.concat([data_by_meal_df, temp_data], ignore_index=True, sort=False)
    data_by_meal.update({directory : data_by_meal_df})

all_data = pd.concat([data_by_meal[key] for key in data_by_meal], names=[key for key in data_by_meal], ignore_index=True, sort=False)
all_data = csv.drop_value(all_data)

# --- Organize food_units.csv

food_units = pd.read_csv('data/food_units_organized.csv')

#--------- STRING MATCHING ---------#

correct_percentage, matches = strMatch.stringMatching(all_data, food_units)

#--------- LOGISTIC REGRESSION ---------#
 # First, logistic multiclass: each database option is a class and every input is classified to one of them
 # Second, logistic binary match: each quantity entry is compared to each database option and is classified
 #                                as match or not. The inputs are encoded using a BOW.
 # Third, logistic binary string comparison: each quantity entry is compared to each database option and is 
 #                                           classified as match or not. The input for the logistic regression
 #                                           are string matching values.

#--------- NEURAL NETWORK ---------#
