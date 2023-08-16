import pandas as pd

# load dataset

user_units_path = 'data/all_data.csv'
user_units = pd.read_csv(user_units_path)

database_units_path = 'data/organized.csv'
database_units = pd.read_csv(database_units_path)

# make BOW

from sklearn.feature_extraction.text import CountVectorizer

bow_vec = CountVectorizer()
bow_vec.fit(list(user_units.loc[:,'Answer.quantity'])+list(database_units.loc[:,'Input.quant']))

# organize all the data and encode with BOW

input_list = []
database_list = []
match_list = []

correct_count = 0
incorrect_count = 0
discarded = 0

for row_index in user_units.index:
    
    row = user_units.iloc[row_index] 
    answer_quant = row['Answer.quantity']   
    nbd = row['Input.nbd']
    
    user_options = []
    for i in range(1,17):
        quant = row['Input.quant'+str(i)]
        if quant != 'None':
            user_options.append(quant)
    
    quant_options_df = database_units.loc[lambda database_units: database_units['Input.nbd'] == nbd]
    quant_options = list(quant_options_df['Input.quant'])
        
    for quant_option in quant_options:
        if quant_option in user_options:
            correct_count +=1
            input_list.append(pd.SparseArray(bow_vec.transform([answer_quant]).toarray()[0]))
            database_list.append(pd.SparseArray(bow_vec.transform([quant_option]).toarray()[0]))
            match_list.append([1])
            
        elif incorrect_count <= correct_count:
            incorrect_count +=1
            input_list.append(pd.SparseArray(bow_vec.transform([answer_quant]).toarray()[0]))
            database_list.append(pd.SparseArray(bow_vec.transform([quant_option]).toarray()[0]))
            match_list.append([0])
            
        else:
            discarded +=1
        
input_df = pd.DataFrame(input_list, dtype = 'Sparse[int]')
database_df = pd.DataFrame(database_list, dtype = 'Sparse[int]')
match_df = pd.DataFrame(match_list, dtype = 'Sparse[int]')

# Save dataframes to file
import pickle

def write_pickle(objects):
    for key in objects:
        with open('Files/{}'.format(key), 'wb') as file:
            pickle.dump(objects[key], file)

write_pickle({'input.df' : input_df, 
              'database.df' : database_df,  
              'match.df' : match_df})

# divide the data in x and y

x = pd.concat([input_df, database_df], axis = 1, sort = False)
y = match_df  

# create training and test sets
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)

#make predictions and confirm effectivity
y_hat = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix,classification_report

cml = confusion_matrix(y_test, y_hat)
print('Logistic Report:')
print(classification_report(y_test, y_hat))
