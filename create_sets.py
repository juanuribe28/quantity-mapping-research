import pandas as pd

user_units_path = 'data/all_data.csv'
user_units = pd.read_csv(user_units_path)

database_units_path = 'data/food_units_organized.csv'
database_units = pd.read_csv(database_units_path)

correct_count = 0
incorrect_count = 0
discarded = 0

data_list = []
discarded_data_list = []

for row_index in user_units.index:
    
    row = user_units.iloc[row_index] 
    answer_quant = row['Answer.quantity'].lower().replace(',','')    # Removed all capitalization and commas
    nbd = row['Input.nbd']
    
    user_options = []
    for i in range(1,17):
        quant = row['Input.quant'+str(i)]
        if quant != 'None':
            user_options.append(quant)
    
    quant_options_df = database_units.loc[lambda database_units: database_units['Input.nbd'] == nbd]
    quant_options = list(quant_options_df['Input.quant'])
        
    for quant_option in quant_options:
        
        quant_option = quant_option.lower().replace(',','') # Removed all capitalization and commas
            
        if quant_option in user_options:
            correct_count +=1
            new_row = [answer_quant, quant_option, 1]
            data_list.append(new_row)
            
        elif incorrect_count <= correct_count:
            incorrect_count +=1
            new_row = [answer_quant, quant_option, 0]
            data_list.append(new_row)
            
        else:
            discarded +=1
            new_row = [answer_quant, quant_option, 0]
            discarded_data_list.append(new_row)
            
    if correct_count > incorrect_count:
        for i in range(correct_count-incorrect_count):
            incorrect_count += 1
            data_list.append(discarded_data_list[-i])
        
data_df = pd.DataFrame(data_list)
discarded_data_df = pd.DataFrame(discarded_data_list)

from sklearn.model_selection import train_test_split
data_train, data_test = train_test_split(data_df, test_size = 0.2, stratify=data_df[2])

data_train.to_csv('data/train_dataset.csv', index = False, header = False)
data_test.to_csv('data/test_dataset.csv', index = False, header = False)

#create a small version to test NN construction in my computer

correct_count = 0
incorrect_count = 0

small_data_list = []
    
row = user_units.iloc[3] 
answer_quant = row['Answer.quantity'].lower().replace(',','')  
nbd = row['Input.nbd']

user_options = []
for i in range(1,17):
    quant = row['Input.quant'+str(i)]
    if quant != 'None':
        user_options.append(quant)

quant_options_df = database_units.loc[lambda database_units: database_units['Input.nbd'] == nbd]
quant_options = list(quant_options_df['Input.quant'])
    
for quant_option in quant_options:
    
    quant_option = quant_option.lower().replace(',','')
    
    if quant_option in user_options:
        correct_count +=1
        new_row = [answer_quant, quant_option, 1]
        small_data_list.append(new_row)
        
    else:
        incorrect_count +=1
        new_row = [answer_quant, quant_option, 0]
        small_data_list.append(new_row)

small_train_i = [2,5,6,7]
small_test_i = [0,3]

small_data_train = pd.DataFrame([small_data_list[i] for i in small_train_i])
small_data_test = pd.DataFrame([small_data_list[i] for i in small_test_i])

small_data_train.to_csv('data/s_train_dataset.csv' , index = False, header = False)
small_data_test.to_csv('data/s_test_dataset.csv' , index = False, header = False)