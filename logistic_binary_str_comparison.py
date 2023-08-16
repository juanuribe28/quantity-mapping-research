import pandas as pd

# load dataset
user_units_path = 'data/all_data.csv'
user_units = pd.read_csv(user_units_path)

database_units_path = 'data/food_units_organized'
database_units = pd.read_csv(database_units_path)

# calculate features and organize them
from py_stringmatching.similarity_measure.monge_elkan import MongeElkan
from py_stringmatching.similarity_measure.jaro_winkler import JaroWinkler
from py_stringmatching.similarity_measure.tfidf import TfIdf
from py_stringmatching.similarity_measure.soft_tfidf import SoftTfIdf
from py_stringmatching.tokenizer.alphanumeric_tokenizer import AlphanumericTokenizer

me = MongeElkan()
jw = JaroWinkler()
tfidf = TfIdf(dampen = False)
stdidf = SoftTfIdf()
tokenizer = AlphanumericTokenizer()

def calculate_features(str1, str2):
    str1 = str1.casefold()
    str2 = str2.casefold()
    bag1 = tokenizer.tokenize(str1)
    bag2 =tokenizer.tokenize(str2)
    
    monge_elkan = me.get_raw_score(bag1, bag2)
    jaro_winkler = jw.get_sim_score(str1, str2)
    tf_idf = tfidf.get_raw_score(bag1, bag2)
    soft_tfidf = stdidf.get_raw_score(bag1, bag2)
    
    return [monge_elkan, jaro_winkler, tf_idf, soft_tfidf]

x_list = []
y_list = []

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
            
            features = calculate_features(answer_quant, quant_option)                     
            x_list.append(features)
            y_list.append([1])
            
        elif incorrect_count <= correct_count:
            incorrect_count +=1
            
            features = calculate_features(answer_quant, quant_option)                     
            x_list.append(features)
            y_list.append([0])
            
        else:
            discarded +=1
    
x = pd.DataFrame(x_list, columns = ['Monge Elkan', 'Jaro Winkler', 'Tf-Idf', 'Soft Tf-Idf'])
y = pd.DataFrame(y_list)


# create training and test sets
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

# LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
classifier.fit(x_train, y_train)

#make predictions and confirm effectivity
y_hat = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix,classification_report

cml = confusion_matrix(y_test, y_hat)
print('Logistic Report:')
print(classification_report(y_test, y_hat))
