from fuzzywuzzy import process
import pandas as pd

def stringMatching(user_units, database_units, output_print = True):
    
    total = 0
    correct = 0
    total_rows = len(user_units.index)
    
    index = []
    user_inputs = []
    databse_matches = []
    identification = []
    correct_matches = []
    all_units =[]
    
    for row_index in user_units.index:
        
        index.append(row_index)
        row = user_units.iloc[row_index]
        
        answer_quant = row['Answer.quantity']
        user_inputs.append(answer_quant)
        
        nbd = row['Input.nbd']
        
        quant_options_df = database_units.loc[lambda database_units: database_units['Input.nbd'] == nbd]
        quant_options = list(quant_options_df['Input.quant'])
        all_units.append(quant_options)
        
        highest = process.extractOne(answer_quant,quant_options)
        databse_matches.append(highest[0])
        
        options = []
        for x in range(1,17):
            quant = row['Input.quant'+str(x)]
            if quant != 'None':
                options.append(quant)
        correct_matches.append(options)
        
        if highest[0] in options:
            correct += 1
            identification.append(True)
        else:
            identification.append(False)
            
        total += 1
        
        if output_print:
            print('Total: {} --- Correct: {}'.format(row_index/total_rows,correct/total))
    
    matches = pd.DataFrame(list(zip(user_inputs, databse_matches, identification, correct_matches, all_units)), columns=['Input', 'Database', 'Match', 'Correct', 'All units'])
        
    correct_percent = (correct/total)*100
    
    return correct_percent, matches