import pandas as pd

def read_and_organize(file_name, column_names, exceptions=[], reps=3):
    
    all_data = pd.read_csv(file_name)

    # Retrivieng the data from the columns that are going to be used
    
    data = pd.DataFrame()
    
    for name in column_names:
        new_columns = all_data.filter(like=name)
        data = pd.concat([data, new_columns], axis=1)
    for exception in exceptions:
        try:
            data = data.drop(columns=exception)
        except:
            pass

    # Organizing the data, so there's only one item per row
    
    column_names_long = []
    
    for name in column_names:
        num_columns = len(data.filter(like=name).columns)/reps
        if num_columns == 1:
            column_names_long.append(name)
        else:
            for x in range(1,int(num_columns)+1):
                column_names_long.append(name+str(x))
    organized_data = pd.DataFrame(columns=column_names_long)
    
    for x in range(1,reps+1):
        actual_column = 0
        temporary_df = pd.DataFrame(columns=column_names_long)
        for column_name in column_names:
            new_data = data.filter(like=(column_name+str(x)))        
            for column in new_data:
                actual_column_name = column_names_long[actual_column]
                temporary_df[actual_column_name] = new_data[column]
                actual_column += 1
        organized_data = pd.concat([organized_data, temporary_df], ignore_index = True)
                    
    return organized_data

def drop_value(data, drop='None'):
    
    for column in data:
        test = True
        for value in data[column]:
            if value != drop:
                test = False
                break
        if test:
            data = data.drop(columns=column)
    
    return data