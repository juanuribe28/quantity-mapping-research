import pandas as pd

food_units = pd.read_csv('data/food_units.csv', names=['Input.nbd', 'Input.quant', 'Input.grams'])

food_units_organized = pd.DataFrame()

total_rows = len(food_units.index)

for index, value in enumerate(food_units['Input.nbd']):
    if 'u' in value or 'm' in value:
        food_units_organized = food_units_organized.append(food_units.iloc[index], ignore_index=True)
    print('Percentage = '+str((index/total_rows)*100)+'%')


food_units_organized.to_csv('data/food_units_u_m_str.csv')


for index, value in enumerate(food_units_organized['Input.nbd']):
    if not type(value) == int:
        food_units_organized.at[index, 'Input.nbd'] = int(value[1:])

food_units_organized.to_csv('data/food_units_organized.csv', index=False)