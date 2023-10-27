def update_values_adult(df):
    df_workclass = {'Never-worked': 1, 'Without-pay': 2, 'Local-gov': 3, 'State-gov': 4, 'Federal-gov': 5, \
            'Self-emp-inc': 6, 'Self-emp-not-inc': 7, 'Private': 8}
    df_marital = {'Never-married': 1, 'Separated': 2, 'Divorced': 3, 'Widowed': 4, 'Married-spouse-absent': 5, \
                'Married-civ-spouse': 6, 'Married-AF-spouse': 7}
    df_occupation = {'Other-service': 1, 'Craft-repair': 2, 'Handlers-cleaners': 3, 'Farming-fishing': 4, \
                    'Transport-moving': 5, 'Priv-house-serv': 6, 'Adm-clerical': 7, 'Machine-op-inspct': 8, \
                    'Tech-support': 9,  'Sales': 10, 'Protective-serv': 11, 'Exec-managerial': 12, \
                    'Prof-specialty': 13, 'Armed-Forces': 14}
    df_relationship = {'Other-relative': 1, 'Not-in-family': 2, 'Unmarried': 3, 'Own-child': 4, 'Wife': 5, 'Husband': 6}
    df_race = {'Other': 1, 'Amer-Indian-Eskimo': 2, 'Asian-Pac-Islander': 3, 'Black': 4, 'White': 5}
    #df_gender = {'Female': 1, 'Male': 2}
    # sorted by frequency in dataset
    df_native_country = {'United-States': 1, 'Mexico': 2, 'Philippines': 3, 'Germany': 4, 'Puerto-Rico': 5, 'Canada': 6,\
                        'El-Salvador': 7, 'India': 8, 'Cuba': 9, 'England': 10, 'China': 11, 'South': 12, 'Jamaica': 13,\
                        'Italy': 14, 'Dominican-Republic': 15, 'Japan': 16, 'Guatemala': 17, 'Poland': 18, 'Vietnam': 19,\
                        'Columbia': 20, 'Haiti': 21, 'Portugal': 22, 'Taiwan': 23, 'Iran': 24, 'Nicaragua': 25, 'Greece': 26,\
                        'Peru': 27, 'Ecuador': 28, 'France': 29, 'Ireland': 30, 'Thailand': 31, 'Hong': 32, 'Cambodia': 33, \
                        'Trinadad&Tobago': 34, 'Outlying-US(Guam-USVI-etc)': 35, 'Laos': 36, 'Yugoslavia': 37, 'Scotland': 38,\
                        'Honduras': 39, 'Hungary': 40, 'Holand-Netherlands': 41}
    
    columns_to_replace = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'native-country']
    values_to_replace = [df_workclass, df_marital, df_occupation, df_relationship, df_race, df_native_country]
    
    for i in range(len(columns_to_replace)):
        df[columns_to_replace[i]].replace(values_to_replace[i], inplace=True)

    return df
        
def update_values_compas(df):
    df_score = {'Low': 0, 'Medium': 1, 'High': 2}
    df['score_text'].replace(df_score, inplace=True)
    
    return df