import pandas as pd

adult = pd.read_csv('full data/adult.csv')
compas = pd.read_csv('full data/compas.csv')
german = pd.read_csv('full data/german_data.csv')

datasets = [adult, compas, german]
names = ['adult', 'compas', 'german']

for i in range(len(datasets)):
    data = datasets[i]
    data.drop(labels='Unnamed: 0', axis=1, inplace=True)
    try:
        del data['Unnamed: 0.1']
    except:
        pass
    for j in range(15):
        data_sample = data.sample(n=500)
        data_sample.reset_index(drop=True, inplace=True)
        data_sample.to_csv(f'samples/{names[i]}_{j}', index=False)