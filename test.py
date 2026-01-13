import pandas as pd

dane = pd.read_csv('healthcare-dataset-stroke-data.csv')


#ile osob mialo udar ile nie mialo 

udar_counts = dane['stroke'].value_counts()
print(udar_counts)
#procentowy udzial
udar_percentage = dane['stroke'].value_counts(normalize=True) * 100
print(udar_percentage)