# Przewidywanie udaru za pomocą TabNet


## Model TabNet
Aby uruchomić trening modelu powinno się wpisać w terminalu komendę:
```bash
python3 src/train.py 
```

Po wpisaniu tego polecenia zostanie zainicjowane szkolenie modelu przewidującego udar (powinno to zająć około 4-5 minut)

Aby sprawdzić działanie modelu należy wypełnić dane do sprawdzenia w pliku example_input.py, przykładowo tak powinno to wyglądać:

```python
gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status
Female,54,1,0,Yes,Private,Urban,171.23,31.4,never smoked
Male,45,0,0,Yes,Self-employed,Rural,98.45,27.8,smokes
```

Po uzupełnieniu danych należy wywołać polecenie:

```bash
python3 src/infer.py --input example_input.csv
```


## Model Regresji Logarytmicznej

Aby wyuczyć na tych samych danych model regresji logarytmiczne, by moc porównać metryki obu modeli należy wywołać polecenie:

```bash
python3 compare_logistic.py
```