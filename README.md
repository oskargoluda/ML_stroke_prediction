# Przewidywanie udaru za pomocą TabNet

Ten projekt jest przygotowany do publicznej publikacji, ale dane źródłowe z Kaggle nie są dołączone do repozytorium. Zamiast tego repo zawiera przykładowe dane testowe i instrukcję, jak pobrać dataset lokalnie.

## Ważne: nie publikuj danych z Kaggle w repo

Dataset `healthcare-dataset-stroke-data.csv` pochodzi z Kaggle i jest zestawem danych medycznych. Nie commituj go do publicznego repozytorium bez sprawdzenia licencji i zgodności z polityką publikacji.

Zamiast tego:
1. Pobierz dataset z Kaggle lokalnie na swoim komputerze.
2. Umieść plik w katalogu `data/raw/`.
3. Uruchom trening lokalnie.

## Model TabNet
Aby uruchomić trening modelu, najpierw pobierz dataset do lokalnego katalogu:

```bash
mkdir -p data/raw
# umieść plik healthcare-dataset-stroke-data.csv w: data/raw/
```

Następnie uruchom trening:

```bash
python3 src/train.py
```

Po wpisaniu tego polecenia zostanie zainicjowane szkolenie modelu przewidującego udar (powinno to zająć około 4-5 minut).

Aby sprawdzić działanie modelu należy wypełnić dane do sprawdzenia w pliku `example_input.csv`, przykładowo tak powinno to wyglądać:

```csv
gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status
Female,54,1,0,Yes,Private,Urban,171.23,31.4,never smoked
Male,45,0,0,Yes,Self-employed,Rural,98.45,27.8,smokes
```

Po uzupełnieniu danych należy wywołać polecenie:

```bash
python3 src/infer.py --input example_input.csv
```

## Model Regresji Logarytmicznej

Aby wyuczyć na tych samych danych model regresji logistycznej, by móc porównać metryki obu modeli, należy najpierw mieć dataset lokalnie w `data/raw/` i wykonać:

```bash
python3 compare_logistic.py
```

## Repozytorium publiczne

To repozytorium zostało przygotowane pod publikację publiczną bez danych źródłowych. Pliki `data/raw/` i `data/processed/` są ignorowane przez Git i nie powinny być commitowane.
