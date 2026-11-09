# Przewidywanie udaru mózgu za pomocą TabNet

Projekt klasyfikacji binarnej: na podstawie danych demograficznych i klinicznych pacjenta (wiek, nadciśnienie, poziom glukozy, BMI itp.) model przewiduje ryzyko wystąpienia udaru mózgu. Głównym modelem jest sieć neuronowa [TabNet](https://arxiv.org/abs/1908.07442) (biblioteka `pytorch-tabnet`), a regresja logistyczna służy jako prostszy punkt odniesienia.

> **Uwaga:** projekt ma charakter edukacyjny. Model nie jest narzędziem medycznym i nie może służyć do diagnozowania ani podejmowania decyzji klinicznych.

## Dane

Projekt korzysta z publicznego zbioru [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) z Kaggle (autor: fedesoriano). Zbiór nie jest częścią tego repozytorium. Przed użyciem zapoznaj się z warunkami podanymi na stronie datasetu.

Zbiór zawiera ok. 5100 rekordów i jest silnie niezbalansowany: udar wystąpił tylko u ok. 5% pacjentów. Dlatego sama accuracy jest tu mało miarodajna i warto patrzeć na recall, F1 oraz ROC-AUC.

Aby uruchomić projekt, pobierz plik `healthcare-dataset-stroke-data.csv` i umieść go w katalogu `data/raw/`:

```
data/
└── raw/
    └── healthcare-dataset-stroke-data.csv
```

Katalogi `data/raw/` i `data/processed/` są ignorowane przez Git.

## Struktura projektu

```
├── src/
│   ├── data_preparation.py   # wczytanie, czyszczenie, inżynieria cech, podział train/val/test
│   ├── model.py              # opakowanie modelu TabNet (trening, predykcja, zapis/odczyt)
│   ├── train.py              # pełny pipeline: dane -> model -> trening -> ewaluacja
│   ├── infer.py              # predykcja dla nowych pacjentów z pliku CSV
│   └── utils.py              # metryki i wykresy
├── evaluate.py               # szczegółowa ewaluacja zapisanego modelu
├── compare_logistic.py       # porównanie z regresją logistyczną
├── notebooks/
│   └── sprawozdanie.ipynb    # sprawozdanie z projektu
├── models/                   # wytrenowany model i informacje o preprocessingu
├── results/                  # wykresy (EDA, macierz pomyłek, ROC, ważność cech) i metrics.json
├── example_input.csv         # przykładowe dane wejściowe do predykcji
└── requirements.txt
```

## Instalacja

Wymagany jest Python 3.10+ (wersje bibliotek są przypięte w `requirements.txt`).

```bash
git clone https://github.com/oskargoluda/ML_stroke_prediction.git
cd ML_stroke_prediction

python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Użycie

### 1. Trening modelu TabNet

Po umieszczeniu datasetu w `data/raw/`:

```bash
python3 src/train.py
```

Skrypt przygotuje dane (zapisze je w `data/processed/`), wytrenuje model (zwykle kilka minut) i zapisze:

- model oraz informacje o preprocessingu w `models/`,
- metryki i wykresy w `results/`.

### 2. Predykcja dla nowych danych

Uzupełnij plik `example_input.csv` (lub przygotuj własny) w formacie:

```csv
gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status
Female,54,1,0,Yes,Private,Urban,171.23,31.4,never smoked
Male,45,0,0,Yes,Self-employed,Rural,98.45,27.8,smokes
```

i uruchom:

```bash
python3 src/infer.py --input example_input.csv
```

Wyniki (prawdopodobieństwo udaru, predykcja i kategoria ryzyka) trafią domyślnie do `predictions.csv`. Dostępne opcje poznasz poleceniem `python3 src/infer.py --help`.

### 3. Szczegółowa ewaluacja

```bash
python3 evaluate.py --csv-path data/raw/healthcare-dataset-stroke-data.csv
```

### 4. Porównanie z regresją logistyczną

Wymaga wcześniej wykonanego treningu (żeby istniały pliki w `data/processed/`):

```bash
python3 compare_logistic.py
```

## Wyniki

Metryki z `results/metrics.json` (zbiór testowy):

| Metryka   | Wartość |
|-----------|---------|
| Accuracy  | 0.768   |
| Precision | 0.115   |
| Recall    | 0.677   |
| F1        | 0.197   |
| ROC-AUC   | 0.824   |

Niska precyzja wynika z silnego niezbalansowania klas: model jest nastawiony na wykrywanie jak największej liczby przypadków udaru kosztem fałszywych alarmów. Wykresy (macierz pomyłek, krzywa ROC, ważność cech, analiza eksploracyjna) znajdziesz w katalogu `results/`, a pełny opis analizy w `notebooks/sprawozdanie.ipynb`.

## Technologie

Python, PyTorch, pytorch-tabnet, scikit-learn, pandas, NumPy, matplotlib, seaborn.
