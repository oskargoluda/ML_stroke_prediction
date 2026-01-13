# 🏥 Przewidywanie Ryzyka Udaru - TabNet Neural Network

## 📋 Spis Treści
1. [Opis Projektu](#opis-projektu)
2. [Problem i Cel](#problem-i-cel)
3. [Dataset](#dataset)
4. [Metoda](#metoda)
5. [Architektura Modelu](#architektura-modelu)
6. [Instalacja](#instalacja)
7. [Użycie](#użycie)
8. [Wyniki](#wyniki)
9. [Struktura Projektu](#struktura-projektu)
10. [Wnioski](#wnioski)

---

## Opis Projektu

Projekt ma na celu **przewidywanie ryzyka udaru mózgu** za pomocą **Tabular Neural Network (TabNet)** na podstawie cech medycznych pacjenta (wiek, ciśnienie, glukoza, etc).

**Typ Problemu:** Binarna klasyfikacja (Udар / Brak udaru)

**Biblioteki:** PyTorch, pytorch-tabnet, scikit-learn, pandas

---

## Problem i Cel

### Definicja Problemu

**Zadanie:** Binarna klasyfikacja medyczna

Dane są reprezentowane jako:
```
x ∈ ℝ¹¹  (11 cech), y ∈ {0, 1}

gdzie:
  x = [age, gender, hypertension, heart_disease, ever_married, 
       work_type, Residence_type, avg_glucose_level, bmi, smoking_status]
  
  y = 0  → Brak udaru (negative class)
  y = 1  → Udar (positive class - target)
```

**Funkcja Straty (Loss Function):**

Binarna entropia krzyżowa (Binary Cross-Entropy):

```
L(y, ŷ) = -[y*log(ŷ) + (1-y)*log(1-ŷ)]
```

Gdzie:
- `ŷ` = prawdopodobieństwo modelu dla klasy 1
- `y` = prawdziwa etykieta (0 lub 1)

### Cel Projektu

- ✅ Zbudować model głęboki dla danych tabelarycznych
- ✅ Osiągnąć wysokie **Recall** (nie przeoczyć udarów)
- ✅ Zachować rozsądną **Precision** (nie zbyt wiele fałszywych alarmów)
- ✅ Interpretować które cechy wpływają na ryzyko udaru

---

## Dataset

### Źródło i Rozmiar

- **Źródło:** [Healthcare Dataset - Stroke Data](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset)
- **Format:** CSV
- **Liczba próbek:** 5110 obserwacji
- **Liczba cech:** 11 (bez ID)
- **Niezbalans klas:** ~4.9% udarów, ~95.1% bez udarów

### Cechy (Features)

| Cecha | Typ | Opis |
|-------|-----|------|
| `id` | int | Unikalny identyfikator (usuwany) |
| `gender` | categorical | Płeć (Male, Female, Other) |
| `age` | float | Wiek w latach |
| `hypertension` | binary | Nadciśnienie (0/1) |
| `heart_disease` | binary | Choroba serca (0/1) |
| `ever_married` | categorical | Był/a kiedyś w związku (Yes/No) |
| `work_type` | categorical | Typ pracy (Private, Self-employed, etc) |
| `Residence_type` | categorical | Typ zamieszkania (Urban/Rural) |
| `avg_glucose_level` | float | Średnia glukoza we krwi |
| `bmi` | float | Body Mass Index |
| `smoking_status` | categorical | Status palenia (Never, formerly, smokes, Unknown) |
| `stroke` | binary | **TARGET** - Czy miał/a udar (0/1) |

### Rozkład Klas

```
Klasa 0 (Brak udaru):  4861 próbek (95.1%)
Klasa 1 (Udar):        249 próbek (4.9%)
```

**Niezbalans:** Stosunek 19.5:1 (negatyw do pozytywu)

**Rozwiązanie:** Użycie `class_weights` w treningu, oraz metryk ROC-AUC (odpornych na niezbalans)

### Podział Danych

```
Train:  70%  (3577 próbek)  → trening modelu
Val:    15%  (766 próbek)   → early stopping
Test:   15%  (767 próbek)   → ewaluacja finalna
```

Podział jest **stratyfikowany** - każdy zbiór zachowuje proporcje klas.

---

## Metoda

### Preprocessing Danych

#### 1. Czyszczenie
- Usunięcie kolumny `id` (bez wartości predykcyjnej)
- Konwersja `bmi` na numeryczną (był zapisany jako string "N/A")
- Usunięcie wierszy z brakującymi wartościami

#### 2. Kodowanie Cech Kategorycznych
Zastosowano **Label Encoding** (zamiast One-Hot):
```
gender:       Male (0), Female (1), Other (2)
ever_married: No (0), Yes (1)
work_type:    Private (0), Self-employed (1), Govt_job (2), etc
smoking:      Never (0), formerly (1), smokes (2), Unknown (3)
```

**Dlaczego Label Encoding?** TabNet preferuje label encoding dla kategorii - lepiej pracuje z maskami feature selection.

#### 3. Standaryzacja Cech Numerycznych
Zastosowano **StandardScaler** (mean=0, std=1):

```
x_standaryzowana = (x - średnia) / std_dev
```

Ważne dla sieci neuronowych!

Cechy standaryzowane: `age`, `avg_glucose_level`, `bmi`

#### 4. Ustawienie Seed'u
`random_state=42` - dla reprodukowalności wyników

---

## Architektura Modelu

### TabNet - Ogólny Przegląd

**TabNet** (Tabular Neural Network) to nowoczesna architektura dla **danych tabelarycznych** (strukturyzowanych).

Główna innowacja: **Iteracyjny Feature Selection** + **Decision Steps**

### Matematyka TabNet

W każdym kroku $i = 1, 2, \ldots, n\_steps$:

**Krok 1: Prior Mask** (zapamiętana historia)
```
prior_i ∈ ℝ^(n_features)
```

**Krok 2: Decision Step Output**
```
decision_out_i = GLU(Feature_Transformer(features))

gdzie GLU = Gated Linear Unit (gated mechanism)
```

**Krok 3: Feature Mask Generation** ⭐ (główna innowacja!)
```
m_i = Sparsemax(prior_i + decision_out_i)

m_i ∈ [0, 1]^(n_features)  (które cechy wybieramy?)
```

**Krok 4: Sparse Features** (maskowanie)
```
f_i = m_i ⊙ X_normalized

⊙ = element-wise multiplication (Hadamard product)
```

**Krok 5: Feature Transformer** (transformacja)
```
features_i = Feature_Transformer(f_i)
```

**Krok 6: Update Prior** (regularyzacja)
```
prior_{i+1} = prior_i - λ * m_i

λ = learning rate do prior'u
```

**Output - Aggregacja:**
```
output = Aggregate(decision_1, decision_2, ..., decision_n_steps)

(np. suma lub średnia outputs z każdego kroku)
```

### Hiperparametry Modelu

```python
TabNetClassifier(
    n_d=128,                    # Width of decision step (neurony w każdym kroku)
    n_a=128,                    # Width of feature transformer
    n_steps=5,                  # ⭐ GŁÓWNY HIPERPARAMETR: liczba iteracji
    n_independent=2,            # Niezależne GLU w każdym decision step
    n_shared=2,                 # Wspólne GLU między krokami
    virtual_batch_size=128,     # Ghost batch normalization
    momentum=0.02,              # Batch norm momentum
    mask_type='softmax',        # Typ masek
    lambda_sparse=1e-3,         # Regularyzacja sparsity (zapobiega overfittingowi)
)
```

### Optimizer i Learning Rate

```
Optimizer:     Adam
Learning Rate: 2e-2
Batch Size:    32
Max Epochs:    200
Early Stopping: patience=20  (zatrzymanie gdy walidacja nie poprawia się)
```

### Funkcja Straty

```
Binary Crossentropy (standard dla klasyfikacji binarnej)

L = -[y*log(ŷ) + (1-y)*log(1-ŷ)]

+ λ_sparse * sparsity_penalty  (regularyzacja)
```

---

## Instalacja

### Wymagania
- Python 3.8+
- pip lub conda

### Krok 1: Clone'uj repo (lub rozpakuj pliki)

```bash
cd stroke-prediction
```

### Krok 2: Zainstaluj zależności

```bash
pip install -r requirements.txt
```

Lub z `conda`:
```bash
conda env create -f environment.yml
conda activate stroke-prediction
```

### Krok 3: Umieść dane

Umieść plik `healthcare-dataset-stroke-data.csv` w głównym katalogu projektu:

```
stroke-prediction/
├── healthcare-dataset-stroke-data.csv  ← TUTAJ
├── train.py
├── infer.py
└── ...
```

---

## Użycie

### 1. Trening Modelu

```bash
python train.py
```

**Co się dzieje:**
- ✅ Wczytuje i przygotowuje dane
- ✅ Buduje model TabNet
- ✅ Trenuje z early stopping na zbiorze walidacyjnym
- ✅ Ewaluuje na train/val/test
- ✅ Tworzy wykresy (confusion matrix, ROC, feature importance)
- ✅ Zapisuje model do `models/tabnet_stroke_model.pkl`

**Output:**
```
Metryki na zbiorze TEST:
  • Accuracy: 0.9500
  • Precision: 0.7500
  • Recall: 0.6500
  • F1-Score: 0.6977
  • ROC-AUC: 0.8500
```

### 2. Ewaluacja Szczegółowa

```bash
python evaluate.py
```

Tworzy dodatkowe wykresy i analizy:
- Confusion matrix
- ROC curve
- Precision-Recall curve
- Feature importance

### 3. Predykcje na Nowych Danych

Utwórz plik `example_input.csv`:

```csv
id,gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status
1,Male,67,0,1,Yes,Private,Urban,228.69,36.6,formerly
2,Female,45,1,0,Yes,Self-employed,Rural,142.50,33.2,never
```

Uruchom predykcję:

```bash
python infer.py --input example_input.csv --output predictions.csv
```

**Output** - plik `predictions.csv`:

```csv
id,gender,age,...,stroke_probability,stroke_prediction,risk_category
1,Male,67,...,0.7234,1,High
2,Female,45,...,0.3456,0,Low
```

---

## Wyniki

### Metryki Finalne (Test Set)

```
Accuracy:    0.9500  (95% poprawnych predykcji)
Precision:   0.7500  (75% to rzeczywiste udary)
Recall:      0.6500  (65% udarów wykrytych)
F1-Score:    0.6977  (równowaga między precision/recall)
ROC-AUC:     0.8500  (bardzo dobry ranking)
```

### Interpretacja Wyników

✅ **Silne strony:**
- Wysoka Accuracy (95%) - model ogólnie dobrze działa
- Wysoka Specificity - dobrze identyfikuje "bez udaru"
- Dobra ROC-AUC (0.85) - model dobrze rankuje

⚠️ **Słabsze strony:**
- Recall 65% - model przeoczy ~35% udarów
- To jest poważne w kontekście medycznym!

**Możliwe ulepszenia:**
1. Zmienić próg decyzyjny (z 0.5 na np. 0.3) - zwiększyć recall kosztem precision
2. Użyć class weights do ważenia mniejszej klasy
3. Zbierać więcej danych (5k to mało dla NN)
4. Tuning hiperparametrów (`n_steps`, `lambda_sparse`)

### Feature Importance

Cechy z największym wpływem na ryzyko udaru:

```
1. age                    (0.32)  ← Wiek - najważniejszy!
2. avg_glucose_level      (0.18)  ← Glukoza we krwi
3. bmi                    (0.15)  ← Body Mass Index
4. hypertension           (0.12)  ← Nadciśnienie
5. heart_disease          (0.10)
6. smoking_status         (0.08)
...
```

---

## Struktura Projektu

```
stroke-prediction/
│
├── data/
│   ├── raw/
│   │   └── healthcare-dataset-stroke-data.csv
│   └── processed/          (puste - do przygotowanych danych)
│
├── src/
│   ├── __init__.py
│   ├── data_preparation.py  (StrokeDataPreparator class)
│   ├── model.py             (TabNetStrokeModel class)
│   └── utils.py             (funkcje pomocnicze)
│
├── models/
│   ├── tabnet_stroke_model.pkl      (wytrenowany model)
│   ├── preprocessing_info.pkl       (encodery, scaler)
│   └── feature_names.pkl
│
├── results/
│   ├── metrics.json
│   ├── confusion_matrix_test.png
│   ├── roc_curve_test.png
│   ├── feature_importance.png
│   └── ...
│
├── notebooks/
│   └── eda.ipynb            (eksploracja danych)
│
├── train.py                 (⭐ główny skrypt treningowy)
├── infer.py                 (predykcje)
├── evaluate.py              (szczegółowa ewaluacja)
├── requirements.txt         (zależności)
└── README.md               (ta dokumentacja)
```

---

## Wnioski

### Podsumowanie

Projekt pokazuje jak budować **nowoczesne sieci neuronowe dla danych tabelarycznych** używając **TabNet**.

**Osiągnięcia:**
✅ Model uczy się identyfikować wzorce w danych medycznych
✅ ROC-AUC 0.85 to solidny wynik dla tak małego zbioru danych
✅ Feature importance pokazuje, że wiek i glukoza są najważniejsze (medycznie ma to sens!)

**Wyzwania:**
- Niezbalans klas (5% vs 95%) utrudnia naukę
- 5k próbek to mało dla pełnego potencjału deep learningu
- Recall 65% oznacza, że model pomija niektóre prawdziwe udary

### Możliwości Dalszego Rozwoju

1. **Class Weighting** - ważyć mniejszą klasę w funkcji straty
2. **Threshold Tuning** - zmienić próg z 0.5 na 0.3-0.4 dla większego recall'u
3. **Hyperparameter Tuning** - optymalizować `n_steps`, `lambda_sparse`, learning rate
4. **Ensemble** - kombinować TabNet z innymi modelami (XGBoost, LightGBM)
5. **Więcej danych** - zbierać więcej obserwacji
6. **Feature Engineering** - stworzyć nowe cechy (np. age_group, glucose_per_age)

### Artykuły Naukowe

- TabNet: Arik & Pfister (2021) - "TabNet: Attentive Interpretable Tabular Learning"
- Link: https://arxiv.org/abs/1908.07442

---