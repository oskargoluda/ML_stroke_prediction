"""
Skrypt do treningu regresji liniowej na tych samych danych co TabNet, co pozwoli na ocene metody wydajności w porównaniu z prostszym modelem.
"""




import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import sys

def main():
    print("Regresja Logistyczna - Porównanie z TabNet\n")

    #odczyt danych
    try:
        print("1. Wczytywanie danych z 'data/processed/'...")
        X_train = pd.read_csv('data/processed/X_train.csv')
        y_train = pd.read_csv('data/processed/y_train.csv').values.ravel() # ravel spłaszcza wektor
        X_test = pd.read_csv('data/processed/X_test.csv')
        y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
        print("Dane wczytane pomyślnie.")
    except FileNotFoundError:
        print("Nie znaleziono danych z plików")
        sys.exit(1)

    # Trening modelu
    print("\n2. Trenowanie modelu Regresji Logistycznej")
    # class_weight='balanced' zmusza model do zwracania wiekszej uwagi na mniejszą klasę 1 czyli udar
    model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)
    print("Trening zakończony.")


    print("\n3. Obliczanie metryk")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Obliczenie metryk
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    # Macierz pomyłek
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()


    print("\n4. Wyniki Regresji Logistycznej:")
    print(f"Accuracy:   {acc:.4f}")
    print(f"Precision:  {prec:.4f}")
    print(f"Recall:     {rec:.4f}")
    print(f"F1-Score:   {f1:.4f}")
    print(f"ROC-AUC:    {auc:.4f}")
    
    print("\nDane z confusion matrix:")
    print(f"Poprawnie wykryte udary: {tp}")
    print(f"Przeoczone udary:        {fn}")
    print(f"Fałszywe alarmy:         {fp}")

    print("\nRaport klasyfikacji:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
