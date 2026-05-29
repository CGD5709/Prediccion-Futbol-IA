import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve, auc

def walk_forward_split(df, col_temporada='anio_fin_temporada', window_type='expanding', window_size=1):
    """
    Generador para realizar Walk-Forward Validation basado en las temporadas.
    
    Parámetros:
    - df: DataFrame ordenado temporalmente.
    - col_temporada: Nombre de la columna que indica la temporada.
    - window_type: 'expanding' (el entrenamiento crece) o 'sliding' (ventana fija de N temporadas).
    - window_size: Cuántas temporadas pasadas se usan para entrenar si window_type es 'sliding'.
    
    Yields:
    - idx_train, idx_test: Índices para usar con iloc.
    """
    temporadas = sorted(df[col_temporada].unique())
    
    # Necesitamos al menos 2 temporadas (1 train, 1 test mínimo)
    if len(temporadas) < 2:
        raise ValueError("No hay suficientes temporadas para hacer Walk-Forward Validation.")
        
    for i in range(1, len(temporadas)):
        test_season = temporadas[i]
        
        if window_type == 'expanding':
            train_seasons = temporadas[:i]
        elif window_type == 'sliding':
            # Cogemos las últimas 'window_size' temporadas disponibles antes de la de test
            start_idx = max(0, i - window_size)
            train_seasons = temporadas[start_idx:i]
        else:
            raise ValueError("window_type debe ser 'expanding' o 'sliding'")
            
        idx_train = df[df[col_temporada].isin(train_seasons)].index
        idx_test = df[df[col_temporada] == test_season].index
        
        yield idx_train, idx_test

def calculate_metrics(y_true, y_pred, y_proba):
    """
    Calcula las métricas solicitadas en la competición usando implementaciones nativas de scikit-learn.
    
    Parámetros:
    - y_true: array-like con las etiquetas reales (ej: 1, X, 2)
    - y_pred: array-like con las clases predichas.
    - y_proba: array-like con las probabilidades predichas, shape (n_samples, n_classes).
    
    Retorna:
    - f1_mac: F1-score macro.
    - auc_pr_mac: AUC-PR macro (Average Precision).
    """
    from sklearn.metrics import f1_score, average_precision_score
    
    # F1-macro nativo de sklearn
    f1_mac = f1_score(y_true, y_pred, average='macro')
    
    # AUC-PR macro se implementa en sklearn como average_precision_score.
    # Nota: y_true debe ser binarizado internamente por la función,
    # pero requiere que le digamos qué tipo de problema es si las clases no son 0, 1.
    # Una forma segura es convertir y_true a One-Hot o binarizar usando sklearn:
    from sklearn.preprocessing import label_binarize
    clases = np.unique(y_true)
    y_true_bin = label_binarize(y_true, classes=clases)
    
    # Calculamos AUC-PR (Average Precision) macro nativo
    auc_pr_mac = average_precision_score(y_true_bin, y_proba, average='macro')
    
    return f1_mac, auc_pr_mac

