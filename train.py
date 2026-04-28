# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error as mae

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from xgboost import XGBRegressor

import config
from preprocess import prepare_pipeline

def get_models():
    """Retorna um dicionário com os modelos instanciados."""
    return {
        "LinearRegression": LinearRegression(),
        "XGBRegressor": XGBRegressor(),
        "Lasso": Lasso(),
        "Ridge": Ridge()
    }

def main():
    # 1. Execução do Pré-processamento
    print("Processando dados...")
    df = prepare_pipeline(config.DATA_PATH, config.OUTLIER_THRESHOLD)

    # 2. Separação de Features e Target
    X = df.drop(columns=[config.TARGET_COL])
    y = df[config.TARGET_COL].values

    # 3. Split de Treino e Validação
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_STATE
    )

    # 4. Treinamento e Avaliação via Pipeline
    models = get_models()
    
    print("Iniciando treinamento dos modelos...\n")
    for name, model in models.items():
        # O Pipeline garante que a normalização fit_transform ocorra apenas no treino
        # e o transform ocorra automaticamente na validação.
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', model)
        ])

        pipeline.fit(X_train, y_train)

        train_preds = pipeline.predict(X_train)
        val_preds = pipeline.predict(X_val)

        print(f"--- {name} ---")
        print(f"Training Error (MAE): {mae(y_train, train_preds):.4f}")
        print(f"Validation Error (MAE): {mae(y_val, val_preds):.4f}\n")

if __name__ == "__main__":
    main()