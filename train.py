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
    print("Processando dados...")
    df = prepare_pipeline(config.DATA_PATH, config.OUTLIER_THRESHOLD)

    X = df.drop(columns=[config.TARGET_COL])
    y = df[config.TARGET_COL].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_STATE
    )


    models = get_models()
    
    print("Iniciando treinamento dos modelos...\n")
    for name, model in models.items():
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