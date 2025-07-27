import pandas as pd
import logging
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import joblib

class Model:
    def __init__(self, file_path):
        self.file = file_path
        self.df = None
    
    def load_data(self):
        try:
            self.df = pd.read_csv(self.file)
        except FileNotFoundError:
            logging.error(f"File not found: {self.file}")
            raise

    def preprocess_data(self):
        if self.df is None:
            logging.error("Data not loaded. Call load_data() first.")
            return
        
        target_column = 'Churn'
        X = self.df.drop(columns=target_column)
        y = self.df[target_column]
        
        # Separate columns by type
        numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categoric_columns = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        # Define preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('numerical', StandardScaler(), numeric_columns),
                ('categorical', OneHotEncoder(drop='first', handle_unknown='ignore'), categoric_columns)
            ]
        )
        data_path = 'data/processed/preprocessed_data.pkl'
        X_transformed = preprocessor.fit_transform(X)
        joblib.dump((preprocessor), data_path)
        logging.info(f"Preprocessed data saved to {data_path}")
        return X_transformed, y
    
    
    def use_model_logistic_regression(self, x, y):
        if x is None or y is None:
            logging.error("Data not preprocessed. Call preprocess_data() first.")
            return
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        model = LogisticRegression()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)
        logging.info(f"Confusion Matrix:\n{cm}")
        logging.info(f"Classification Report:\n{cr}")
        return model

    def save_model(self, model):
        if model is None:
            logging.error("Model not trained. Call use_model_logistic_regression() first.")
            return
        model_path = 'models/logistic_regression_model.pkl'
        joblib.dump(model, model_path)
        logging.info(f"Model saved to {model_path}")


    def main(self):
        self.load_data()
        x, y = self.preprocess_data()
        model = self.use_model_logistic_regression(x, y)
        self.save_model(model)
        
if __name__ == "__main__":
    file_path = 'data/cleaned/cleaned_telco_churn.csv'
    model = Model(file_path)
    model.main()
