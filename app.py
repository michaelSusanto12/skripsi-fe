from flask import Flask, render_template, request, redirect, flash, jsonify
import matplotlib.pyplot as plt
from io import BytesIO
from base64 import b64encode
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from os import environ
environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.losses import Huber

import locale
locale.setlocale(locale.LC_TIME, 'id_ID')

app = Flask(__name__)
app.secret_key = 'supersecretkey'

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/skripsi_tes'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Sales(db.Model):
    __tablename__ = 'sales'
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.String(7), nullable=False) 
    total = db.Column(db.Float, nullable=False)
    upload_time = db.Column(db.DateTime, nullable=True, default=datetime.utcnow)
    category_id = db.Column(db.Integer, db.ForeignKey('categories.id'), nullable=False)

class Categories(db.Model):
    __tablename__ = 'categories'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)

with app.app_context():
    db.create_all()


def read_and_process_data(category):
    category_obj = Categories.query.filter_by(name=category).first()
    if not category_obj:
        raise ValueError(f"Kategori '{category}' tidak ditemukan.")

    sales_data = Sales.query.filter_by(category_id=category_obj.id).all()
    
    print(f"Raw data fetched from the database for category '{category}':")
    for sale in sales_data:
        print(f"Date: {sale.date}, Total: {sale.total}")
    
    df = pd.DataFrame([{
        'Tanggal': sale.date,
        'Total': sale.total
    } for sale in sales_data])

    print("Initial DataFrame from database data:")
    print(df.head())

    df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='%m-%Y')
    df.set_index('Tanggal', inplace=True)

    Q1 = df['Total'].quantile(0.25)
    Q3 = df['Total'].quantile(0.75)
    IQR = Q3 - Q1

    df['Total'] = np.where((df['Total'] < (Q1 - 1.5 * IQR)) | (df['Total'] > (Q3 + 1.5 * IQR)),
                            np.nan,
                            df['Total'])
    df['Total'] = df['Total'].interpolate(method='linear')
    
    df['Total'].fillna(method='ffill', inplace=True)  
    df['Total'].fillna(method='bfill', inplace=True)
    
    print("Processed DataFrame after handling outliers:")
    print(df.head())
    
    return df

def convert_daily_to_monthly(df):
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    df['YearMonth'] = df['date'].dt.to_period('M')
    df_monthly = df.groupby('YearMonth')['total'].sum().reset_index()
    df_monthly['date'] = df_monthly['YearMonth'].dt.strftime('%m-%Y')
    df_monthly.drop(columns=['YearMonth'], inplace=True)
    return df_monthly

def load_old_data(product_type):
    all_categories = Categories.query.all()
    category = Categories.query.filter_by(name=product_type).first()

    sales_data = Sales.query.filter_by(category_id=category.id).all()

    if not sales_data:
        raise ValueError(f"Tidak ada data penjualan untuk kategori '{product_type}'.")

    df = pd.DataFrame([(sale.date, sale.total) for sale in sales_data], columns=['date', 'total'])

    if df.empty or 'date' not in df.columns:
        raise ValueError("Kolom 'date' tidak ditemukan atau DataFrame kosong.")

    return df

def save_monthly_data(df, category_id):
    print("Saving data to database:")
    print(df.head())  

    for index, row in df.iterrows():
        existing_record = Sales.query.filter_by(category_id=category_id, date=row['date']).first()
        if existing_record:
            existing_record.total = row['total']
        else:
            new_record = Sales(category_id=category_id, date=row['date'], total=row['total'])
            db.session.add(new_record)
    
    db.session.commit()



def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

def record_upload_time(category):
    pass

def create_plot(product_type):
    plt.figure(figsize=(15, 6))

    df_selected = load_old_data(product_type)

    if df_selected.empty or 'date' not in df_selected.columns:
        raise ValueError("Kolom 'date' tidak ditemukan atau DataFrame kosong.")

    df_selected['date'] = pd.to_datetime(df_selected['date'], format='%m-%Y', errors='coerce')
    df_selected = df_selected.sort_values(by='date')

    plt.plot(df_selected['date'], df_selected['total'], marker='o')
    plt.title(f'Penjualan {product_type.replace("_", " ").title()}')
    plt.xlabel('Tanggal')
    plt.ylabel('Jumlah Terjual')

    plt.xticks(
        df_selected['date'],
        df_selected['date'].dt.strftime('%m - %Y'),
        rotation=45,
        ha='right'
    )
    plt.grid(True)
    plt.tight_layout()

    img = BytesIO()
    plt.savefig(img, format='png', dpi=100)
    img.seek(0)
    plt.close()

    plot_url = b64encode(img.getvalue()).decode('utf8')

    start_date = df_selected['date'].min().strftime('%B %Y')
    end_date = df_selected['date'].max().strftime('%B %Y')
    date_range = f'Penjualan per tanggal {start_date} sampai dengan {end_date}'
    
    return plot_url, date_range


@app.route('/')
def index():
    plot_url, date_range = create_plot('lada_halus')
    return render_template('index.html', plot_url=plot_url, date_range=date_range)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        category = request.form.get('category')
        file = request.files['file']

        if category not in ['lada_halus', 'lada_butir']:
            flash('Kategori tidak valid', 'danger')
            return redirect('/upload')
        if not file or not file.filename.endswith('.xlsx'):
            flash('File harus berformat .xlsx', 'danger')
            return redirect('/upload')

        try:
            df_new = pd.read_excel(file)

            print("Data from uploaded Excel:")
            print(df_new.head())  
            print("Column names in uploaded Excel:", df_new.columns.tolist())  

            df_new.columns = df_new.columns.str.strip()

            if 'Tanggal' not in df_new.columns:
                flash("Kolom 'Tanggal' tidak ditemukan di file.", 'danger')
                return redirect('/upload')
            if 'Total' not in df_new.columns:
                flash("Kolom 'Total' tidak ditemukan di file.", 'danger')
                return redirect('/upload')

            df_new.rename(columns={'Tanggal': 'date', 'Total': 'total'}, inplace=True)
            print("Data types of columns:")
            print(df_new.dtypes)

            df_new_monthly = convert_daily_to_monthly(df_new)

            category_obj = Categories.query.filter_by(name=category).first()
            if not category_obj:
                flash(f"Kategori '{category}' tidak ditemukan.", 'danger')
                return redirect('/upload')

            category_id = category_obj.id

            df_old = load_old_data(category)
            
            df_combined = pd.concat([pd.DataFrame(df_old), df_new_monthly]).drop_duplicates(subset=['date'], keep='last')

            with db.session.no_autoflush:
                save_monthly_data(df_combined, category_id)

            flash(f'Data {category.replace("_", " ")} berhasil diunggah dan digabungkan dengan data lama!', 'success')
            return redirect('/upload')
        except Exception as e:
            flash(f'Terjadi kesalahan: {e}', 'danger')
            return redirect('/upload')

    return render_template('upload.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/penjualan', methods=['GET'])
def penjualan():
    product_type = request.args.get('product', 'lada_halus')
    plot_url, date_range = create_plot(product_type)
    return render_template('index.html', plot_url=plot_url, selected_product=product_type, date_range=date_range)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        category = request.form.get('category')
        n_months = int(request.form.get('n_months'))

        try:
            df = read_and_process_data(category)

            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(df['Total'].values.reshape(-1, 1))

            units_1 = 128  
            units_2 = 64
            batch_size = 16
            look_back = 1 

            epoch = 100
            learning_rate = 0.001

            X, y = create_dataset(scaled_data, look_back)

            X = np.reshape(X, (X.shape[0], X.shape[1], 1))

            train_size = int(len(X) * 0.7)
            val_size = int(len(X) * 0.15)
            X_train, y_train = X[:train_size], y[:train_size]
            X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
            X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

            model = Sequential()

            if category == 'lada_butir':
                model.add(LSTM(units_1, return_sequences=True, input_shape=(look_back, 1)))
                model.add(Dropout(0.2))
                model.add(LSTM(units_2)) 
                model.add(Dropout(0.2))
            else:
                model.add(LSTM(units_1, return_sequences=True, input_shape=(look_back, 1)))
                model.add(Dropout(0.2))
                model.add(LSTM(units_2))
                model.add(Dropout(0.2))

            model.add(Dense(1))

            optimizer = Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss=Huber())

            early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

            history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epoch,
                                batch_size=batch_size, verbose=2, callbacks=[early_stopping])

            epochs_run = len(history.history['loss'])

            train_predictions = model.predict(X_train)
            val_predictions = model.predict(X_val)
            test_predictions = model.predict(X_test)

            train_predictions = scaler.inverse_transform(train_predictions)
            val_predictions = scaler.inverse_transform(val_predictions)
            test_predictions = scaler.inverse_transform(test_predictions)

            y_train_rescaled = scaler.inverse_transform(y_train.reshape(-1, 1))
            y_val_rescaled = scaler.inverse_transform(y_val.reshape(-1, 1))
            y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

            train_mae = mean_absolute_error(y_train_rescaled, train_predictions)
            train_mse = mean_squared_error(y_train_rescaled, train_predictions)
            train_rmse = np.sqrt(train_mse)
            train_mape = mean_absolute_percentage_error(y_train_rescaled, train_predictions)
            
            val_mae = mean_absolute_error(y_val_rescaled, val_predictions)
            val_mse = mean_squared_error(y_val_rescaled, val_predictions)
            val_rmse = np.sqrt(val_mse)
            val_mape = mean_absolute_percentage_error(y_val_rescaled, val_predictions)

            test_mae = mean_absolute_error(y_test_rescaled, test_predictions)
            test_mse = mean_squared_error(y_test_rescaled, test_predictions)
            test_rmse = np.sqrt(test_mse)
            test_mape = mean_absolute_percentage_error(y_test_rescaled, test_predictions)

            train_dates = df.index[look_back:train_size + look_back]
            val_dates = df.index[train_size + look_back:train_size + val_size + look_back]
            test_dates = df.index[train_size + val_size + look_back:]

            predicted_df = pd.DataFrame({
                'Tanggal': np.concatenate([train_dates, val_dates, test_dates]),
                'Aktual': np.concatenate([y_train_rescaled.flatten(), y_val_rescaled.flatten(), y_test_rescaled.flatten()]),
                'Prediksi': np.concatenate([train_predictions.flatten(), val_predictions.flatten(), test_predictions.flatten()])
            })

            future_predictions = []
            last_data = scaled_data[-look_back:]
            for _ in range(n_months):
                prediction = model.predict(last_data.reshape(1, look_back, 1))
                future_predictions.append(prediction[0, 0])
                last_data = np.append(last_data[1:], prediction)

            future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

            last_date = df.index[-1]
            future_dates = pd.date_range(last_date, periods=n_months + 1, freq='M')[1:]

            plt.figure(figsize=(15, 6))
            combined_dates = np.concatenate([train_dates, val_dates, test_dates, future_dates])
            combined_actual = np.concatenate([y_train_rescaled.flatten(), y_val_rescaled.flatten(), y_test_rescaled.flatten(), [np.nan] * n_months])
            combined_predictions = np.concatenate([train_predictions.flatten(), val_predictions.flatten(), test_predictions.flatten(), future_predictions.flatten()])
            plt.plot(combined_dates, combined_actual, label='Aktual', color='black', linestyle='-', marker='o')
            plt.plot(combined_dates, combined_predictions, label='Prediksi', color='orange', linestyle='--', marker='x')
            plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
            plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%Y'))
            plt.xlabel('Tanggal')
            plt.ylabel('Total')
            plt.grid(True)
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            img = BytesIO()
            plt.savefig(img, format='png', dpi=100)
            img.seek(0)
            plt.close()

            plot_url = b64encode(img.getvalue()).decode('utf8')

            return jsonify({
                'plot_url': plot_url,
                'train_mae': train_mae,
                'train_mse': train_mse,
                'train_rmse': train_rmse,
                'train_mape': train_mape,
                'val_mae': val_mae,
                'val_mse': val_mse,
                'val_rmse': val_rmse,
                'val_mape': val_mape,
                'test_mae': test_mae,
                'test_mse': test_mse,
                'test_rmse': test_rmse,
                'test_mape': test_mape,
                'future_predictions': [{'Tanggal': str(date), 'Prediksi': float(pred)} for date, pred in zip(future_dates, future_predictions)]
            })

        except Exception as e:
            print(f'An error occurred: {e}')  
            flash(f'An error occurred: {e}', 'danger')
            return redirect('/')

    return render_template('predict.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')