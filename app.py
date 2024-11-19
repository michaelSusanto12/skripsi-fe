from flask import Flask, render_template, request, redirect, flash, jsonify, url_for, session
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
from flask_bcrypt import Bcrypt
from flask_migrate import Migrate
from flask_login import login_user, current_user, LoginManager, UserMixin, login_required, logout_user
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from admin import admin_bp

import locale
locale.setlocale(locale.LC_TIME, 'id_ID')

app = Flask(__name__)
app.secret_key = 'supersecretkey'
login_manager = LoginManager(app)
login_manager.login_view = 'login'
app.register_blueprint(admin_bp, url_prefix='/admin')

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/skripsi_tes'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
migrate = Migrate(app, db)

class User(db.Model, UserMixin):
    __tablename__ = 'user'  
    id = db.Column(db.Integer, primary_key=True, autoincrement=True) 
    email = db.Column(db.String(120), unique=True, nullable=False)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    is_approved = db.Column(db.Boolean, default=False) 
    role = db.Column(db.String(20), default='user') 

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class Categories(db.Model):
    __tablename__ = 'categories'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    sales = db.relationship('Sales', backref='category', lazy=True) 

class Sales(db.Model):
    __tablename__ = 'sales'
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.String(7), nullable=False)  
    total = db.Column(db.Float, nullable=False)
    upload_time = db.Column(db.DateTime, nullable=True)
    category_id = db.Column(db.Integer, db.ForeignKey('categories.id'), nullable=False)  

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

    df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='%m-%Y', errors='coerce')
    
    df.dropna(subset=['Tanggal'], inplace=True)
    
    df = df.sort_values(by='Tanggal', ascending=True)
    
    df.set_index('Tanggal', inplace=True)

    Q1 = df['Total'].quantile(0.25)
    Q3 = df['Total'].quantile(0.75)
    IQR = Q3 - Q1

    df['Total'] = np.where(
        (df['Total'] < (Q1 - 1.5 * IQR)) | (df['Total'] > (Q3 + 1.5 * IQR)),
        np.nan,
        df['Total']
    )
    
    df['Total'] = df['Total'].interpolate(method='linear')

    df['Total'].fillna(method='ffill', inplace=True)
    df['Total'].fillna(method='bfill', inplace=True)

    print("Processed DataFrame after handling outliers and sorting:")
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
    
    upload_timestamp = datetime.now()

    for index, row in df.iterrows():
        existing_record = Sales.query.filter_by(category_id=category_id, date=row['date']).first()
        if existing_record:
            existing_record.total = row['total']
            existing_record.upload_time = upload_timestamp  
        else:

            new_record = Sales(
                category_id=category_id,
                date=row['date'],
                total=row['total'],
                upload_time=upload_timestamp  
            )
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

def create_plot(product_type, year=None):
    plt.figure(figsize=(15, 6))

    df_selected = load_old_data(product_type)

    if df_selected.empty or 'date' not in df_selected.columns:
        raise ValueError("Kolom 'date' tidak ditemukan atau DataFrame kosong.")

    df_selected['date'] = pd.to_datetime(df_selected['date'], format='%m-%Y', errors='coerce')
    df_selected = df_selected.sort_values(by='date')

    available_years = df_selected['date'].dt.year.dropna().unique()

    if year:
        df_selected = df_selected[df_selected['date'].dt.year == int(year)]

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

    if not df_selected.empty:
        start_date = df_selected['date'].min().strftime('%B %Y')
        end_date = df_selected['date'].max().strftime('%B %Y')
        date_range = f'Penjualan per tanggal {start_date} sampai dengan {end_date}'
    else:
        date_range = "Tidak ada data untuk tahun yang dipilih."

    return plot_url, date_range, available_years

@app.route('/')
def home():
    if 'user_id' in session: 
        return redirect(url_for('beranda')) 
    return redirect(url_for('login'))  



@app.route('/beranda')
def beranda():
    product_type = request.args.get('product', 'lada_halus')  
    selected_year = request.args.get('year') 

    plot_url, date_range, available_years = create_plot(product_type, selected_year)

    return render_template(
        'index.html',
        plot_url=plot_url,
        date_range=date_range,
        available_years=available_years,
        selected_product=product_type,
        selected_year=selected_year
    )
    

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        username = request.form['username']
        password = request.form['password']
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')  

        user_exists = User.query.filter_by(username=username).first()
        if user_exists:
            flash('Username already exists, try another one!', 'danger')
            return redirect(url_for('register'))

        user_exists_email = User.query.filter_by(email=email).first()
        if user_exists_email:
            flash('Email is already registered, try another one!', 'danger')
            return redirect(url_for('register'))

        new_user = User(email=email, username=username, password=hashed_password, is_approved=False, role='user')
        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful, please wait for admin approval!', 'info')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()

        if user and bcrypt.check_password_hash(user.password, password): 
            if not user.is_approved:
                flash('Your account is awaiting admin approval.', 'warning')
                return redirect(url_for('login'))

            session['user_id'] = user.id 
            login_user(user)
            flash('Login successful!', 'success')

            if user.role == 'admin':
                return redirect(url_for('admin_dashboard'))  
            else:
                return redirect(url_for('beranda'))  

        else:
            flash('Username or password is incorrect.', 'danger')

    return render_template('login.html')


@app.route('/logout', methods=['POST'])
@login_required
def logout():
    role = current_user.role
    logout_user()
    session.clear()
    
    if role == 'admin':
        flash('Admin logged out successfully.', 'success')
        return redirect(url_for('admin_login'))
    elif role == 'user':
        flash('Anda sudah logout.', 'success')
        return redirect(url_for('login'))
    
    return redirect(url_for('login'))


@app.route('/admin/dashboard')
@login_required
def admin_dashboard():
    if not current_user.is_authenticated or current_user.role != 'admin':
        flash('Access denied. Admin access only.', 'danger')
        return redirect(url_for('admin_login'))

    approved_users = User.query.filter_by(is_approved=True).all()
    pending_users = User.query.filter_by(is_approved=False).all()
    approved_count = len(approved_users)
    pending_count = len(pending_users)

    return render_template('admin_dashboard.html', 
                           approved_users=approved_users, 
                           pending_users=pending_users,
                           approved_count=approved_count,
                           pending_count=pending_count)


@app.route('/admin/dashboard_data', methods=['GET'])
@login_required
def get_dashboard_data():
    if not current_user.is_authenticated or current_user.role != 'admin':
        return jsonify({'error': 'Access denied'}), 403

    approved_users = User.query.filter_by(is_approved=True).all()
    pending_users = User.query.filter_by(is_approved=False).all()
    approved_count = len(approved_users)
    pending_count = len(pending_users)

    data = {
        'approved_count': approved_count,
        'pending_count': pending_count,
        'approved_users': [{'id': u.id, 'username': u.username, 'email': u.email, 'role': u.role} for u in approved_users],
        'pending_users': [{'id': u.id, 'username': u.username, 'email': u.email, 'role': u.role} for u in pending_users]
    }

    return jsonify(data)
    
    
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if current_user.is_authenticated:
        if current_user.role == 'admin':
            return redirect(url_for('admin_dashboard')) 
        else:
            flash('You are already logged in as a user!', 'warning')
            return render_template('index.html') 

    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        user = User.query.filter_by(email=email).first()

        if user and user.role == 'admin' and bcrypt.check_password_hash(user.password, password):
            login_user(user) 
            flash('Admin logged in successfully!', 'success')
            return redirect(url_for('admin_dashboard')) 
        else:
            flash('Invalid credentials, please try again.', 'danger') 

    return render_template('admin_login.html')


@app.route('/admin/register', methods=['GET', 'POST'])
@login_required  
def admin_register():
    if not current_user.is_authenticated or current_user.role != 'admin':
        flash('Access denied. Only admins can register new admins.', 'danger')
        return redirect(url_for('index'))  

    if request.method == 'POST':
        email = request.form['email']
        username = request.form['username']
        password = request.form['password']
        
        if not email or not username or not password:
            flash('All fields are required!', 'danger')
            return redirect(url_for('admin_register'))

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')  

        try:
            existing_user = User.query.filter((User.username == username) | (User.email == email)).first()
            if existing_user:
                flash('Username or email already exists. Please use a different one.', 'danger')
                return redirect(url_for('admin_register'))

            new_admin = User(email=email, username=username, password=hashed_password, is_approved=True, role='admin')
            db.session.add(new_admin)
            db.session.commit()

            flash('New admin registered successfully!', 'success')
            return redirect(url_for('admin_dashboard')) 

        except Exception as e:
            db.session.rollback() 
            flash('An error occurred while registering the new admin. Please try again.', 'danger')
            print(f"Error: {e}") 

    return render_template('admin_register.html')

@app.route('/admin/approve_user/<int:user_id>', methods=['POST'])
@login_required
def approve_user(user_id):
    if not current_user.is_authenticated or current_user.role != 'admin':
        flash('Access denied. You are not an admin.', 'danger')
        return redirect(url_for('login'))

    user = User.query.get(user_id)
    if not user:
        flash('User not found.', 'danger')
        return redirect(url_for('admin_dashboard'))

    try:
        user.is_approved = True
        db.session.commit()
        flash(f'User {user.username} has been approved.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error approving user: {str(e)}', 'danger')

    return redirect(url_for('admin_dashboard'))


@app.route('/admin/delete_user/<int:user_id>', methods=['POST'])
@login_required
def delete_user(user_id):
    print(f"DEBUG: Deleting user with ID: {user_id}")
    user = User.query.get(user_id)
    if not user:
        flash('User not found.', 'danger')
        return redirect(url_for('admin_dashboard'))
    try:
        db.session.delete(user)
        db.session.commit()
        flash(f'User {user.username} has been deleted.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting user: {str(e)}', 'danger')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/delete_approved_user/<int:user_id>', methods=['POST'])
@login_required
def delete_approved_user(user_id):
    if current_user.role != 'admin':
        flash('Access denied. You are not an admin.', 'danger')
        return redirect(url_for('login'))

    user = User.query.get(user_id)
    if user:
        db.session.delete(user)
        db.session.commit()
        flash('User deleted successfully.', 'success')
    else:
        flash('User not found!', 'danger')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/admin_list_data', methods=['GET', 'POST'])
def admin_list_data():
    category = request.args.get('category', 'lada_halus')  
    items_per_page = request.args.get('items_per_page', '5') 
    page = int(request.args.get('page', 1))  

    if items_per_page == 'all':
        items_per_page = None
    else:
        items_per_page = int(items_per_page)

    query = Sales.query

    if category:
        query = query.filter(Sales.category.has(name=category))

    sales_data = query.all()

    print("Raw sales data:")
    for sale in sales_data:
        print(sale.date)

    sales_data = sorted(sales_data, key=lambda sale: datetime.strptime(sale.date, '%m-%Y'))

    print("Sorted sales data:")
    for sale in sales_data:
        print(sale.date)

    total_sales = len(sales_data)
    total_pages = (total_sales + (items_per_page - 1)) // items_per_page if items_per_page else 1

    if items_per_page:
        start = (page - 1) * items_per_page
        end = start + items_per_page
        sales_data = sales_data[start:end]

    for sale in sales_data:
        sale.month_year = datetime.strptime(sale.date, '%m-%Y').strftime('%B %Y') 

    if request.method == 'POST':
        action = request.form.get('action') 
        sale_id = request.form.get('sale_id')  

        if action == 'delete' and sale_id:
            sale = Sales.query.get(sale_id)
            if sale:
                db.session.delete(sale)
                db.session.commit()
                flash('Data berhasil dihapus', 'success')
            else:
                flash('Data tidak ditemukan', 'danger')
            return redirect(url_for('admin_list_data', category=category, page=page, items_per_page=items_per_page))

        elif action == 'edit' and sale_id:
            sale = Sales.query.get(sale_id)
            if sale:
                new_total = request.form.get('total') 
                try:
                    sale.total = float(new_total)  
                    db.session.commit()
                    flash('Data berhasil diperbarui', 'success')
                except ValueError:
                    flash('Total harus berupa angka yang valid', 'danger')
            else:
                flash('Data tidak ditemukan', 'danger')
            return redirect(url_for('admin_list_data', category=category, page=page, items_per_page=items_per_page))

    return render_template(
        'admin_list_data.html',
        sales_data=sales_data,
        category=category,
        items_per_page=items_per_page,
        page=page,
        total_pages=total_pages
    )

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
            df_new.columns = df_new.columns.str.strip()

            if 'Tanggal' not in df_new.columns or 'Total' not in df_new.columns:
                flash("Kolom 'Tanggal' dan 'Total' harus ada di file.", 'danger')
                return redirect('/upload')

            df_new.rename(columns={'Tanggal': 'date', 'Total': 'total'}, inplace=True)
            df_new_monthly = convert_daily_to_monthly(df_new)

            category_obj = Categories.query.filter_by(name=category).first()
            if not category_obj:
                flash(f"Kategori '{category}' tidak ditemukan.", 'danger')
                return redirect('/upload')
            category_id = category_obj.id

            df_old = load_old_data(category)
            
            duplicate_dates = df_new_monthly[df_new_monthly['date'].isin(df_old['date'])]
            if not duplicate_dates.empty:
                for _, row in duplicate_dates.iterrows():
                    old_data = df_old[df_old['date'] == row['date']]
                    if not old_data.empty:
                        old_data_index = old_data.index[0]
                        df_old.at[old_data_index, 'total'] = row['total'] 
                        flash(f"Data per tanggal {row['date']} telah diganti dengan data yang baru.", 'info')

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
            batch_size = 32
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

            train_predictions = scaler.inverse_transform(model.predict(X_train))
            val_predictions = scaler.inverse_transform(model.predict(X_val))
            test_predictions = scaler.inverse_transform(model.predict(X_test))

            y_train_rescaled = scaler.inverse_transform(y_train.reshape(-1, 1))
            y_val_rescaled = scaler.inverse_transform(y_val.reshape(-1, 1))
            y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

            train_dates = df.index[look_back:train_size + look_back]
            val_dates = df.index[train_size + look_back:train_size + val_size + look_back]
            test_dates = df.index[train_size + val_size + look_back:]

            last_data_point = scaled_data[-look_back:].reshape(1, look_back, 1)
            future_predictions = []
            for i in range(n_months):
                future_pred = model.predict(last_data_point)
                print(f'Future prediction {i+1}: {future_pred[0][0]}')  
                future_predictions.append(future_pred[0][0])
                last_data_point = np.roll(last_data_point, -1, axis=1)  
                last_data_point[0, -1, 0] = future_pred[0, 0]  

            future_predictions_rescaled = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
            future_predictions_rescaled = future_predictions_rescaled.flatten()

            future_dates_start = df.index[-1] + pd.DateOffset(months=1)
            future_dates = pd.date_range(start=future_dates_start.replace(day=1), periods=n_months, freq='MS')

            print(f'Last date in the dataset: {df.index[-1]}')
            print(f'Future dates: {future_dates}')
            print(f'Future predictions: {future_predictions_rescaled}')
            
            print(f'Future Dates shape: {future_dates.shape}')
            print(f'Future Predictions shape: {future_predictions_rescaled.shape}')
            print(f'prediksi di tanggal {future_dates[0]} adalah {future_predictions_rescaled[0]}')
            print(future_dates.min())

            fig, axs = plt.subplots(3, 1, figsize=(15, 18))
        
            axs[0].plot(train_dates, y_train_rescaled, label='Aktual', color='black', linestyle='-', marker='o')
            axs[0].plot(train_dates, train_predictions, label='Prediksi', color='orange', linestyle='--', marker='x')
            axs[0].set_title('Data Training')
            axs[0].set_xlabel('Tanggal')
            axs[0].set_ylabel('Total')
            axs[0].legend()
            axs[0].grid(True)

            axs[1].plot(val_dates, y_val_rescaled, label='Aktual', color='black', linestyle='-', marker='o')
            axs[1].plot(val_dates, val_predictions, label='Prediksi', color='orange', linestyle='--', marker='x')
            axs[1].set_title('Data Validation')
            axs[1].set_xlabel('Tanggal')
            axs[1].set_ylabel('Total')
            axs[1].legend()
            axs[1].grid(True)


            extended_dates = np.concatenate([test_dates, future_dates])
            extended_predictions = np.concatenate([test_predictions.flatten(), future_predictions_rescaled.flatten()])
            
            print(f"future_predictions_rescaled: {future_predictions_rescaled}")
            print(f"test dates : {test_dates}")
            print(f"test predictions : {test_predictions.flatten()}")
            print(f"test values : {y_test_rescaled}")

            axs[2].set_xlim([test_dates.min(), max(future_dates.max(), test_dates.max())])

            axs[2].plot(test_dates, y_test_rescaled, label='Aktual (Testing)', color='black', linestyle='-', marker='o', markersize=8)

            axs[2].plot(extended_dates, extended_predictions, label='Prediksi (Testing + Masa Depan)', 
                        color='orange', linestyle='--', marker='x', markersize=8)

            axs[2].set_title('Data Testing dan Prediksi Masa Depan')
            axs[2].set_xlabel('Tanggal')
            axs[2].set_ylabel('Total')
            axs[2].legend(loc='upper left')  

            for date, value in zip(extended_dates, extended_predictions):
                axs[2].text(date, value, f'{int(value)}', ha='center', va='bottom', fontsize=10, color='orange')

            axs[2].legend()
            axs[2].grid(True)

            plt.tight_layout()

            img = BytesIO()
            plt.savefig(img, format='png', dpi=100)
            img.seek(0)
            plt.close()

            plot_url = b64encode(img.getvalue()).decode('utf8')
            
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
            })

        except Exception as e:
            print(f'An error occurred: {e}')
            flash(f'An error occurred: {e}', 'danger')
            return redirect('/')

    return render_template('predict.html')

def get_sales_data_by_months(category, start_month, end_month):
    category_obj = Categories.query.filter_by(name=category).first()
    if not category_obj:
        raise ValueError(f"Kategori '{category}' tidak ditemukan.")

    sales_data = Sales.query.filter_by(category_id=category_obj.id).filter(
        db.extract('month', db.func.str_to_date(Sales.date, '%m-%Y')).between(start_month, end_month)
    ).all()
    
    print(f"Raw data fetched from the database for category '{category}' and months {start_month} to {end_month}:")
    for sale in sales_data:
        print(f"Date: {sale.date}, Total: {sale.total}")
    
    df = pd.DataFrame([{
        'Tanggal': sale.date,
        'Total': sale.total
    } for sale in sales_data])
    
    return df

def get_sales_data(category, page, items_per_page):
    query = db.session.query(Sales).join(Categories)  

    if category:
        query = query.filter(Categories.name == category)

    if items_per_page is None or items_per_page == 'all':
        sales_data = query.all()
        total_count = len(sales_data)
    else:
        offset = (page - 1) * items_per_page
        sales_data = query.offset(offset).limit(items_per_page).all()
        total_count = query.count()

    formatted_sales_data = []
    for sale in sales_data:
        formatted_sales_data.append({
            'id': sale.id,
            'date': sale.date,
            'total': sale.total,
            'upload_time': sale.upload_time,
            'category': sale.category.name if sale.category else 'Unknown'  
        })

    return formatted_sales_data, total_count


@app.route('/delete', methods=['GET'])
def delete():
    category = request.args.get('category', default=None)
    items_per_page = request.args.get('items_per_page', default='5')  
    page = request.args.get('page', default=1, type=int)

    if items_per_page == 'all':
        items_per_page = None  
    else:
        items_per_page = int(items_per_page) 

    sales_data, total_count = get_sales_data(category, page, items_per_page)

    if items_per_page is None:
        total_pages = 1
    else:
        total_pages = (total_count + items_per_page - 1) // items_per_page

    return render_template(
        'delete.html',
        sales_data=sales_data,
        selected_category=category,
        items_per_page=items_per_page,
        page=page,
        total_pages=total_pages
    )


@app.route('/edit/<int:id>', methods=['POST'])
def edit_total(id):
    sale = Sales.query.get(id)  
    if sale:
        sale.total = request.form.get('total', type=float)
        
        db.session.commit()
        
        flash('Data updated successfully!', 'success')
    else:
        flash('Sale not found!', 'error')
    
    return redirect(url_for('delete', category=request.args.get('category'), items_per_page=request.args.get('items_per_page')))


@app.route('/delete/<int:id>', methods=['POST'])
def delete_data(id):
    sale = Sales.query.get_or_404(id)
    db.session.delete(sale)
    db.session.commit()
    flash("Data successfully deleted!")
    return redirect(url_for('delete'))

@app.route('/admin/edit_total/<int:data_id>', methods=['POST'])
@login_required
def edit_total_admin(data_id):
    if current_user.role != 'admin':
        flash('Access denied. You are not an admin.', 'danger')
        return redirect(url_for('login'))

    new_total = request.form.get('new_total')
    if not new_total:
        flash('Total cannot be empty.', 'danger')
        return redirect(url_for('admin_list_data'))

    data = Sales.query.get(data_id)
    if data:
        data.total = new_total
        db.session.commit()
        flash('Total successfully updated.', 'success')
    else:
        flash('Data not found.', 'danger')

    return redirect(url_for('admin_list_data'))


@app.route('/admin/delete_data/<int:data_id>', methods=['POST'])
@login_required
def delete_data_admin(data_id):
    if current_user.role != 'admin':
        flash('Access denied. You are not an admin.', 'danger')
        return redirect(url_for('login'))

    data = Sales.query.get(data_id)
    if data:
        db.session.delete(data)
        db.session.commit()
        flash('Data successfully deleted.', 'success')
    else:
        flash('Data not found.', 'danger')

    return redirect(url_for('admin_list_data'))

def read_and_process_data_monthly_user(category):
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

    # Ensure 'Tanggal' column is in datetime format
    df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='%m-%Y', errors='coerce')
    
    # Drop rows with invalid dates
    df.dropna(subset=['Tanggal'], inplace=True)
    
    # Sort the DataFrame by 'Tanggal'
    df = df.sort_values(by='Tanggal', ascending=True)
    
    # Handle outliers
    Q1 = df['Total'].quantile(0.25)
    Q3 = df['Total'].quantile(0.75)
    IQR = Q3 - Q1

    df['Total'] = np.where(
        (df['Total'] < (Q1 - 1.5 * IQR)) | (df['Total'] > (Q3 + 1.5 * IQR)),
        np.nan,
        df['Total']
    )
    
    # Interpolate missing values
    df['Total'] = df['Total'].interpolate(method='linear')

    # Forward fill and backward fill remaining missing values
    df['Total'].fillna(method='ffill', inplace=True)
    df['Total'].fillna(method='bfill', inplace=True)

    print("Processed DataFrame after handling outliers and sorting:")
    print(df.head())
    
    return df

def create_dataset2(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back + 1):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back - 1, 0])
    return np.array(X), np.array(Y)

@app.route('/predict_by_month', methods=['GET', 'POST'])
def predict_by_month():
    if request.method == 'POST':
        # Ambil data input dari form
        category = request.form.get('category')
        start_month = int(request.form.get('start_month'))
        end_month = int(request.form.get('end_month'))

        # Debugging log
        print(f"Received category: {category}, start_month: {start_month}, end_month: {end_month}")

        try:
            # Baca dan proses data berdasarkan kategori
            df = read_and_process_data_monthly_user(category)
            print("Data read successfully")

            if 'Tanggal' not in df.columns:
                raise ValueError("The 'Tanggal' column is missing from the data.")
            if 'Total' not in df.columns:
                raise ValueError("The 'Total' column is missing from the data.")

            # Ubah kolom 'Tanggal' ke format datetime
            df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='%Y-%m', errors='coerce')
            df.dropna(subset=['Tanggal'], inplace=True)
            print("Tanggal column converted to datetime")

            # Filter data berdasarkan bulan
            if start_month <= end_month:
                df_filtered = df[df['Tanggal'].dt.month.isin(range(start_month, end_month + 1))]
            else:
                df_filtered = df[df['Tanggal'].dt.month.isin(
                    list(range(start_month, 13)) + list(range(1, end_month + 1))
                )]

            if df_filtered.empty:
                raise ValueError(f"No data available for the selected months ({start_month} to {end_month}).")

            print(f"Filtered data for months {start_month} to {end_month}:")
            print(df_filtered)
            print(f"Data length: {len(df_filtered)}")

            # Normalisasi data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(df_filtered['Total'].values.reshape(-1, 1))
            print("Data scaled successfully")

            # Hyperparameter
            units_1, units_2 = 128, 64
            batch_size, look_back, epochs = 32, 1, 100
            learning_rate = 0.001

            # Persiapkan dataset untuk LSTM
            X, y = create_dataset2(scaled_data, look_back)
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))
            print("Dataset created for LSTM")
            
            total_samples = len(X)
            print(f"Total samples available: {total_samples}")

            # Split data menjadi training, validation, dan testing
            train_size = int(total_samples * 0.7)
            val_size = int(total_samples * 0.15)
            test_size = total_samples - train_size - val_size
            
            if train_size + val_size + test_size > total_samples:
                raise ValueError("Not enough data to split into training, validation, and testing sets.")

            X_train, y_train = X[:train_size], y[:train_size]
            X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
            X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]
            print("Data split successfully")
            print(f"the data: {X}")
            print(f"the test before scaled : {y_test}")

            # Bangun dan latih model
            model = Sequential([
                LSTM(units_1, return_sequences=True, input_shape=(look_back, 1)),
                Dropout(0.2),
                LSTM(units_2),
                Dropout(0.2),
                Dense(1)
            ])
            optimizer = Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss=Huber())
            print("Model compiled successfully")

            early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
            history = model.fit(X_train, y_train, validation_data=(X_val, y_val), 
                                epochs=epochs, batch_size=batch_size, verbose=2, callbacks=[early_stopping])
            print("Model training completed")

            # Prediksi dan inverse transform
            train_predictions = scaler.inverse_transform(model.predict(X_train))
            val_predictions = scaler.inverse_transform(model.predict(X_val))
            test_predictions = scaler.inverse_transform(model.predict(X_test))

            y_train_rescaled = scaler.inverse_transform(y_train.reshape(-1, 1))
            y_val_rescaled = scaler.inverse_transform(y_val.reshape(-1, 1))
            y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

            # Generate tanggal untuk prediksi
            train_dates = df_filtered['Tanggal'][:train_size]
            val_dates = df_filtered['Tanggal'][train_size:train_size + val_size]
            test_dates = df_filtered['Tanggal'][train_size + val_size:train_size + val_size + len(y_test)]

            # Prediksi masa depan untuk bulan yang diminta pada tahun depan
            last_data_point = scaled_data[-look_back:].reshape(1, look_back, 1)
            future_predictions = []
            future_dates_start = df_filtered['Tanggal'].max() + pd.DateOffset(months=1)

            future_year = future_dates_start.year + 1 if future_dates_start.month == 12 else future_dates_start.year
            future_dates = pd.date_range(
                start=f"{future_year + 1}-{start_month:02d}-01", periods=(end_month - start_month + 1), freq='MS'
            )

            for _ in range(len(future_dates)):
                future_pred = model.predict(last_data_point)
                future_predictions.append(future_pred[0][0])
                last_data_point = np.roll(last_data_point, -1, axis=1)
                last_data_point[0, -1, 0] = future_pred

            future_predictions_rescaled = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
            print(f"Train size: {train_size}, Val size: {val_size}, Test size: {len(y_test)}")
            print(df_filtered['Tanggal'][train_size + val_size:train_size + val_size + len(y_test)])

            # Plotting data
            fig, axs = plt.subplots(4, 1, figsize=(15, 18))

            # Plot Training
            axs[0].plot(train_dates.dt.strftime('%m-%Y'), y_train_rescaled.flatten(), label='Aktual', color='black', marker='o')
            axs[0].plot(train_dates.dt.strftime('%m-%Y'), train_predictions.flatten(), label='Prediksi', color='orange', linestyle='--', marker='x')
            axs[0].set_title('Data Training')
            axs[0].legend()
            axs[0].grid(True)
            print(f"train_dates: {train_dates}")

            # Plot Validation
            axs[1].plot(val_dates.dt.strftime('%m-%Y'), y_val_rescaled.flatten(), label='Aktual', color='black', marker='o')
            axs[1].plot(val_dates.dt.strftime('%m-%Y'), val_predictions.flatten(), label='Prediksi', color='orange', linestyle='--', marker='x')
            axs[1].set_title('Data Validation')
            axs[1].legend()
            axs[1].grid(True)
            print(f"val_dates: {val_dates}")

            axs[2].plot(test_dates.dt.strftime('%m-%Y'), y_test_rescaled.flatten(), label='Aktual (Testing)', color='black', marker='o')
            axs[2].plot(test_dates.dt.strftime('%m-%Y'), test_predictions.flatten(), label='Prediksi (Testing)', color='orange', linestyle='--', marker='x')
            axs[2].set_title('Data Testing')
            axs[2].legend()
            axs[2].grid(True)
            print(f"test_dates: {test_dates}")

            # Plot Future Predictions
            axs[3].plot(future_dates.strftime('%m-%Y'), future_predictions_rescaled.flatten(), label='Prediksi (Masa Depan)', color='blue', linestyle='--', marker='x')
            axs[3].set_title('Prediksi Masa Depan')
            axs[3].legend()
            axs[3].grid(True)
            


            # Simpan Plot ke Buffer
            plt.tight_layout()
            img = BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            plt.close()
            
            plot_url = b64encode(img.getvalue()).decode('utf8')
            
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
            })

        except Exception as e:
            print(f"Error occurred: {e}")
            flash(str(e))
            return redirect(request.url)

    return render_template('predict_by_month.html')

if __name__ == '__main__':
    app.run(debug=True)