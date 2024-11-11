from flask import Flask
from extensions import db, bcrypt
from models import User

# Membuat instance Flask
app = Flask(__name__)

# Mengonfigurasi aplikasi
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/skripsi_tes'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'supersecretkey'

# Menginisialisasi ekstensi
db.init_app(app)
bcrypt.init_app(app)

# Menjalankan konteks aplikasi untuk melakukan query
with app.app_context():
    admin_user = User.query.filter_by(username='admin').first()
    if not admin_user:
        hashed_password = bcrypt.generate_password_hash('12345678').decode('utf-8')
        admin_user = User(
            username='admin',
            email='yesayamichael13@gmail.com',
            password=hashed_password,
            is_approved=True,
            role='admin'
        )
        db.session.add(admin_user)
        db.session.commit()
        print("Admin account has been created successfully!")
    else:
        print("Admin account already exists.")
