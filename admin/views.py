from flask import render_template
from . import admin_bp  # Import Blueprint dari __init__.py

@admin_bp.route('/admin_dashboard')
def admin_dashboard():
    return render_template('admin_dashboard.html')
