
from flask import (
    Flask, render_template, request, send_file,
    redirect, url_for, session, flash
)
from functools import wraps
from datetime import timedelta
from werkzeug.security import generate_password_hash, check_password_hash

import mysql.connector
from mysql.connector import Error, IntegrityError
import pandas as pd
import numpy as np
import io
import json
import re
import math

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, PageBreak
)


# ==========================================================
# 1. KONFIGURASI APLIKASI
# ==========================================================

app = Flask(__name__)

app.config["SECRET_KEY"] = "sales_forecasting_secret_change_this_key"
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=24)
app.config["SESSION_REFRESH_EACH_REQUEST"] = True
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"


# ==========================================================
# 2. KONFIGURASI DATABASE MYSQL
# ==========================================================

MYSQL_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "sales_forecasting",
    "port": 3306
}


def get_db_connection():
    return mysql.connector.connect(**MYSQL_CONFIG)


def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                nama VARCHAR(100) NOT NULL,
                username VARCHAR(50) NOT NULL UNIQUE,
                password VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

    finally:
        cursor.close()
        conn.close()


# ==========================================================
# 3. PROTEKSI LOGIN
# ==========================================================

def login_required(view_function):
    @wraps(view_function)
    def wrapped_view(*args, **kwargs):
        if "user_id" not in session:
            flash("Silakan login terlebih dahulu.", "error")
            return redirect(url_for("login"))

        return view_function(*args, **kwargs)

    return wrapped_view


# ==========================================================
# 4. DETEKSI KOLOM DATASET
# ==========================================================

def normalize_column_name(value):
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def detect_column(columns, candidates):
    normalized = {
        normalize_column_name(column): column
        for column in columns
    }

    for candidate in candidates:
        key = normalize_column_name(candidate)

        if key in normalized:
            return normalized[key]

    return None


DATE_CANDIDATES = [
    "Order Date", "Date", "Tanggal",
    "Transaction Date", "Invoice Date",
    "Purchase Date", "OrderDate",
    "TransactionDate", "Timestamp",
    "Created At", "Sale Date", "Sale_Date",
    "Posting Date", "Date of Invoice",
    "InvoiceDate", "Sales Date"
]

SALES_CANDIDATES = [
    "Sales", "Penjualan", "Revenue",
    "Amount", "Total Amount", "Total Sales",
    "Sales Amount", "Total Revenue",
    "Net Sales", "Gross Sales", "Turnover",
    "Value", "Sales_Amount", "TotalAmount",
    "Quantity", "Qty", "Units Sold",
    "Total Units", "Quantity Sold",
    "Product_Quantity"
]


# ==========================================================
# 5. MEMBACA FILE CSV
# ==========================================================

def read_csv_flexible(file_bytes):
    encodings = ["utf-8-sig", "utf-8", "latin1"]

    for encoding in encodings:
        try:
            df = pd.read_csv(
                io.BytesIO(file_bytes),
                encoding=encoding,
                sep=None,
                engine="python"
            )

            if not df.empty and len(df.columns) > 1:
                return df

        except Exception:
            continue

    raise ValueError(
        "File CSV tidak dapat dibaca. "
        "Pastikan file menggunakan format CSV yang benar."
    )


# ==========================================================
# 6. MEMBERSIHKAN NILAI PENJUALAN
# ==========================================================

def parse_sales_values(series):

    def convert(value):
        if pd.isna(value):
            return np.nan

        value = str(value).strip()

        if value.lower() in ("", "nan", "none", "null"):
            return np.nan

        negative = value.startswith("(") and value.endswith(")")

        value = value.replace("(", "").replace(")", "")
        value = re.sub(r"[^0-9,.\-]", "", value)

        if not value:
            return np.nan

        if "," in value and "." in value:
            if value.rfind(",") > value.rfind("."):
                value = value.replace(".", "")
                value = value.replace(",", ".")
            else:
                value = value.replace(",", "")

        elif "," in value:
            parts = value.split(",")

            if len(parts[-1]) in (1, 2):
                value = value.replace(",", ".")
            else:
                value = value.replace(",", "")

        elif value.count(".") > 1:
            parts = value.split(".")

            if len(parts[-1]) in (1, 2):
                value = "".join(parts[:-1]) + "." + parts[-1]
            else:
                value = "".join(parts)

        try:
            number = float(value)

            if negative:
                number = -number

            return number

        except (ValueError, TypeError):
            return np.nan

    return series.map(convert)


# ==========================================================
# 7. MEMBACA FORMAT TANGGAL
# ==========================================================

def parse_date_values(series):
    """
    Membaca tanggal secara fleksibel.

    Jika tanggal memiliki tahun di depan, gunakan yearfirst.
    Jika terdapat bukti format MM/DD/YYYY atau DD/MM/YYYY,
    format dipilih berdasarkan komponen tanggal yang > 12.

    Jika seluruh tanggal ambigu, default DD/MM/YYYY.
    """

    values = series.astype("string").str.strip()
    non_empty = values.dropna()

    if non_empty.empty:
        return pd.to_datetime(values, errors="coerce")

    sample = non_empty.head(500).astype(str)
    first_text = sample.iloc[0]

    if re.match(
        r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}",
        first_text
    ):
        return pd.to_datetime(
            values,
            errors="coerce",
            yearfirst=True
        )

    dayfirst_votes = 0
    monthfirst_votes = 0

    for value in sample:
        match = re.match(
            r"^(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})(?:\D|$)",
            value
        )

        if not match:
            continue

        first = int(match.group(1))
        second = int(match.group(2))

        if first > 12 and second <= 12:
            dayfirst_votes += 1

        elif second > 12 and first <= 12:
            monthfirst_votes += 1

    if monthfirst_votes > dayfirst_votes:
        return pd.to_datetime(
            values,
            errors="coerce",
            dayfirst=False
        )

    return pd.to_datetime(
        values,
        errors="coerce",
        dayfirst=True
    )


# ==========================================================
# 8. DETEKSI BULAN TERAKHIR YANG MUNGKIN BELUM LENGKAP
# ==========================================================

def detect_incomplete_last_month(df, date_column):
    """
    Deteksi indikatif bulan terakhir yang kemungkinan belum lengkap.

    Kriteria:
    - Tanggal transaksi terakhir berada pada tanggal 1 sampai 7.
    - Transaksi hanya tercatat pada maksimal 3 hari kalender
      di bulan terakhir tersebut.

    Ini merupakan heuristik, bukan kepastian.
    """

    valid_dates = df[date_column].dropna()

    if valid_dates.empty:
        return None

    latest_date = valid_dates.max()
    latest_period = latest_date.to_period("M")

    last_month_dates = valid_dates[
        valid_dates.dt.to_period("M") == latest_period
    ]

    unique_days = last_month_dates.dt.normalize().nunique()

    if latest_date.day <= 7 and unique_days <= 3:
        return latest_period

    return None


# ==========================================================
# 9. PERHITUNGAN TREND MOMENT
# ==========================================================

def calculate_trend_moment(y):
    y = np.asarray(y, dtype=float)
    n = len(y)

    if n < 2:
        raise ValueError(
            "Minimal diperlukan 2 periode untuk menghitung Trend Moment."
        )

    if not np.all(np.isfinite(y)):
        raise ValueError(
            "Data penjualan mengandung nilai tidak valid."
        )

    x = np.arange(1, n + 1, dtype=float)

    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x ** 2)

    denominator = (n * sum_x2) - (sum_x ** 2)

    if denominator == 0:
        raise ValueError(
            "Penyebut perhitungan Trend Moment bernilai nol."
        )

    b = (
        (n * sum_xy) - (sum_x * sum_y)
    ) / denominator

    a = (
        (sum_x2 * sum_y) - (sum_x * sum_xy)
    ) / denominator

    return {
        "a": float(a),
        "b": float(b),
        "x": x,
        "sum_x": float(sum_x),
        "sum_y": float(sum_y),
        "sum_xy": float(sum_xy),
        "sum_x2": float(sum_x2)
    }


# ==========================================================
# 10. PERHITUNGAN MAPE DAN MAE
# ==========================================================

def calculate_mape(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    valid = (
        np.isfinite(actual)
        & np.isfinite(predicted)
        & (actual != 0)
    )

    if not np.any(valid):
        return None, 0

    mape = np.mean(
        np.abs(
            (actual[valid] - predicted[valid]) / actual[valid]
        )
    ) * 100

    return float(mape), int(np.sum(valid))


def calculate_mae(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    return float(np.mean(np.abs(actual - predicted)))


# ==========================================================
# 11. REGISTER
# ==========================================================

@app.route("/register", methods=["GET", "POST"])
def register():

    if "user_id" in session:
        return redirect(url_for("index"))

    if request.method == "POST":

        nama = request.form.get("nama", "").strip()
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not nama or not username or not password or not confirm_password:
            return render_template(
                "register.html",
                error="Semua kolom wajib diisi."
            )

        if len(username) < 4:
            return render_template(
                "register.html",
                error="Username minimal 4 karakter."
            )

        if len(password) < 6:
            return render_template(
                "register.html",
                error="Password minimal 6 karakter."
            )

        if password != confirm_password:
            return render_template(
                "register.html",
                error="Konfirmasi password tidak cocok."
            )

        conn = None
        cursor = None

        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            hashed_password = generate_password_hash(password)

            cursor.execute("""
                INSERT INTO users (nama, username, password)
                VALUES (%s, %s, %s)
            """, (nama, username, hashed_password))

            conn.commit()

            return render_template(
                "login.html",
                success="Registrasi berhasil! Silakan login."
            )

        except IntegrityError:
            return render_template(
                "register.html",
                error="Username sudah digunakan."
            )

        except Error:
            app.logger.exception(
                "Kesalahan database saat registrasi"
            )

            return render_template(
                "register.html",
                error="Tidak dapat menyimpan akun. Periksa koneksi MySQL."
            )

        finally:
            if cursor is not None:
                cursor.close()

            if conn is not None and conn.is_connected():
                conn.close()

    return render_template("register.html")


# ==========================================================
# 12. LOGIN
# ==========================================================

@app.route("/login", methods=["GET", "POST"])
def login():

    if "user_id" in session:
        return redirect(url_for("index"))

    if request.method == "POST":

        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if not username or not password:
            return render_template(
                "login.html",
                error="Username dan password wajib diisi."
            )

        conn = None
        cursor = None

        try:
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)

            cursor.execute("""
                SELECT id, nama, username, password
                FROM users
                WHERE username = %s
            """, (username,))

            user = cursor.fetchone()

            if user and check_password_hash(user["password"], password):

                session.clear()

                session["user_id"] = user["id"]
                session["nama"] = user["nama"]
                session["username"] = user["username"]
                session.permanent = True

                return redirect(url_for("index"))

            return render_template(
                "login.html",
                error="Username atau password salah."
            )

        except Error:
            app.logger.exception(
                "Kesalahan database saat login"
            )

            return render_template(
                "login.html",
                error="Koneksi MySQL gagal. Periksa konfigurasi database."
            )

        finally:
            if cursor is not None:
                cursor.close()

            if conn is not None and conn.is_connected():
                conn.close()

    return render_template("login.html")


# ==========================================================
# 13. LOGOUT
# ==========================================================

@app.route("/logout")
@login_required
def logout():
    session.clear()
    flash("Kamu berhasil logout.", "success")
    return redirect(url_for("login"))


# ==========================================================
# 14. DASHBOARD
# ==========================================================

@app.route("/")
@login_required
def index():
    return render_template(
        "index.html",
        nama=session.get("nama"),
        username=session.get("username")
    )


# ==========================================================
# 15. PROSES PREDIKSI
# ==========================================================

@app.route("/predict", methods=["POST"])
@login_required
def predict():

    def show_error(message):
        return render_template(
            "index.html",
            nama=session.get("nama"),
            username=session.get("username"),
            error=message
        )

    if "file" not in request.files:
        return show_error(
            "Silakan pilih file dataset terlebih dahulu."
        )

    file = request.files["file"]

    if file.filename == "":
        return show_error("File belum dipilih.")

    if not file.filename.lower().endswith(".csv"):
        return show_error("Format file harus CSV.")

    try:
        df = read_csv_flexible(file.read())
        df.columns = df.columns.astype(str).str.strip()

        date_column = detect_column(
            df.columns,
            DATE_CANDIDATES
        )

        sales_column = detect_column(
            df.columns,
            SALES_CANDIDATES
        )

        if date_column is None or sales_column is None:
            return show_error(
                "Kolom tanggal atau penjualan tidak ditemukan. "
                "Kolom tersedia: "
                + ", ".join(map(str, df.columns))
            )

        original_rows = len(df)

        df[date_column] = parse_date_values(df[date_column])
        df[sales_column] = parse_sales_values(df[sales_column])

        invalid_date_count = int(df[date_column].isna().sum())
        invalid_sales_count = int(df[sales_column].isna().sum())

        df = df.dropna(
            subset=[date_column, sales_column]
        )

        df = df[
            np.isfinite(df[sales_column])
        ]

        if df.empty:
            return show_error(
                "Tidak ditemukan data tanggal dan penjualan yang valid."
            )

        incomplete_period = detect_incomplete_last_month(
            df,
            date_column
        )

        df["Periode"] = df[date_column].dt.to_period("M")

        monthly_sales_all = (
            df.groupby("Periode")[sales_column]
            .sum()
            .sort_index()
        )

        if incomplete_period is not None:
            monthly_sales = monthly_sales_all[
                monthly_sales_all.index < incomplete_period
            ]
        else:
            monthly_sales = monthly_sales_all.copy()

        if monthly_sales.empty:
            return show_error(
                "Tidak ada periode bulanan lengkap yang dapat digunakan. "
                "Periksa rentang tanggal dataset."
            )

        full_period = pd.period_range(
            start=monthly_sales.index.min(),
            end=monthly_sales.index.max(),
            freq="M"
        )

        missing_periods = [
            str(period)
            for period in full_period
            if period not in monthly_sales.index
        ]

        monthly_sales = monthly_sales.reindex(
            full_period,
            fill_value=0
        )

        n = len(monthly_sales)

        if n < 4:
            return show_error(
                "Minimal diperlukan 4 bulan data untuk "
                "melakukan pembagian training dan testing."
            )

        y = monthly_sales.to_numpy(dtype=float)

        # Model akhir menggunakan seluruh periode yang dipakai.
        full_model = calculate_trend_moment(y)

        a = full_model["a"]
        b = full_model["b"]
        x = full_model["x"]

        sum_x = full_model["sum_x"]
        sum_y = full_model["sum_y"]
        sum_xy = full_model["sum_xy"]
        sum_x2 = full_model["sum_x2"]

        xy = x * y
        x2 = x ** 2

        next_x = n + 1
        prediction = a + (b * next_x)
        trend_values = a + (b * x)

        # Pembagian data kronologis 80% training dan 20% testing.
        test_n = max(2, math.ceil(n * 0.20))
        test_n = min(test_n, n - 2)
        train_n = n - test_n

        y_train = y[:train_n]
        y_test = y[train_n:]

        # Model evaluasi hanya dilatih dengan data training.
        eval_model = calculate_trend_moment(y_train)

        a_eval = eval_model["a"]
        b_eval = eval_model["b"]

        x_test = np.arange(
            train_n + 1,
            n + 1,
            dtype=float
        )

        test_prediction = a_eval + (b_eval * x_test)

        mape, mape_count = calculate_mape(
            y_test,
            test_prediction
        )

        mae = calculate_mae(
            y_test,
            test_prediction
        )

        periods = [
            str(period)
            for period in monthly_sales.index
        ]

        next_period = monthly_sales.index[-1] + 1

        calculation = []

        for i in range(n):
            calculation.append({
                "periode": periods[i],
                "x": round(float(x[i]), 4),
                "y": round(float(y[i]), 2),
                "xy": round(float(xy[i]), 2),
                "x2": round(float(x2[i]), 4)
            })

        evaluation_rows = []

        for i in range(test_n):

            actual = float(y_test[i])
            predicted = float(test_prediction[i])

            if actual != 0:
                percentage_error = (
                    abs((actual - predicted) / actual) * 100
                )
            else:
                percentage_error = None

            evaluation_rows.append({
                "periode": periods[train_n + i],
                "actual": round(actual, 2),
                "predicted": round(predicted, 2),
                "error": (
                    round(percentage_error, 2)
                    if percentage_error is not None
                    else None
                )
            })

        warnings = []

        if incomplete_period is not None:
            warnings.append(
                f"Periode {incomplete_period} tidak disertakan karena "
                "data hanya tercatat pada 1-3 hari kalender pertama. "
                "Ini merupakan deteksi indikatif. Pastikan periode tersebut "
                "memang belum lengkap berdasarkan sumber dataset."
            )

        if mape is not None and mape > 100:
            warnings.append(
                f"MAPE sangat tinggi ({mape:.2f}%). "
                "Periksa periode testing dengan nilai aktual sangat kecil "
                "atau perubahan penjualan ekstrem. "
                "Jangan mengubah data hanya untuk menurunkan nilai MAPE."
            )

        positive_sales = y[y > 0]

        if len(positive_sales) >= 3:
            median_sales = float(np.median(positive_sales))

            low_period_indices = [
                i
                for i, value in enumerate(y)
                if (
                    value > 0
                    and median_sales > 0
                    and value < 0.20 * median_sales
                )
            ]

            if low_period_indices:
                low_period_names = ", ".join(
                    periods[i]
                    for i in low_period_indices
                )

                warnings.append(
                    "Penjualan pada periode berikut jauh di bawah median "
                    "dan dapat meningkatkan MAPE: "
                    + low_period_names
                    + "."
                )

        if n < 24:
            warnings.append(
                "Dataset hanya memiliki kurang dari 24 periode bulanan. "
                "Hasil evaluasi dapat berubah cukup besar karena "
                "jumlah data terbatas."
            )

        if missing_periods:
            warnings.append(
                "Bulan tanpa transaksi ditemukan: "
                + ", ".join(missing_periods)
                + ". Bulan tersebut diisi nol."
            )

        if invalid_date_count > 0:
            warnings.append(
                f"{invalid_date_count} baris memiliki tanggal tidak valid."
            )

        if invalid_sales_count > 0:
            warnings.append(
                f"{invalid_sales_count} baris memiliki nilai penjualan tidak valid."
            )

        chart_labels = periods + [str(next_period)]

        actual_values = [
            round(float(value), 2)
            for value in y
        ]

        actual_values.append(None)

        trend_chart = [
            round(float(value), 2)
            for value in trend_values
        ]

        trend_chart.append(
            round(float(prediction), 2)
        )

        evaluation_labels = [
            row["periode"]
            for row in evaluation_rows
        ]

        evaluation_actual = [
            row["actual"]
            for row in evaluation_rows
        ]

        evaluation_predicted = [
            row["predicted"]
            for row in evaluation_rows
        ]

        return render_template(
            "result.html",
            calculation=calculation,
            total_data=n,
            excluded_period=(
                str(incomplete_period)
                if incomplete_period is not None
                else None
            ),
            dataset_date_column=date_column,
            dataset_sales_column=sales_column,
            sum_x=round(sum_x, 4),
            sum_y=round(sum_y, 2),
            sum_xy=round(sum_xy, 2),
            sum_x2=round(sum_x2, 4),
            a=round(a, 4),
            b=round(b, 4),
            prediction=round(float(prediction), 2),
            next_period=str(next_period),
            mape=(
                round(mape, 2)
                if mape is not None
                else None
            ),
            mape_count=mape_count,
            mae=round(mae, 2),
            train_periods=train_n,
            test_periods=test_n,
            evaluation_rows=evaluation_rows,
            evaluation_labels=evaluation_labels,
            evaluation_actual=evaluation_actual,
            evaluation_predicted=evaluation_predicted,
            warnings=warnings,
            original_rows=original_rows,
            valid_rows=len(df),
            chart_labels=chart_labels,
            actual_values=actual_values,
            trend_chart=trend_chart
        )

    except Exception as e:
        app.logger.exception("Kesalahan proses prediksi")

        return show_error(
            "Terjadi kesalahan saat memproses dataset: " + str(e)
        )


# ==========================================================
# 16. EXPORT EXCEL
# ==========================================================

@app.route("/export", methods=["POST"])
@login_required
def export_excel():

    try:
        calculation = json.loads(
            request.form["calculation"]
        )

        if not isinstance(calculation, list) or not calculation:
            return "Data perhitungan tidak tersedia.", 400

        df_calculation = pd.DataFrame(calculation)

        # Mengubah nama kolom agar mudah dibaca.
        df_calculation = df_calculation.rename(
            columns={
                "periode": "Periode",
                "x": "X",
                "y": "Y (Penjualan)",
                "xy": "XY",
                "x2": "X²"
            }
        )

        # PERBAIKAN:
        # Mengatur urutan kolom agar sama dengan tabel aplikasi.
        df_calculation = df_calculation[
            [
                "Periode",
                "X",
                "Y (Penjualan)",
                "XY",
                "X²"
            ]
        ]

        # Menambahkan baris jumlah.
        total_row = pd.DataFrame([{
            "Periode": "Jumlah",
            "X": float(request.form["sum_x"]),
            "Y (Penjualan)": float(request.form["sum_y"]),
            "XY": float(request.form["sum_xy"]),
            "X²": float(request.form["sum_x2"])
        }])

        df_calculation = pd.concat(
            [
                df_calculation,
                total_row
            ],
            ignore_index=True
        )

        mape_value = request.form.get("mape", "")

        summary = pd.DataFrame({
            "Keterangan": [
                "Metode",
                "Jumlah Periode",
                "Data Training",
                "Data Testing",
                "Konstanta (a)",
                "Koefisien (b)",
                "Periode Prediksi",
                "Hasil Prediksi",
                "MAPE",
                "MAE"
            ],

            "Nilai": [
                "Trend Moment",
                request.form["total_data"],
                request.form["train_periods"],
                request.form["test_periods"],
                request.form["a"],
                request.form["b"],
                request.form["next_period"],
                request.form["prediction"],
                (
                    mape_value + "%"
                    if mape_value
                    else "Tidak tersedia"
                ),
                request.form.get("mae", "")
            ]
        })

        evaluation = pd.DataFrame(
            json.loads(
                request.form.get(
                    "evaluation_rows",
                    "[]"
                )
            )
        )

        # Mengatur urutan kolom evaluasi agar lebih mudah dibaca.
        if not evaluation.empty:
            evaluation = evaluation[
                [
                    "periode",
                    "actual",
                    "predicted",
                    "error"
                ]
            ]

            evaluation = evaluation.rename(
                columns={
                    "periode": "Periode",
                    "actual": "Data Aktual",
                    "predicted": "Hasil Prediksi",
                    "error": "Error (%)"
                }
            )

        output = io.BytesIO()

        with pd.ExcelWriter(
            output,
            engine="openpyxl"
        ) as writer:

            summary.to_excel(
                writer,
                sheet_name="Ringkasan",
                index=False
            )

            df_calculation.to_excel(
                writer,
                sheet_name="Perhitungan",
                index=False
            )

            evaluation.to_excel(
                writer,
                sheet_name="Evaluasi Testing",
                index=False
            )

            from openpyxl.styles import (
                Font,
                PatternFill,
                Alignment
            )

            # Format setiap sheet Excel.
            for sheet in writer.sheets.values():

                sheet.freeze_panes = "A2"
                sheet.auto_filter.ref = sheet.dimensions

                for cell in sheet[1]:

                    cell.font = Font(
                        bold=True,
                        color="FFFFFF"
                    )

                    cell.fill = PatternFill(
                        start_color="173B70",
                        end_color="173B70",
                        fill_type="solid"
                    )

                    cell.alignment = Alignment(
                        horizontal="center",
                        vertical="center"
                    )

                # Mengatur lebar kolom otomatis.
                for column in sheet.columns:

                    max_length = max(
                        len(str(cell.value or ""))
                        for cell in column
                    )

                    sheet.column_dimensions[
                        column[0].column_letter
                    ].width = min(
                        max_length + 4,
                        35
                    )

                # Perataan isi tabel.
                for row in sheet.iter_rows(min_row=2):
                    for cell in row:
                        cell.alignment = Alignment(
                            vertical="center"
                        )

                sheet.row_dimensions[1].height = 22

        output.seek(0)

        return send_file(
            output,
            as_attachment=True,
            download_name="Hasil_Prediksi_Trend_Moment.xlsx",
            mimetype=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            )
        )

    except Exception as e:
        app.logger.exception("Gagal membuat Excel")

        return (
            "Gagal membuat file Excel: " + str(e),
            500
        )


# ==========================================================
# 17. EXPORT PDF
# PDF DIPERTAHANKAN SEPERTI VERSI SEBELUMNYA
# ==========================================================

@app.route("/export-pdf", methods=["POST"])
@login_required
def export_pdf():

    try:
        calculation = json.loads(
            request.form["calculation"]
        )

        if not isinstance(calculation, list) or not calculation:
            return "Data perhitungan tidak tersedia.", 400

        total_data = int(
            request.form["total_data"]
        )

        train_periods = int(
            request.form["train_periods"]
        )

        test_periods = int(
            request.form["test_periods"]
        )

        a = float(request.form["a"])
        b = float(request.form["b"])
        prediction = float(request.form["prediction"])

        next_period = request.form["next_period"]

        mape = request.form.get("mape", "")
        mae = request.form.get("mae", "")

        sum_x = float(request.form["sum_x"])
        sum_y = float(request.form["sum_y"])
        sum_xy = float(request.form["sum_xy"])
        sum_x2 = float(request.form["sum_x2"])

        evaluation_rows = json.loads(
            request.form.get(
                "evaluation_rows",
                "[]"
            )
        )

        output = io.BytesIO()

        doc = SimpleDocTemplate(
            output,
            pagesize=landscape(A4),
            rightMargin=1.5 * cm,
            leftMargin=1.5 * cm,
            topMargin=1.5 * cm,
            bottomMargin=1.5 * cm,
            title="Laporan Sales Forecasting"
        )

        styles = getSampleStyleSheet()

        navy = colors.HexColor("#173B70")
        light_blue = colors.HexColor("#EAF1FB")
        grey = colors.HexColor("#64748B")

        title_style = ParagraphStyle(
            "TitleCustom",
            parent=styles["Title"],
            fontSize=18,
            textColor=navy,
            alignment=TA_CENTER,
            spaceAfter=10
        )

        heading_style = ParagraphStyle(
            "HeadingCustom",
            parent=styles["Heading2"],
            fontSize=12,
            textColor=navy,
            spaceBefore=10,
            spaceAfter=7
        )

        def create_table(data, widths):

            table = Table(
                data,
                colWidths=widths,
                repeatRows=1
            )

            table.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), navy),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.4,
                        colors.HexColor("#D5DEEA")
                    ),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, light_blue]
                    ),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6)
                ])
            )

            return table

        elements = []

        elements.append(
            Paragraph(
                "LAPORAN HASIL PREDIKSI PENJUALAN",
                title_style
            )
        )

        elements.append(
            Paragraph(
                "Sistem Prediksi Penjualan Menggunakan Metode Trend Moment",
                ParagraphStyle(
                    "Subtitle",
                    parent=styles["Normal"],
                    alignment=TA_CENTER
                )
            )
        )

        elements.append(Spacer(1, 15))

        elements.append(
            Paragraph(
                "1. Informasi Dataset",
                heading_style
            )
        )

        info_data = [
            ["Keterangan", "Hasil"],
            ["Metode", "Trend Moment"],
            ["Jumlah periode", f"{total_data} bulan"],
            ["Data training", f"{train_periods} bulan"],
            ["Data testing", f"{test_periods} bulan"],
            ["Periode prediksi", next_period]
        ]

        elements.append(
            create_table(
                info_data,
                [7 * cm, 18 * cm]
            )
        )

        elements.append(
            Paragraph(
                "2. Perhitungan Trend Moment",
                heading_style
            )
        )

        formula_data = [
            ["Komponen", "Hasil"],
            ["Persamaan", "Y = a + bX"],
            ["Konstanta (a)", f"{a:,.4f}"],
            ["Koefisien (b)", f"{b:,.4f}"],
            ["Jumlah X", f"{sum_x:,.4f}"],
            ["Jumlah Y", f"{sum_y:,.2f}"],
            ["Jumlah XY", f"{sum_xy:,.2f}"],
            ["Jumlah X²", f"{sum_x2:,.4f}"],
            ["Prediksi", f"{prediction:,.2f}"]
        ]

        elements.append(
            create_table(
                formula_data,
                [7 * cm, 18 * cm]
            )
        )

        elements.append(PageBreak())

        elements.append(
            Paragraph(
                "3. HASIL EVALUASI MODEL",
                title_style
            )
        )

        prediction_data = [
            ["Keterangan", "Hasil"],
            ["Periode prediksi", next_period],
            ["Prediksi penjualan", f"{prediction:,.2f}"],
            [
                "MAPE",
                f"{mape}%"
                if mape
                else "Tidak tersedia"
            ],
            ["MAE", str(mae)]
        ]

        elements.append(
            create_table(
                prediction_data,
                [7 * cm, 18 * cm]
            )
        )

        elements.append(
            Paragraph(
                "4. Evaluasi Data Testing",
                heading_style
            )
        )

        eval_data = [[
            "Periode",
            "Aktual",
            "Prediksi",
            "Error (%)"
        ]]

        for row in evaluation_rows:

            eval_data.append([
                row["periode"],
                f'{row["actual"]:,.2f}',
                f'{row["predicted"]:,.2f}',
                (
                    f'{row["error"]:,.2f}%'
                    if row["error"] is not None
                    else "Tidak tersedia"
                )
            ])

        if len(eval_data) == 1:
            eval_data.append([
                "-",
                "-",
                "-",
                "-"
            ])

        elements.append(
            create_table(
                eval_data,
                [
                    5 * cm,
                    6 * cm,
                    6 * cm,
                    6 * cm
                ]
            )
        )

        elements.append(
            Paragraph(
                "5. Tabel Perhitungan",
                heading_style
            )
        )

        detail_data = [[
            "Periode",
            "X",
            "Y",
            "XY",
            "X²"
        ]]

        for row in calculation:

            detail_data.append([
                str(row["periode"]),
                str(row["x"]),
                f'{float(row["y"]):,.2f}',
                f'{float(row["xy"]):,.2f}',
                str(row["x2"])
            ])

        detail_data.append([
            "JUMLAH",
            f"{sum_x:,.4f}",
            f"{sum_y:,.2f}",
            f"{sum_xy:,.2f}",
            f"{sum_x2:,.4f}"
        ])

        elements.append(
            create_table(
                detail_data,
                [
                    5 * cm,
                    4 * cm,
                    5 * cm,
                    5 * cm,
                    5 * cm
                ]
            )
        )

        def add_page_number(canvas, document):

            canvas.saveState()

            width, height = landscape(A4)

            canvas.setFont(
                "Helvetica",
                8
            )

            canvas.setFillColor(grey)

            canvas.drawString(
                1.5 * cm,
                0.8 * cm,
                "SALES FORECASTING | TREND MOMENT"
            )

            canvas.drawRightString(
                width - 1.5 * cm,
                0.8 * cm,
                f"Halaman {document.page}"
            )

            canvas.restoreState()

        doc.build(
            elements,
            onFirstPage=add_page_number,
            onLaterPages=add_page_number
        )

        output.seek(0)

        return send_file(
            output,
            as_attachment=True,
            download_name="Laporan_Sales_Forecasting.pdf",
            mimetype="application/pdf"
        )

    except Exception as e:
        app.logger.exception("Gagal membuat PDF")

        return (
            "Gagal membuat laporan PDF: " + str(e),
            500
        )


# ==========================================================
# 18. MENJALANKAN APLIKASI
# ==========================================================

if __name__ == "__main__":

    try:
        init_db()
        print("Database MySQL siap digunakan.")

    except Error as e:
        print("Gagal terhubung ke MySQL:", e)
        print("Pastikan MySQL aktif dan database tersedia.")
        raise

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )
