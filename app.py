from flask import Flask, render_template, request, send_file, redirect, url_for, session
import joblib
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

app = Flask(__name__)
app.secret_key = "predictmaint_ai_secret_key"

UPLOAD_FOLDER = "uploads"
REPORT_FOLDER = "reports"
DATA_FILE = "dashboard_data.json"
USER_FILE = "user_data.json"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

model = joblib.load("maintenance_model.pkl")

DEFAULT_SUMMARY = {
    "total": 10000,
    "healthy": 9660,
    "warning": 279,
    "critical": 61,
    "overall_risk": "LOW",
    "filename": "Demo Data",
    "updated_at": "Not uploaded yet",
    "latest_report": ""
}

DEFAULT_USERS = {
    "admin": {
        "password": "admin123",
        "name": "Lakhan",
        "role": "Admin User"
    }
}

REQUIRED_COLUMNS = {
    "air_temp": [
        "air temperature [k]", "air_temp", "air temperature", "air temp",
        "ambient temp", "ambient temperature", "oil temp", "oil temperature",
        "temperature", "equipment temperature", "inlet temperature"
    ],
    "process_temp": [
        "process temperature [k]", "process_temp", "process temperature", "process temp",
        "bearing temp", "bearing temperature", "machine temp", "pump temperature",
        "compressor temperature", "outlet temperature"
    ],
    "rpm": [
        "rotational speed [rpm]", "rpm", "motor rpm", "shaft rpm",
        "speed", "motor speed", "rotational speed", "rotation speed"
    ],
    "torque": [
        "torque [nm]", "torque", "shaft torque", "load torque",
        "motor torque", "load", "motor load"
    ],
    "tool_wear": [
        "tool wear [min]", "tool_wear", "tool wear", "wear",
        "running hours", "operating hours", "runtime", "machine hours",
        "equipment age", "usage hours", "cycles", "cycle", "time_cycle"
    ]
}

def is_logged_in():
    return session.get("logged_in") is True


def load_users():
    if os.path.exists(USER_FILE):
        with open(USER_FILE, "r") as f:
            data = json.load(f)

            # Old single-user format ko multi-user format me convert karta hai
            if "username" in data and "password" in data:
                return {
                    data["username"]: {
                        "password": data["password"],
                        "name": data.get("name", "Lakhan"),
                        "role": data.get("role", "Admin User")
                    }
                }

            return data

    return DEFAULT_USERS


def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f, indent=4)


def load_dashboard_summary():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return DEFAULT_SUMMARY


def save_dashboard_summary(summary):
    with open(DATA_FILE, "w") as f:
        json.dump(summary, f, indent=4)


def find_column(df, possible_names):
    cleaned_cols = {col.lower().strip(): col for col in df.columns}
    for name in possible_names:
        if name in cleaned_cols:
            return cleaned_cols[name]

    for col in df.columns:
        col_clean = col.lower().strip()
        for name in possible_names:
            if name in col_clean:
                return col

    return None


def get_status(probability):
    if probability < 30:
        return "Healthy", "Machine is safe. Continue regular monitoring."
    elif probability < 70:
        return "Warning", "Maintenance inspection recommended soon."
    else:
        return "Critical", "Immediate maintenance required. High failure risk."



def detect_product_ids(df):
    possible_ids = [
        "Product ID", "Machine ID", "Equipment ID", "equipment_id", "machine_id",
        "asset_id", "Asset ID", "vehicle_id", "unit_number", "id", "ID"
    ]
    for col in possible_ids:
        if col in df.columns:
            return df[col].astype(str)
    return pd.Series([f"EQP_{i+1:05d}" for i in range(len(df))])


def run_ai4i_model(df, mapped_columns):
    input_data = pd.DataFrame({
        "Air temperature [K]": pd.to_numeric(df[mapped_columns["air_temp"]], errors="coerce"),
        "Process temperature [K]": pd.to_numeric(df[mapped_columns["process_temp"]], errors="coerce"),
        "Rotational speed [rpm]": pd.to_numeric(df[mapped_columns["rpm"]], errors="coerce"),
        "Torque [Nm]": pd.to_numeric(df[mapped_columns["torque"]], errors="coerce"),
        "Tool wear [min]": pd.to_numeric(df[mapped_columns["tool_wear"]], errors="coerce"),
    })
    input_data = input_data.fillna(input_data.mean(numeric_only=True)).fillna(0)
    probabilities = model.predict_proba(input_data)[:, 1] * 100
    result_df = pd.DataFrame({
        "Product ID": detect_product_ids(df),
        "Air Temp [K]": input_data["Air temperature [K]"],
        "Process Temp [K]": input_data["Process temperature [K]"],
        "RPM": input_data["Rotational speed [rpm]"],
        "Torque [Nm]": input_data["Torque [Nm]"],
        "Tool Wear [min]": input_data["Tool wear [min]"],
        "Failure Probability (%)": np.round(probabilities, 2),
    })
    result_df["Analysis Mode"] = "ML Prediction Mode"
    return result_df


def run_universal_analysis(df):
    numeric_df = df.select_dtypes(include=[np.number]).copy()

    # Numeric columns agar text ke form me aaye hon to convert karne ki koshish
    if numeric_df.shape[1] < 2:
        converted = {}
        for col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            if vals.notna().sum() >= max(3, int(len(df) * 0.4)):
                converted[col] = vals
        numeric_df = pd.DataFrame(converted)

    if numeric_df.empty or numeric_df.shape[1] < 2:
        raise ValueError("CSV me enough numeric sensor columns nahi hain. At least 2 numeric columns required hain.")

    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
    numeric_df = numeric_df.fillna(numeric_df.median(numeric_only=True)).fillna(0)

    # Constant columns remove
    useful_cols = [c for c in numeric_df.columns if numeric_df[c].nunique(dropna=True) > 1]
    if len(useful_cols) >= 2:
        numeric_df = numeric_df[useful_cols]

    # Robust anomaly/risk score using percentile ranks + z-score intensity
    ranks = numeric_df.rank(pct=True).fillna(0.5)
    med = numeric_df.median()
    mad = (numeric_df - med).abs().median().replace(0, np.nan)
    z = ((numeric_df - med).abs() / (1.4826 * mad)).replace([np.inf, -np.inf], np.nan).fillna(0)
    z_score = z.clip(0, 5) / 5

    risk_raw = (0.60 * ranks.mean(axis=1)) + (0.40 * z_score.mean(axis=1))
    probabilities = (risk_raw * 100).clip(0, 100)

    cols = list(numeric_df.columns)
    def col_at(i):
        return numeric_df[cols[i % len(cols)]]

    result_df = pd.DataFrame({
        "Product ID": detect_product_ids(df),
        "Air Temp [K]": np.round(col_at(0), 2),
        "Process Temp [K]": np.round(col_at(1), 2),
        "RPM": np.round(col_at(2), 2),
        "Torque [Nm]": np.round(col_at(3), 2),
        "Tool Wear [min]": np.round(col_at(4), 2),
        "Failure Probability (%)": np.round(probabilities, 2),
    })
    result_df["Analysis Mode"] = "Universal Analysis Mode"
    return result_df


def finalize_prediction_result(result_df):
    statuses = []
    recommendations = []
    for prob in result_df["Failure Probability (%)"]:
        status, recommendation = get_status(float(prob))
        statuses.append(status)
        recommendations.append(recommendation)
    result_df["Risk Status"] = statuses
    result_df["Recommendation"] = recommendations
    return result_df, statuses


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None

    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]

        users = load_users()

        if username in users and users[username]["password"] == password:
            session["logged_in"] = True
            session["account_username"] = username
            session["username"] = users[username]["name"]
            session["role"] = users[username]["role"]
            return redirect(url_for("dashboard"))
        else:
            error = "Invalid username or password"

    return render_template("login.html", error=error)


@app.route("/register", methods=["GET", "POST"])
def register():
    error = None
    message = None

    if request.method == "POST":
        username = request.form["username"].strip()
        name = request.form["name"].strip()
        role = request.form["role"].strip()
        password = request.form["password"]
        confirm_password = request.form["confirm_password"]

        users = load_users()

        if username in users:
            error = "Username already exists."
        elif password != confirm_password:
            error = "Passwords do not match."
        elif len(password) < 4:
            error = "Password must be at least 4 characters."
        else:
            users[username] = {
                "password": password,
                "name": name,
                "role": role
            }

            save_users(users)
            message = "Account created successfully. Please login."

    return render_template("register.html", error=error, message=message)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/settings", methods=["GET", "POST"])
def settings():
    if not is_logged_in():
        return redirect(url_for("login"))

    users = load_users()
    account_username = session.get("account_username", "admin")

    if account_username not in users:
        session.clear()
        return redirect(url_for("login"))

    user = users[account_username]
    message = None
    error = None

    if request.method == "POST":
        user["name"] = request.form["name"].strip()
        user["role"] = request.form["role"].strip()

        old_password = request.form.get("old_password", "")
        new_password = request.form.get("new_password", "")
        confirm_password = request.form.get("confirm_password", "")

        if new_password:
            if old_password != user["password"]:
                error = "Old password is incorrect."
            elif new_password != confirm_password:
                error = "New password and confirm password do not match."
            elif len(new_password) < 4:
                error = "New password must be at least 4 characters."
            else:
                user["password"] = new_password
                message = "Profile and password updated successfully."
        else:
            message = "Profile updated successfully."

        users[account_username] = user
        save_users(users)

        session["username"] = user["name"]
        session["role"] = user["role"]

    return render_template("settings.html", user=user, message=message, error=error)


@app.route("/")
def dashboard():
    if not is_logged_in():
        return redirect(url_for("login"))

    summary = load_dashboard_summary()
    return render_template("dashboard.html", summary=summary)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if not is_logged_in():
        return redirect(url_for("login"))

    result = None

    if request.method == "POST":
        input_data = pd.DataFrame({
            "Air temperature [K]": [float(request.form["air_temp"])],
            "Process temperature [K]": [float(request.form["process_temp"])],
            "Rotational speed [rpm]": [float(request.form["rpm"])],
            "Torque [Nm]": [float(request.form["torque"])],
            "Tool wear [min]": [float(request.form["tool_wear"])]
        })

        probability = model.predict_proba(input_data)[0][1] * 100
        status, recommendation = get_status(probability)

        result = {
            "status": status,
            "probability": round(probability, 2),
            "recommendation": recommendation
        }

    return render_template("predict.html", result=result)


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if not is_logged_in():
        return redirect(url_for("login"))

    results = None
    error = None
    download_file = None
    summary = None

    if request.method == "POST":
        company = request.form.get("company", "Unknown")
        file = request.files.get("csv_file")

        if not file or file.filename == "":
            error = "Please upload a CSV file."
            return render_template("upload.html", results=results, error=error)

        if not file.filename.lower().endswith(".csv"):
            error = "Only CSV files are allowed."
            return render_template("upload.html", results=results, error=error)

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            error = f"CSV file could not be read: {e}"
            return render_template("upload.html", results=results, error=error)

        try:
            mapped_columns = {}
            missing_columns = []

            for key, names in REQUIRED_COLUMNS.items():
                detected_col = find_column(df, names)
                if detected_col:
                    mapped_columns[key] = detected_col
                else:
                    missing_columns.append(key)

            # Mode 1: AI4I/compatible data -> trained ML model
            if len(missing_columns) == 0:
                result_df = run_ai4i_model(df, mapped_columns)
                analysis_mode = "ML Prediction Mode"
                note = "Compatible sensor columns detected. Trained ML model used."
            else:
                # Mode 2: Any company CSV -> universal numeric risk analysis
                result_df = run_universal_analysis(df)
                analysis_mode = "Universal Analysis Mode"
                note = "Required ML columns not found. Universal numeric sensor analysis used."

            result_df, statuses = finalize_prediction_result(result_df)

        except Exception as e:
            error = f"Analysis failed: {e}"
            return render_template("upload.html", results=results, error=error)

        total = len(result_df)
        healthy = statuses.count("Healthy")
        warning = statuses.count("Warning")
        critical = statuses.count("Critical")

        if critical > 0:
            overall_risk = "HIGH"
        elif warning > 0:
            overall_risk = "MEDIUM"
        else:
            overall_risk = "LOW"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"prediction_result_{timestamp}.csv"
        output_path = os.path.join(REPORT_FOLDER, output_filename)

        summary = {
            "filename": file.filename,
            "company": company,
            "analysis_mode": analysis_mode,
            "note": note,
            "total": total,
            "healthy": healthy,
            "warning": warning,
            "critical": critical,
            "overall_risk": overall_risk,
            "updated_at": datetime.now().strftime("%d %b %Y, %I:%M %p"),
            "latest_report": output_filename
        }

        save_dashboard_summary(summary)
        result_df.to_csv(output_path, index=False)

        results = result_df.head(20).to_dict(orient="records")
        download_file = output_filename

    return render_template(
        "upload.html",
        results=results,
        error=error,
        download_file=download_file,
        summary=summary
    )

@app.route("/download/<filename>")
def download(filename):
    if not is_logged_in():
        return redirect(url_for("login"))

    path = os.path.join(REPORT_FOLDER, filename)

    if not os.path.exists(path):
        return "File not found"

    return send_file(path, as_attachment=True)


@app.route("/analytics")
def analytics():
    if not is_logged_in():
        return redirect(url_for("login"))

    summary = load_dashboard_summary()
    latest_report = summary.get("latest_report")

    analytics_data = None
    top_machines = []

    if latest_report:
        report_path = os.path.join(REPORT_FOLDER, latest_report)

        if os.path.exists(report_path):
            df = pd.read_csv(report_path)

            total = len(df)
            healthy = len(df[df["Risk Status"] == "Healthy"])
            warning = len(df[df["Risk Status"] == "Warning"])
            critical = len(df[df["Risk Status"] == "Critical"])
            avg_probability = round(df["Failure Probability (%)"].mean(), 2)

            top_machines = df.sort_values(
                by="Failure Probability (%)",
                ascending=False
            ).head(10).to_dict(orient="records")

            analytics_data = {
                "total": total,
                "healthy": healthy,
                "warning": warning,
                "critical": critical,
                "avg_probability": avg_probability,
                "healthy_percent": round((healthy / total) * 100, 2),
                "warning_percent": round((warning / total) * 100, 2),
                "critical_percent": round((critical / total) * 100, 2)
            }

    return render_template(
        "analytics.html",
        data=analytics_data,
        top_machines=top_machines
    )


@app.route("/machines")
def machines():
    if not is_logged_in():
        return redirect(url_for("login"))

    summary = load_dashboard_summary()
    latest_report = summary.get("latest_report")

    machines_data = []
    error = None

    search = request.args.get("search", "").strip()
    status = request.args.get("status", "").strip()
    sort = request.args.get("sort", "").strip()

    if latest_report:
        report_path = os.path.join(REPORT_FOLDER, latest_report)

        if os.path.exists(report_path):
            df = pd.read_csv(report_path)

            if search:
                df = df[df["Product ID"].astype(str).str.contains(search, case=False, na=False)]

            if status:
                df = df[df["Risk Status"] == status]

            if sort == "high_risk":
                df = df.sort_values(by="Failure Probability (%)", ascending=False)

            machines_data = df.head(100).to_dict(orient="records")
        else:
            error = "No latest report file found."
    else:
        error = "Please upload CSV first."

    return render_template(
        "machine_details.html",
        machines=machines_data,
        error=error,
        search=search,
        status=status,
        sort=sort
    )


@app.route("/reports")
def reports():
    if not is_logged_in():
        return redirect(url_for("login"))

    report_list = []

    if os.path.exists(REPORT_FOLDER):
        files = sorted(os.listdir(REPORT_FOLDER), reverse=True)

        for file in files:
            path = os.path.join(REPORT_FOLDER, file)

            if os.path.isfile(path):
                report_list.append({
                    "name": file,
                    "size": round(os.path.getsize(path) / 1024, 2),
                    "created": datetime.fromtimestamp(os.path.getctime(path)).strftime("%d %b %Y, %I:%M %p")
                })

    return render_template("reports.html", reports=report_list)


@app.route("/delete_report/<filename>")
def delete_report(filename):
    if not is_logged_in():
        return redirect(url_for("login"))

    path = os.path.join(REPORT_FOLDER, filename)

    if os.path.exists(path):
        os.remove(path)

    return redirect(url_for("reports"))
FACTORY_FILE = "factories.json"


def load_factories():
    if os.path.exists(FACTORY_FILE):
        with open(FACTORY_FILE, "r") as f:
            return json.load(f)

    return {
        "active_factory": "ONGC Plant",
        "factories": [
            {"name": "ONGC Plant", "location": "Dehradun", "status": "Active"},
            {"name": "IOCL Refinery", "location": "Panipat", "status": "Active"},
            {"name": "BPCL Unit", "location": "Mumbai", "status": "Active"},
            {"name": "Shell Facility", "location": "Gujarat", "status": "Active"},
            {"name": "Reliance Plant", "location": "Jamnagar", "status": "Active"}
        ]
    }


def save_factories(data):
    with open(FACTORY_FILE, "w") as f:
        json.dump(data, f, indent=4)


@app.route("/notifications")
def notifications():
    if not is_logged_in():
        return redirect(url_for("login"))

    summary = load_dashboard_summary()

    notifications_data = [
        {
            "title": "Critical Machine Alert",
            "message": f"{summary.get('critical', 0)} machines need immediate maintenance.",
            "level": "Critical",
            "time": summary.get("updated_at", "Not available")
        },
        {
            "title": "Warning Machines",
            "message": f"{summary.get('warning', 0)} machines require inspection soon.",
            "level": "Warning",
            "time": summary.get("updated_at", "Not available")
        },
        {
            "title": "Dataset Analyzed",
            "message": f"Latest dataset: {summary.get('filename', 'No dataset uploaded')}",
            "level": "Info",
            "time": summary.get("updated_at", "Not available")
        }
    ]

    return render_template("notifications.html", notifications=notifications_data)


@app.route("/history")
def history():
    if not is_logged_in():
        return redirect(url_for("login"))

    summary = load_dashboard_summary()
    latest_report = summary.get("latest_report")
    rows = []

    if latest_report:
        path = os.path.join(REPORT_FOLDER, latest_report)

        if os.path.exists(path):
            df = pd.read_csv(path)

            if "Failure Probability (%)" in df.columns:
                df = df.sort_values(by="Failure Probability (%)", ascending=False)

            rows = df.head(30).to_dict(orient="records")

    return render_template("maintenance_history.html", rows=rows)


@app.route("/factories", methods=["GET", "POST"])
def factories():
    if not is_logged_in():
        return redirect(url_for("login"))

    data = load_factories()

    if request.method == "POST":
        action = request.form.get("action")

        if action == "add":
            name = request.form.get("name", "").strip()
            location = request.form.get("location", "").strip()

            if name and location:
                data["factories"].append({
                    "name": name,
                    "location": location,
                    "status": "Active"
                })

        elif action == "set_active":
            data["active_factory"] = request.form.get("active_factory")

        save_factories(data)

    return render_template("factories.html", data=data)


@app.route("/calendar")
def calendar():
    if not is_logged_in():
        return redirect(url_for("login"))

    summary = load_dashboard_summary()
    latest_report = summary.get("latest_report")
    tasks = []

    if latest_report:
        path = os.path.join(REPORT_FOLDER, latest_report)

        if os.path.exists(path):
            df = pd.read_csv(path)

            if "Failure Probability (%)" in df.columns:
                df = df.sort_values(by="Failure Probability (%)", ascending=False)

            top_rows = df.head(20).to_dict(orient="records")

            for index, row in enumerate(top_rows):
                prob = float(row.get("Failure Probability (%)", 0))
                risk = row.get("Risk Status", "Unknown")

                if risk == "Critical":
                    days = 0
                elif risk == "Warning":
                    days = 3
                else:
                    days = 30

                date_value = datetime.now() + pd.Timedelta(days=days + index // 5)

                tasks.append({
                    "date": date_value.strftime("%d %b %Y"),
                    "machine": row.get("Product ID", "Unknown"),
                    "risk": risk,
                    "probability": prob,
                    "action": row.get("Recommendation", "Regular inspection required.")
                })

    return render_template("calendar.html", tasks=tasks)


@app.route("/admin/users", methods=["GET", "POST"])
def admin_users():
    if not is_logged_in():
        return redirect(url_for("login"))

    users = load_users()
    message = None

    if request.method == "POST":
        username = request.form.get("username")
        role = request.form.get("role")

        if username in users:
            users[username]["role"] = role
            save_users(users)
            message = "User role updated successfully."

    return render_template("admin_users.html", users=users, message=message)


@app.route("/admin/delete_user/<username>")
def delete_user(username):
    if not is_logged_in():
        return redirect(url_for("login"))

    users = load_users()
    current_user = session.get("account_username")

    if username in users and username != current_user:
        del users[username]
        save_users(users)

    return redirect(url_for("admin_users"))

if __name__ == "__main__":
    app.run(debug=True)