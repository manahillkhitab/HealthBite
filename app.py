"""
HealthBite — Flask Web Application
Clean routing layer. All ML inference is delegated to ml_service.py.
"""
import os
import json
import platform
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from thefuzz import process

# Bypass Windows WMI or slow system queries when importing packages
platform.machine = lambda: 'AMD64'
try:
    platform.uname = lambda: platform.uname_result('Windows', 'node', '10', '10.0.19045', 'AMD64', 'Intel64 Family 6 Model 158 Stepping 10, GenuineIntel')
except AttributeError:
    pass
platform.win32_ver = lambda *args, **kwargs: ('10', '10.0.19045', 'SP0', 'Multiprocessor Free')
platform.system = lambda: 'Windows'
platform.release = lambda: '10'
platform.version = lambda: '10.0.19045'
platform.processor = lambda: 'Intel64 Family 6 Model 158 Stepping 10, GenuineIntel'

import ml_service as ml

app = Flask(__name__)

# ── Fix #4: Secret key from environment variable ─────────────────
app.secret_key = os.environ.get('SECRET_KEY', 'hb-dev-only-change-in-prod')

# ── Production secret key check ────────────────────────────────
# Exit if the default key is used outside local development.
_debug_mode = os.environ.get('FLASK_DEBUG', 'true').lower() == 'true'
if not _debug_mode and app.secret_key == 'hb-dev-only-change-in-prod':
    raise RuntimeError(
        'FATAL: SECRET_KEY is not set. Use a strong, unique key in production.'
    )

# ── Session & DB config ───────────────────────────────────────────
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

basedir = os.path.abspath(os.path.dirname(__name__))
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(basedir, "healthbite.db")}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    food_name = db.Column(db.String(500), nullable=False)
    prediction = db.Column(db.String(20), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.String(50), nullable=False)
    # Storing raw JSON string of macro data for visualization
    nutrition_json = db.Column(db.Text, nullable=True)

with app.app_context():
    try:
        # Try a test query to verify the schema matches our model definition
        db.create_all()
        PredictionHistory.query.first()
    except Exception as e:
        app.logger.warning(f"Database schema mismatch or corruption detected. Re-creating tables: {e}")
        try:
            db.drop_all()
            db.create_all()
        except Exception as err:
            app.logger.error(f"Failed to rebuild database: {err}")

# Max number of predictions to fetch for dashboard
MAX_HISTORY = 50

# ── Dashboard Helpers ─────────────────────────────────────────────

def get_stats() -> dict:
    """Compute dashboard statistics from the database."""
    preds = PredictionHistory.query.order_by(PredictionHistory.id.desc()).all()
    total = len(preds)

    if total == 0:
        return {
            'total_predictions':  0,
            'healthy_count':      0,
            'unhealthy_count':    0,
            'healthy_percentage': 0,
            'unhealthy_percentage': 0,
            'average_confidence': 0,
            'recent_predictions': [],
        }

    healthy   = sum(1 for p in preds if p.prediction == 'Healthy')
    unhealthy = total - healthy
    avg_conf  = round(sum(p.confidence for p in preds) / total, 1)

    return {
        'total_predictions':    total,
        'healthy_count':        healthy,
        'unhealthy_count':      unhealthy,
        'healthy_percentage':   round(healthy / total * 100, 1),
        'unhealthy_percentage': round(unhealthy / total * 100, 1),
        'average_confidence':   avg_conf,
        'recent_predictions':   preds[:MAX_HISTORY],
    }


def _save_prediction(food_name: str, result: dict, nutrition: dict):
    """Save prediction to the SQLite database."""
    entry = PredictionHistory(
        food_name=food_name,
        prediction=result['prediction'],
        confidence=result['confidence'],
        timestamp=datetime.now().strftime('%b %d, %H:%M'),
        nutrition_json=json.dumps(nutrition)
    )
    db.session.add(entry)
    db.session.commit()


# ── Routes ────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if request.method == 'POST':
        food_name = request.form.get('name', '').strip() or 'Unknown Food'
        serving_size = float(request.form.get('serving_size', 100) or 100)

        # Scale inputs down to 100g if user provided a different serving size
        normalized_data = {}
        for col in ml.FEATURE_COLS:
            val = request.form.get(col, 0)
            try:
                normalized_data[col] = (float(val) / serving_size) * 100
            except (ValueError, TypeError, ZeroDivisionError):
                normalized_data[col] = val

        errors = ml.validate_nutrition(normalized_data)
        if errors:
            return render_template('analyze.html', error='; '.join(errors))

        try:
            features = ml.extract_features(normalized_data)

            # ML inference
            pred_result = ml.predict_health(features)
            recs        = ml.get_recommendations(features)

            # The actual values the user ate (not scaled)
            actual_nutrition = {col: round(float(request.form[col]), 1)
                                for col in ml.FEATURE_COLS}

            full_result = {
                'food_name':        food_name,
                'serving_size':     serving_size,
                'prediction':       pred_result['prediction'],
                'confidence':       pred_result['confidence'],
                'individual_models': pred_result['individual_models'],
                'nutrition':        actual_nutrition,
                'recommendations':  recs,
                'timestamp':        datetime.now().strftime('%Y-%m-%d %H:%M'),
            }

            _save_prediction(food_name, pred_result, actual_nutrition)
            return render_template('results.html', result=full_result)

        except (ValueError, KeyError) as e:
            return render_template('analyze.html',
                error=f"Invalid input data: {e}")
        except Exception as e:
            app.logger.error(f"Analysis failed for '{food_name}': {e}")
            return render_template('analyze.html',
                error="Our AI models encountered an unexpected error. "
                      "Please check your values and try again.")

    return render_template('analyze.html')


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html', stats=get_stats())



# ── REST API ──────────────────────────────────────────────────────

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """JSON API for programmatic access to the food health classifier."""
    data = request.get_json(silent=True) or {}

    food_name = data.get('name', '').strip() or 'Unknown Food'
    serving_size = float(data.get('serving_size', 100) or 100)

    # Scale to 100g equivalent
    normalized_data = {}
    for col in ml.FEATURE_COLS:
        val = data.get(col, 0)
        try:
            normalized_data[col] = (float(val) / serving_size) * 100
        except (ValueError, TypeError, ZeroDivisionError):
            normalized_data[col] = val

    errors = ml.validate_nutrition(normalized_data)
    if errors:
        return jsonify({'error': errors}), 400

    try:
        features    = ml.extract_features(normalized_data)
        pred_result = ml.predict_health(features)
        recs        = ml.get_recommendations(features)
        nutrition   = {col: round(float(data.get(col, 0)), 1) for col in ml.FEATURE_COLS}

        return jsonify({
            'food_name':         food_name,
            'serving_size':      serving_size,
            'prediction':        pred_result['prediction'],
            'confidence':        pred_result['confidence'],
            'individual_models': pred_result['individual_models'],
            'nutrition':         nutrition,
            'recommendations':   recs,
        })
    except Exception as e:
        app.logger.error(f"API analysis failed for '{food_name}': {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/search')
def api_search():
    """Autocomplete endpoint — returns up to 10 matching foods with full nutritional profiles."""
    q = request.args.get('q', '').strip().lower()
    if len(q) < 2:
        return jsonify([])
    
    # process.extract returns a list of tuples: (string, score)
    results = process.extract(q, [str(n) for n in ml.food_names], limit=10)
    
    matches = []
    for name, score in results:
        if score >= 60:
            try:
                # Retrieve the exact index of this food in our dataset
                idx = ml.food_names[ml.food_names == name].index[0]
                row = ml.food_features.iloc[idx]
                matches.append({
                    'name':          name,
                    'calories':      round(float(row.get('calories', 0)), 1),
                    'protein':       round(float(row.get('protein', 0)), 1),
                    'carbs':         round(float(row.get('carbs', 0)), 1),
                    'fat':           round(float(row.get('fat', 0)), 1),
                    'sugar':         round(float(row.get('sugar', 0)), 1),
                    'sodium':        round(float(row.get('sodium', 0)), 1),
                    'saturated_fat': round(float(row.get('saturated_fat', 0)), 1)
                })
            except Exception as e:
                app.logger.warning(f"Failed to load details for autocomplete matched food '{name}': {e}")
                matches.append({'name': name})
                
    return jsonify(matches)


@app.route('/clear-history', methods=['POST'])
def clear_history():
    PredictionHistory.query.delete()
    db.session.commit()
    return redirect(url_for('dashboard'))


# ── Error handlers ────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500


# ── Entry point (dev only) ────────────────────────────────────────

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'true').lower() == 'true'
    print(f"\n[HealthBite] Running on http://127.0.0.1:{port}  debug={debug}\n")
    app.run(debug=debug, port=port, use_reloader=False)
