# HealthBite 🥗

HealthBite is an AI-powered food health classification and smart nutrition recommendation web application. It leverages an ensemble machine learning approach (KNN, SVM, Naive Bayes) to classify whether a given food item (based on its nutritional macro values per 100g) is generally "Healthy" or "Unhealthy", while also offering confidence intervals and similar food recommendations.

## Features
- **Instant AI Analysis:** Input your food's nutritional facts and get an instant health verdict.
- **Ensemble ML Classification:** Uses a combination of Support Vector Machines, K-Nearest Neighbors, and Gaussian Naive Bayes to output a voting breakdown and a reliable prediction.
- **Nutritional Recommender System:** Built-in KNN Recommender suggests similar foods based on a nutrient profile.
- **Interactive UI:** A glassmorphism, responsive interface.
- **History Dashboard:** Keeps a local database track of all food items you've analyzed.

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/manahillkhitab/HealthBite.git
   cd HealthBite
   ```

2. **Create a virtual environment and activate it:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask application:**
   ```bash
   python app.py
   ```
   The application will run locally at `http://127.0.0.1:5000/`.

## Machine Learning Details
The models are trained using a comprehensive food dataset (`data/cleaned/foods_clean.csv`). You can re-train the models anytime by running:
```bash
python train_models.py
```
This will recreate the `.pkl` files in the `models/` directory.

## License
MIT License
