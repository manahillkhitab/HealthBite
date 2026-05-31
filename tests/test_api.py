import pytest
from app import app, db

@pytest.fixture
def client():
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    
    with app.test_client() as client:
        with app.app_context():
            db.create_all()
        yield client
        with app.app_context():
            db.session.remove()
            db.drop_all()

def test_api_search_empty(client):
    res = client.get('/api/search?q=a')
    assert res.status_code == 200
    assert res.json == []

def test_api_search_fuzzy(client):
    # Depending on what's in models/food_names.pkl
    # "appl" should fuzzy match "Apple"
    res = client.get('/api/search?q=appl')
    assert res.status_code == 200
    assert type(res.json) == list

def test_api_analyze_valid(client):
    data = {
        'name': 'Test Apple',
        'serving_size': 100,
        'calories': 52,
        'protein': 0.3,
        'carbs': 14,
        'fat': 0.2,
        'sugar': 10,
        'sodium': 1,
        'saturated_fat': 0.0
    }
    res = client.post('/api/analyze', json=data)
    assert res.status_code == 200
    json_data = res.json
    assert json_data['food_name'] == 'Test Apple'
    assert 'prediction' in json_data
    assert 'recommendations' in json_data

def test_api_analyze_invalid(client):
    data = {
        'calories': -50
    }
    res = client.post('/api/analyze', json=data)
    assert res.status_code == 400
    assert 'error' in res.json
