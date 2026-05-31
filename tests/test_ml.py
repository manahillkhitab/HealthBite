import pytest
import ml_service as ml

def test_validate_nutrition_valid():
    data = {
        'calories': 100,
        'protein': 10,
        'carbs': 5,
        'fat': 2,
        'sugar': 1,
        'sodium': 50,
        'saturated_fat': 0.5
    }
    errors = ml.validate_nutrition(data)
    assert len(errors) == 0

def test_validate_nutrition_negative_calories():
    data = {'calories': -10}
    errors = ml.validate_nutrition(data)
    assert any("must be between 0 and 900" in e for e in errors)

def test_validate_nutrition_too_much_fat():
    data = {'calories': 500, 'fat': 120}
    errors = ml.validate_nutrition(data)
    assert any("between 0 and 100" in e for e in errors)

def test_extract_features():
    data = {
        'calories': '100.5',
        'protein': '10',
        'carbs': 5,
        'fat': 2,
        'sugar': 1,
        'sodium': 50,
        'saturated_fat': 0.5
    }
    features = ml.extract_features(data)
    assert features.shape == (1, 7)
    assert features[0][0] == 100.5
