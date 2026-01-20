import requests
import json 

url = 'http://localhost:9696/predict'

tumor = {
    "radius_mean": 17.99,
    "texture_mean": 10.38,
    "perimeter_mean": 122.8,
    "area_mean": 1001.0,
    "smoothness_mean": 0.1184,
    "compactness_mean": 0.2776,
    "concavity_mean": 0.3001,
    "concave_points_mean": 0.1471,
    "symmetry_mean": 0.2419,
    "fractal_dimension_mean": 0.07871,
    "radius_se": 1.095,
    "texture_se": 0.9053,
    "perimeter_se": 8.589,
    "area_se": 153.4,
    "smoothness_se": 0.006399,
    "compactness_se": 0.04904,
    "concavity_se": 0.05373,
    "concave_points_se": 0.01587,
    "symmetry_se": 0.03003,
    "fractal_dimension_se": 0.006193,
    "radius_worst": 25.38,
    "texture_worst": 17.33,
    "perimeter_worst": 184.6,
    "area_worst": 2019.0,
    "smoothness_worst": 0.1622,
    "compactness_worst": 0.6656,
    "concavity_worst": 0.7119,
    "concave_points_worst": 0.2654,
    "symmetry_worst": 0.4601,
    "fractal_dimension_worst": 0.1189,
}

response = requests.post(url, json=tumor)

print(f"Response status code: {response.status_code}")

if response.status_code == 200:
    try:
        predictions = response.json()
        if predictions.get('malignant'):
            print('tumor is malignant')
        else:
            print('tumor is benign')
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON response for status 200. Raw text: {response.text}")
elif response.status_code == 422:
    print("Validation Error (422):")
    print(json.dumps(response.json(), indent=2)) 
else:
    print(f"Request failed with status code {response.status_code}. Raw text: {response.text}")