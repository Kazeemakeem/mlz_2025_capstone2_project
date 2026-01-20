import pickle

import pandas as pd
import numpy as np
import sklearn


from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier


print(f'pandas=={pd.__version__}')
print(f'numpy=={np.__version__}')
print(f'sklearn=={sklearn.__version__}')

def load_data():

    df = pd.read_csv('breast-cancer.csv')

    df.columns = df.columns.str.lower().str.replace(' ', '_')
    df.columns = df.columns.str.lower().str.replace('-', '_')
    
    df['diagnosis'] = df['diagnosis'].str.strip()
    df['diagnosis'] = (df['diagnosis'] == "M").astype(int)

    return df

def train_model(df):
    
	features = ['radius_mean', 'texture_mean', 'perimeter_mean',
			'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
			'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
			'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
			'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se',
			'fractal_dimension_se', 'radius_worst', 'texture_worst',
			'perimeter_worst', 'area_worst', 'smoothness_worst',
			'compactness_worst', 'concavity_worst', 'concave_points_worst',
			'symmetry_worst', 'fractal_dimension_worst']
			
	y_train = df.diagnosis
	
	train_dict = df[features].to_dict(orient='records')

	dv = DictVectorizer(sparse=False)
	X_train = dv.fit_transform(train_dict)


	rf = RandomForestClassifier(n_estimators=100,
			max_depth=20,
			min_samples_leaf=3,
			random_state=1)
	
	model = rf.fit(X_train, y_train)

	return dv, model


def save_model(dv, model, output_file):
    with open(output_file, 'wb') as f_out:
        pickle.dump({'vectorizer': dv, 'model': model}, f_out)



df = load_data()
dv, model = train_model(df)
save_model(dv, model, 'model.bin')

print('Model saved to model.bin')