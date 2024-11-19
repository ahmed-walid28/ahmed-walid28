import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

titanic = sns.load_dataset('titanic')

encoder = LabelEncoder()
titanic['sex_encoded'] = encoder.fit_transform(titanic['sex'])
titanic['alive_encoded'] = encoder.fit_transform(titanic['alive'])

features = titanic[['sex_encoded', 'alive_encoded', 'pclass']]
target = titanic['survived']  # Assuming we want to predict survival

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model accuracy on the test set:", accuracy)
