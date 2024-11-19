import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

penguins = sns.load_dataset('penguins')
penguins = penguins.dropna()

encoder = LabelEncoder()
penguins['species_encoded'] = encoder.fit_transform(penguins['species'])

X = penguins[['flipper_length_mm', 'body_mass_g']]
y = penguins['species_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
comparison_df = pd.DataFrame({'Actual': encoder.inverse_transform(y_test), 'Predicted': encoder.inverse_transform(y_pred)})
print(comparison_df.head())
