###Israa_code 

#Ce TP porte sur la mise en place d'un pipeline complet de Machine Learning.
#  L'objectif est de simuler des données hospitalières, de les analyser visuellement,
#  puis d'entraîner plusieurs modèles de classification pour prédire la gravité de 
# l'état des patients.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_hospital_data(n_samples=1000):
    np.random.seed(42)
    data = {
        'Patient_ID': range(1, n_samples + 1),
        'Sexe': np.random.binomial(1, 0.5, n_samples), # Bernoulli
        'Age': np.random.uniform(20, 80, n_samples).round(1), # Uniforme
        'Pression_Artérielle': np.random.normal(120, 15, n_samples).round(1), # Normale
        'Gravité': np.random.choice(['Faible', 'Moyenne', 'Élevée'], n_samples, p=[0.5, 0.3, 0.2])
    }
    return pd.DataFrame(data)

df = generate_hospital_data()
print(df.head())

def visualize_data(df):
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    sns.histplot(df['Age'], kde=True, color='blue')
    plt.title('Distribution de l\'Âge')

    plt.subplot(2, 2, 2)
    sns.countplot(x='Gravité', data=df, palette='viridis')
    plt.title('Répartition de la Gravité')
    
    plt.tight_layout()
    plt.savefig('data_visualization.png')
    plt.show()

visualize_data(df)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Encoding
le = LabelEncoder()
df['Gravité'] = le.fit_transform(df['Gravité'])

X = df[['Sexe', 'Age', 'Pression_Artérielle']]
y = df['Gravité']

# Splitting
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Normalisation
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.metrics import confusion_matrix, classification_report


def plot_confusion(model, name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
    plt.title(f'Matrice de Confusion : {name}')
    plt.savefig('confusion_matrix.png')
    plt.show()

# Exemple pour KNN
plot_confusion(KNeighborsClassifier(), "KNN")
