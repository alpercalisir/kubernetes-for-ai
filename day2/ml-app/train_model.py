from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the classic Iris dataset
data = load_iris()
X, y = data.data, data.target

# Train a random forest classifier
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y)

# Serialize the trained model to disk
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved as model.pkl")
print(f"Training accuracy: {model.score(X, y):.2f}")