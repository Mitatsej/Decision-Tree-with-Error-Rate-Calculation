
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# Ngarkimi i dataset-it (munjna me e ndryshu dataset-in sipas nevojes)
data = load_iris() # ky është një dataset  nga sklearn
X = data.data # Keto janë karakteristikat (features)
y = data.target  # Keto janë etiketat (labels)

# Ndarja e te dhenave në grupet e trajnimit dhe testimit
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Krijimi dhe trajnimi i modelit të pemes vendimmarrese
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Parashikimi i rezultateve mbi të dhenat e testimit
y_pred = model.predict(X_test)

# Llogaritja e saktesiss dhe "Error Rate"
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy

#SHfaqja e rezultit
print(f"Saktësia: {accuracy}")
print(f"Error Rate: {error_rate}")
