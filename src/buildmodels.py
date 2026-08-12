from __future__ import annotations
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from Data_Preprocessing import preprocess, separate_target , scale_columns , read_data

MODEL_BUILDERS = {
    "logistic_regression": lambda: LogisticRegression(max_iter=500, random_state=42 , solver="liblinear"),
    "decision_tree": lambda: DecisionTreeClassifier( random_state=42 , criterion="entropy"),
    "random_forest": lambda: RandomForestClassifier(random_state=42),
    "svc": lambda: SVC( random_state=42),
    "knn": lambda: KNeighborsClassifier(n_neighbors=5)
}
PARAM_GRIDS = {
    "logistic_regression": {
        "C": [0.1, 1, 10]
    },
    "decision_tree": {
        "max_depth": [3, 4, 5]
    },
    "random_forest": {
    "n_estimators": [50, 120, 200],
    "max_depth": [3, 4, 5, None]
    },
    "svc": {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"]
    },
    "knn": {
        "metric":["euclidean" , "manhattan"]
    }
}


def main() -> None:
    train = read_data()
    train = preprocess(train)
    x, y = separate_target(train, "Survived")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42
    )

    x_train,  x_test  = scale_columns(x_train , x_test)

    metrics: dict[str, dict] = {}
    for name, factory in MODEL_BUILDERS.items():
        grid_search = GridSearchCV(
            estimator=factory(),
            param_grid=PARAM_GRIDS[name],
            scoring='accuracy',
            cv=5
        )
        grid_search.fit(x_train, y_train)
        model = grid_search.best_estimator_
        pred = model.predict(x_test)
        acc = float(accuracy_score(y_test, pred))
        metrics[name] = {
            "id": name,
            "display_name": name.replace("_", " ").title(),
            "accuracy": round(acc, 4)
        }

    best_model = None
    best_acc = -1
    for model in metrics.values():
        acc = model["accuracy"]
        if acc > best_acc:
            best_acc = acc
            best_model = model

    print("Best Model:", best_model["display_name"])
    for m in metrics.values():
        print(f"  {m['display_name']}: accuracy={m['accuracy']}")

    return metrics
if __name__ == "__main__":
    main()