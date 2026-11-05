"""Task 2.7 (toy part) — Keras Tuner random search on Fashion-MNIST.
Searches learning rate and number of hidden units; prints a results table.
Run:  python labs/keras_tuner_toy.py
"""
import keras
import keras_tuner as kt
import pandas as pd

keras.utils.set_random_seed(42)
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train, x_test = x_train[:20000] / 255.0, x_test / 255.0  # subset keeps the lab fast
y_train = y_train[:20000]


def build(hp):
    model = keras.Sequential([
        keras.Input((28, 28)),
        keras.layers.Flatten(),
        keras.layers.Dense(hp.Int("units", 32, 512, step=32), activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(keras.optimizers.Adam(hp.Float("lr", 1e-4, 1e-2, sampling="log")),
                  "sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


tuner = kt.RandomSearch(build, objective="val_accuracy", max_trials=10, seed=42,
                        directory="labs/_tuner", project_name="fashion_toy", overwrite=True)
tuner.search(x_train, y_train, epochs=5, validation_split=0.2, verbose=0)

rows = [{"trial": t.trial_id, "units": t.hyperparameters.get("units"),
         "lr": round(t.hyperparameters.get("lr"), 5), "val_accuracy": round(t.score, 4)}
        for t in tuner.oracle.get_best_trials(num_trials=10)]
print(pd.DataFrame(rows).to_markdown(index=False))
best = tuner.get_best_hyperparameters(1)[0]
print(f"\nBest: units={best.get('units')} lr={best.get('lr'):.5f}")
model = tuner.get_best_models(1)[0]
print("Test accuracy of best model:", round(model.evaluate(x_test, y_test, verbose=0)[1], 4))
