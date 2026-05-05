from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import (
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    MaxPooling1D,
)
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils import compute_class_weight


class CNN:
    def __init__(
        self,
        num_classes: int = 7,
        epochs: int = 25,
        checkpoint_dir: Optional[str] = None,
        kernel_size: int = 5,
    ):
        self.num_classes = num_classes
        self.epochs = epochs
        self.results = []
        self.history_logs = []
        self.checkpoint_dir = (
            Path(checkpoint_dir) if checkpoint_dir else Path("models")
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.kernel_size = kernel_size

    def build_model(
        self,
        input_shape: Sequence[int],
        filters_per_layer: List[int],
        lr: float,
    ):
        K.clear_session()
        model = Sequential()
        model.add(Input(shape=tuple(input_shape)))

        for i, f in enumerate(filters_per_layer):
            model.add(
                Conv1D(
                    f,
                    self.kernel_size,
                    padding="same",
                    activation="relu",
                )
            )
            model.add(MaxPooling1D(2))

        model.add(GlobalAveragePooling1D())
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation="softmax"))

        model.compile(
            optimizer=Adam(learning_rate=lr),
            loss="sparse_categorical_crossentropy",
            metrics=["sparse_categorical_accuracy"],
        )
        return model

    def train_and_evaluate(
        self,
        feature_name,
        X_train,
        y_train,
        X_dev,
        y_dev,
        X_test,
        y_test,
        cnn_configs,
        batch_sizes,
        learning_rates,
    ):
        best_f1 = 0.0
        best_history = None
        best_y_test = None
        best_y_pred = None

        class_weights = self.compute_class_weights(y_train)
        in_shape = X_train.shape[1:]

        for filters_per_layer in cnn_configs:
            for batch_size in batch_sizes:
                for lr in learning_rates:
                    model = self.build_model(in_shape, filters_per_layer, lr)

                    checkpoint_path = (
                        self.checkpoint_dir
                        / f"best_cnn_{feature_name}_val.keras"
                    )
                    callbacks = [
                        EarlyStopping(
                            monitor="val_loss",
                            patience=5,
                            restore_best_weights=True,
                        ),
                        ModelCheckpoint(
                            filepath=str(checkpoint_path),
                            monitor="val_loss",
                            save_best_only=True,
                        ),
                    ]

                    history = model.fit(
                        X_train,
                        y_train,
                        validation_data=(X_dev, y_dev),
                        epochs=self.epochs,
                        class_weight=class_weights,
                        batch_size=batch_size,
                        callbacks=callbacks,
                        verbose=0,
                    )

                    model.load_weights(str(checkpoint_path))
                    y_pred_probs = model.predict(X_test, verbose=0)
                    y_pred = np.argmax(y_pred_probs, axis=1)

                    acc = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average="weighted")

                    self.results.append(
                        {
                            "Feature": feature_name,
                            "Filters": str(filters_per_layer),
                            "Batch Size": batch_size,
                            "Learning Rate": lr,
                            "Accuracy": acc,
                            "F1": f1,
                        }
                    )

                    print(
                        f"[CNN] accuracy {acc:.4f} F1 {f1:.4f} "
                        f"config=[{filters_per_layer},{batch_size},{lr}]"
                    )

                    if f1 > best_f1:
                        best_f1 = f1
                        best_history = history.history
                        best_y_test = y_test
                        best_y_pred = y_pred
                        model.save(
                            str(
                                self.checkpoint_dir
                                / f"best_cnn_{feature_name}.keras"
                            )
                        )

        return best_history, best_y_test, best_y_pred

    def compute_class_weights(self, y):
        classes = np.unique(y)
        weights = compute_class_weight(
            "balanced", classes=classes, y=y
        )
        return dict(zip(classes, weights))
