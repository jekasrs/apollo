from pathlib import Path
from typing import Optional

import numpy as np
from keras import backend as K
from keras import Input
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from sklearn.utils import compute_class_weight
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, f1_score


class DNN:
    def __init__(self, num_classes: int = 7, epochs: int = 25, checkpoint_dir: Optional[str] = None):
        self.num_classes = num_classes
        self.epochs = epochs
        self.results = []
        self.history_logs = []
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("models")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def build_model(self, input_dim, neurons, lr):
        K.clear_session()

        model = Sequential()
        model.add(Input(shape=(input_dim,)))
        model.add(Dense(neurons[0], activation='relu'))
        model.add(Dropout(0.5))
        for n in neurons[1:]:
            model.add(Dense(n, activation='relu'))
            model.add(Dropout(0.3))

        model.add(Dense(self.num_classes, activation='softmax'))

        model.compile(
            optimizer=Adam(learning_rate=lr),
            loss='sparse_categorical_crossentropy',
            metrics=['sparse_categorical_accuracy']
        )
        return model

    def train_and_evaluate(self, feature_name, X_train, y_train, X_dev, y_dev, X_test, y_test,
                           neuron_configs, batch_sizes, learning_rates):

        best_f1 = 0
        best_history = None
        best_y_test = None
        best_y_pred = None

        class_weights = self.compute_class_weights(y_train)

        for neurons in neuron_configs:
            for batch_size in batch_sizes:
                for lr in learning_rates:

                    model = self.build_model(X_train.shape[1], neurons, lr)

                    checkpoint_path = self.checkpoint_dir / f'best_dnn_{feature_name}_val.keras'
                    callbacks = [
                        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                        ModelCheckpoint(filepath=str(checkpoint_path), monitor='val_loss', save_best_only=True)
                    ]

                    history = model.fit(
                        X_train, y_train,
                        validation_data=(X_dev, y_dev),
                        epochs=self.epochs,
                        class_weight=class_weights,
                        batch_size=batch_size,
                        callbacks=callbacks,
                        verbose=0
                    )

                    model.load_weights(str(checkpoint_path))
                    y_pred_probs = model.predict(X_test, verbose=0)
                    y_pred = np.argmax(y_pred_probs, axis=1)

                    acc = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average='weighted')

                    self.results.append({
                        'Feature': feature_name,
                        'Neurons': str(neurons),
                        'Batch Size': batch_size,
                        'Learning Rate': lr,
                        'Accuracy': acc,
                        'F1': f1
                    })

                    if f1 > best_f1:
                        best_f1 = f1
                        best_history = history.history
                        best_y_test = y_test
                        best_y_pred = y_pred
                        model.save(str(self.checkpoint_dir / f'best_dnn_{feature_name}.keras'))

        return best_history, best_y_test, best_y_pred


    def compute_class_weights(self, y):
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        return dict(zip(classes, weights))
