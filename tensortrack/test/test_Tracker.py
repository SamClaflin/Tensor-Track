from tensortrack.Tracker import Tracker
from tensortrack.ModelParams import ModelParams
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
import os
import shutil

TRAIN_SIZE = 50


def init_data(size=TRAIN_SIZE):
    x = np.random.randint(0, 300, size)
    y = 2 * x
    return x, y


def init_model_params():
    model = Sequential()
    model.add(Dense(1, activation="relu", input_shape=[1, 1]))
    model.add(Dense(1, activation="relu"))
    model_param_args = {
        "epochs": 10,
        "model": model,
        "steps_per_epoch": TRAIN_SIZE,
        "loss_func": MeanSquaredError(),
    }
    return ModelParams(**model_param_args)


def init_tracker():
    x, y = init_data()
    model_params_instance = init_model_params()
    model_params_instance.fit_from_params(x, y)
    tracker = Tracker(model_params_instance, ".")
    return tracker


class TestTracker:
    def test_track_params(self):
        tracker = init_tracker()
        tracker.track_params()
        assert "output" in os.listdir(".")
        assert "Model" in os.listdir("output")
        model_dir = os.path.join("output", "Model")
        assert "model_1" in os.listdir(model_dir)
        model_instance_dir = os.path.join(model_dir, "model_1")
        assert "model_params.txt" in os.listdir(model_instance_dir)
        shutil.rmtree("output")

    def test_plot_loss(self):
        tracker = init_tracker()
        tracker.plot_loss()
        model_dir = os.path.join("output", "Model")
        model_instance_dir = os.path.join(model_dir, "model_1")
        assert "train_curve.png" in os.listdir(model_instance_dir)
        shutil.rmtree("output")

    def test_track_loss(self):
        tracker = init_tracker()
        tracker.track_loss()
        model_dir = os.path.join("output", "Model")
        model_instance_dir = os.path.join(model_dir, "model_1")
        assert "model_performance.txt" in os.listdir(model_instance_dir)
        shutil.rmtree("output")

    def test_make_and_store_predictions(self):
        x, _ = init_data(10)
        tracker = init_tracker()
        tracker.make_and_store_predictions(x)
        model_dir = os.path.join("output", "Model")
        model_instance_dir = os.path.join(model_dir, "model_1")
        assert "predict" in os.listdir(model_instance_dir)
        prediction_dir = os.path.join(model_instance_dir, "predict")
        assert "predictions.txt" in os.listdir(prediction_dir)
        shutil.rmtree("output")
