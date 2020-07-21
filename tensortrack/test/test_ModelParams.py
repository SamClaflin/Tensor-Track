from tensortrack.ModelParams import ModelParams
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.losses import MeanSquaredError
import numpy as np

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
        "pretrained_model": False
    }
    return ModelParams(**model_param_args)


class TestModelParams:
    def test_fit(self):
        x, y = init_data()
        model_params_instance = init_model_params()
        model_params_instance.fit_from_params(x, y)
        assert model_params_instance.history is not None

    def test_make_predictions(self):
        x, y = init_data()
        x_pred, _ = init_data(10)
        model_params_instance = init_model_params()
        model_params_instance.fit_from_params(x, y)
        predictions = model_params_instance.make_predictions(x_pred)
        assert predictions is not None

    def test_evaluate_model(self):
        x, y = init_data()
        x_test, y_test = init_data(10)
        model_params_instance = init_model_params()
        model_params_instance.fit_from_params(x, y)
        loss, acc = model_params_instance.evaluate_model(x_test, y_test)
        assert loss is not None
        assert acc is not None

    def test_load_pretrained_model(self):
        x, y = init_data()
        pretrained_model = "model.hdf5"
        model = load_model(pretrained_model)
        model_param_args = {
            "epochs": 10,
            "model": model,
            "steps_per_epoch": TRAIN_SIZE,
            "loss_func": MeanSquaredError(),
            "pretrained_model": True
        }
        model_params_instance = ModelParams(**model_param_args)
        model_params_instance.fit_from_params(x, y)
        assert model_params_instance.history is not None
