from .ModelParams import ModelParams
from .TrackExceptions import *
import matplotlib.pyplot as plt
import numpy as np
import os


class Tracker:
    """
    Encapsulates all tracking/plotting operations for TensorFlow.
    """
    def __init__(self, model_params: ModelParams, root_dir):
        self.model_params = model_params
        self.root_dir = root_dir
        self.model_number = 1
        self.model_instance_dir = None

    # Private Methods
    def __gen_output_structure(self):
        output_dir = os.path.join(self.root_dir, "output")
        if "output" not in os.listdir(self.root_dir):
            os.mkdir(output_dir)

        model_dir = os.path.join(output_dir, self.model_params.model_name)
        if self.model_params.model_name not in os.listdir(output_dir):
            os.mkdir(model_dir)

        if not self.model_instance_dir:
            self.__get_model_number(model_dir)
            temp_dir = f"model_{self.model_number}"
            self.model_instance_dir = os.path.join(model_dir, temp_dir)
            os.mkdir(self.model_instance_dir)

    def __get_model_number(self, model_dir):
        self.model_number += len(os.listdir(model_dir))

    def __base_performance_tracker(self, training_phase, loss_key=None, acc_key=None, test_loss=None, test_acc=None):
        self.__gen_output_structure()
        try:
            history = self.model_params.history
            if not history:
                raise NoHistory

            if loss_key and acc_key:
                loss = history[loss_key]
                loss = loss[len(loss) - 1]
                accuracy = history[acc_key]
                accuracy = accuracy[len(accuracy) - 1]
            else:
                loss = test_loss
                accuracy = test_acc

            model_performance_file = os.path.join(self.model_instance_dir, "model_performance.txt")
            with open(model_performance_file, "a") as f:
                f.write(f"=== {self.model_params.model_name} {self.model_number}: {training_phase} ===\n")
                f.write(f"Final Loss: {loss}\n")
                f.write(f"Final Accuracy: {accuracy}\n\n")

        except NoHistory:
            print("Error: No history object to derive plot from. "
                  "Call fit_from_params on a ModelParams instance first.")
        except KeyError:
            print(f"Error: Model history does not contain requested data: {loss_key}")

    def __base_plot(self, loss_key, acc_key, title, output_file, loss_label, acc_label):
        self.__gen_output_structure()
        try:
            history = self.model_params.history
            if not history:
                raise NoHistory
            val_loss = history[loss_key]
            val_acc = history[acc_key]
            epochs = [i for i in range(1, len(val_loss) + 1)]

            plt.figure(figsize=(8, 8))
            plt.title(title, fontsize=20)
            plt.xlabel("Epochs", fontsize=18)
            plt.ylabel("Accuracy", fontsize=18)
            plt.grid()
            plt.plot(epochs, val_loss, lw=2, label=loss_label, color="#c93a0a")
            plt.plot(epochs, val_acc, lw=2, label=acc_label, color="#0384fc")
            plt.legend(loc=0)
            plt.savefig(os.path.join(self.model_instance_dir, output_file))

        except NoHistory:
            print("Error: No history object to derive plot from. "
                  "Call fit_from_params on a ModelParams instance first.")
        except KeyError:
            print(f"Error: Model history doesn't contain requested data: {loss_key}")

    # Public Methods
    def track_all(self):
        self.track_params()
        self.plot_loss()
        self.plot_val_loss()
        self.track_loss()
        self.track_val_loss()

    def predict_and_evaluate(self, x, y=None, verbose=1, callbacks=None, batch_size=None, output_format="txt",
                             multi_class_labels=None):
        self.make_and_store_predictions(x, verbose, callbacks, batch_size, output_format, multi_class_labels)
        self.evaluate_and_track_test(x, y, verbose, callbacks, batch_size)

    def track_params(self):
        self.__gen_output_structure()
        model_params_file = os.path.join(self.model_instance_dir, "model_params.txt")
        with open(model_params_file, "w") as f:
            f.write(f"=== {self.model_params.model_name} {self.model_number} ===\n")
            f.write(f"Learning Rate: {self.model_params.lr}\n")
            f.write(f"Epochs: {self.model_params.epochs}\n")
            f.write(f"Steps Per Epoch: {self.model_params.steps_per_epoch}\n")
            f.write(f"Loss Function: {self.model_params.loss_func}\n")
            f.write(f"Optimizer: {self.model_params.optimizer}\n")
            f.write(f"Callbacks: {self.model_params.callbacks}\n")
            f.write(f"Batch Size: {self.model_params.batch_size}\n\n")

    def plot_loss(self):
        self.__base_plot("loss", "accuracy", "Train Learning Curve", "train_curve.png", "Train Loss", "Train Accuracy")

    def plot_val_loss(self):
        self.__base_plot("val_loss", "val_accuracy", "Validation Learning Curve", "validation_curve.png",
                         "Validation Loss", "Validation Accuracy")

    def track_loss(self):
        self.__base_performance_tracker("Train", "loss", "accuracy")

    def track_val_loss(self):
        self.__base_performance_tracker("Validation", "val_loss", "val_accuracy")

    def evaluate_and_track_test(self, x, y=None, verbose=1, callbacks=None, batch_size=None):
        self.__gen_output_structure()
        try:
            test_loss, test_acc = self.model_params.evaluate_model(x, y, verbose, callbacks, batch_size)
            self.__base_performance_tracker("Test", test_loss=test_loss, test_acc=test_acc)

        except NoHistory:
            print("Error: model must first be trained before conducting evaluation.")

    def make_and_store_predictions(self, x, verbose=1, callbacks=None, batch_size=None, output_format="txt",
                                   multi_class_labels=None, cmap="gray"):
        self.__gen_output_structure()
        text_formats = ["txt", "csv"]
        image_formats = ["png", "jpg", "jpeg"]

        try:
            predictions = self.model_params.make_predictions(x, verbose, callbacks, batch_size)
            predictions_dir = os.path.join(self.model_instance_dir, "predict")
            os.mkdir(predictions_dir)

            output_format = output_format.lower()

            if output_format in text_formats:
                output_file = os.path.join(predictions_dir, f"predictions.{output_format}")
                with open(output_file, "w") as f:
                    for prediction in predictions:
                        if multi_class_labels:
                            prediction = list(prediction)
                            predicted_index = prediction.index(max(prediction))
                            f.write(f"{multi_class_labels[predicted_index]}\n")
                        else:
                            f.write(f"{prediction}\n")
                    f.write("\n")

            elif output_format in image_formats:
                count = 0
                fig = plt.figure(frameon=False)
                for prediction in predictions:
                    count += 1
                    prediction = np.squeeze(prediction)
                    ax = fig.Axes(fig, [0., 0., 1., 1.])
                    ax.set_axis_off()
                    fig.add_axes(ax)
                    ax.imshow(prediction, cmap=cmap)
                    fig.savefig(os.path.join(predictions_dir, f"predict_{count}.{output_format}"))

            else:
                raise InvalidFormat

        except NoHistory:
            print("Error: model must first be trained before making predictions.")
        except InvalidFormat:
            print(f"Error: file format '{output_format}' is either invalid or not supported.")
            print("Please select one of the following formats:")
            print(f"Text data: {text_formats}")
            print(f"Image data: {image_formats}")
        except IndexError:
            print("Index Error: Invalid class labels provided")
