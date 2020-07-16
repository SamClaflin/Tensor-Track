from .ModelParams import ModelParams
from .TrackExceptions import *
import matplotlib.pyplot as plt
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
