from .ModelParams import  ModelParams
import os


class Tracker:
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
