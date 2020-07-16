from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam


class ModelParams:
    """
    Encapsulates all parameters required by a TensorFlow model in order to easily record them later.
    """
    def __init__(self, epochs, model: Model, lr=1e-3, loss_func=None, steps_per_epoch=None, batch_size=None,
                 callbacks=None, optimizer=Adam, model_name="Model"):
        self.lr = lr
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.loss_func = loss_func
        self.callbacks = callbacks
        self.optimizer = optimizer(lr=self.lr)
        self.model = model
        self.history = None
        self.model_name = model_name

    # Configure and compile a pre-existing model instance
    def __gen_model(self):
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_func,
            metrics=["accuracy"]
        )

    # Train the model using attributes of the ModelParams instance
    def fit_from_params(self, x, y=None, verbose=1, validation_data=None, validation_steps=None):
        self.__gen_model()
        history = self.model.fit(
            x=x,
            y=y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=verbose,
            callbacks=self.callbacks,
            validation_data=validation_data,
            validation_steps=validation_steps,
        )

        self.history = history.history

    # Print a summary of the model attribute
    def print_summary(self):
        print(self.model.summary())
