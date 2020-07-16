# Tensor Track

Tensor Track is a simple, lightweight wrapper library for TensorFlow that allows the process of 
gathering and recording model parameters, performance, and predictions over several training
iterations to be completely automated and tailored to your liking.  
  
The process is incredibly simple: after installing this library, a few lines of code can be added
to any existing TensorFlow script in order to gather and record model parameters, evaluation metrics,
and more!  
  
# Installation  
```shell script
$ pip install tensor-track
```
  
# Getting Started
Import the library
```python
# Import the library
from tensortrack.ModelParams import ModelParams
from tensortrack.Tracker import Tracker
```
  
Rather than passing parameters into a TensorFlow model, pass them into a dictionary:
  
```python
# Add standard model parameters to a dictionary
model_param_args = {
    "epochs": 50,
    "model": model,
    "callbacks": [model_checkpoint],
    "steps_per_epoch": len(x),
    "loss_func": MeanSquaredError(),
}
```  
Instantiate a ModelParams object using the dictionary:
```python
# Create an instance of the ModelParams class (used to wrap a standard TensorFlow model)
model_param_instance = ModelParams(**model_param_args)
```  
Finally, fit the model and use an instance of the Tracker class to record all desired metrics!
```python
# Fit the model using the parameters from the model_param_args dictionary
model_param_instance.fit_from_params(x, y, validation_data=(x_val, y_val), validation_steps=len(x_val))

# Create an instance of the Tracker class, passing in the desired location for the output directory
tracker = Tracker(model_param_instance, ".")

# Call Tracker methods to record any desired parameters or metrics
tracker.track_all()
tracker.predict_and_evaluate(x_pred, y_true)
```  
  
Result:  
```
project
|   your_script.py
|
|___output/
|   |
|   |___ModelName/
|       |
|       |___model_1
|           |   model_performance.txt
|           |   model_params.txt
|           |   train_curve.png
|           |   validation_curve.png
|           |
|           |___predict/
|               |   predictions.txt
```
  
# License
[MIT](https://github.com/SamClaflin/Tensor-Track/blob/master/LICENSE)
