# Passengers Traveling Trend

Model to predict the number of traveling passengers

## Goal

Time Series Prediction using LSTM with PyTorch to predict traveling passengers count for future based on historical travel data.

## Background

* Time series data is a type of data that changes with time.
* Advanced deep learning models such as **Long Short Term Memory** (LSTM) Networks are capable of capturing patterns in the time series data, and can be used to make predictions regarding the future trend of the data.
* **PyTorch** library, developed by *Facebook* is one of the most commonly used Python libraries for deep learning.

## Dependencies

* PyTorch
* Pandas
* Numpy
* Matplotlib
* Scikit-learn

`pip install -r requirements.txt`

## Dataset

United States' International Passenger Enplanements data, *serially arranged* month wise from January 2010 to November 2019 taken from the [Bureau of Transportation Statistics](https://www.bts.gov/TRAFFIC).

Saved in *data/USCarrier_Traffic_2010-2019.csv*

## Data preprocessing

The types of the columns in our dataset is object.

* Set data type of the ***Total*** column to *float*

### Split data into training and test sets

Total number of months (no. of records) = 117 = 9.0 years 9 months<br>
So, out training set will be the first 9 years = 108 months data, and the test set will be the last 9 months data.

### Training Data Normalization

The total number of passengers in the initial years is far less compared to the total number of passengers in the later years. So, We normalize the data for time series predictions.  

#### Perform *min/max scaling*

[Ref](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)

* Used min value -1 and max value 1

### Convert the normalized training data to Tensor

* ***PyTorch*** models are trained using tensors.

### Convert the training data into sequences and labels

* Accept raw input data
* Return a list of tuples
  * First element = list of 12 items: the number of passengers traveling in 12 months
  * Second element = one item: the number of passengers in the 12+1st month
* Sequence length / Training window = 12, since there are 12 months
  * Based on domain knowledge
* No. of sequence_labels data = 96
  * Total no. of train data - Length of sequence = 108 - 12 = 96

## Create model

### Define the model

* Create LSTM model class that inherits from the ***Module*** class of **PyTorch**'s **nn** module
* Initialization:
  * **input size**: number of features in the input
    * For each month we have only 1 value for total passengers => input size is 1
  * **hidden_layer_size**: list of neurons in each hidden layer
  * **output size**: number of items in the output
    * To predict no. of passengers for 1 month in future => output size is 1
* Constructor Variables:
  * **hidden_layer_size**: Neurons for each hidden layer
  * **lstm**: LSTM layer; accepts input_size and hidden_layers
  * **linear**: Linear layer
  * **hidden_cell**: previous hidden and cell states
* Passing inputs and predicting:
  * The input_seq is passed to the lstm layer
  * Hidden and cell states are obtain along with the lstm output
  * Lstm output is passed through the linear layer
  * Prediction is obtained and stored

### Prepare the model

* Set loss function: **MSE** loss
* Set optimizer: **Adam**

```
LSTM(
  (lstm): LSTM(1, 200)
  (linear): Linear(in_features=200, out_features=1, bias=True)
)
```

## Train the model

* Set the number of epochs
* Weights are initialized randomly in a *PyTorch* neural network
* Reset optimizer
* Find the hidden and cell states
* Make prediction
* Find loss
* Update weights
* Update gradient for optimizing

```
epoch:   1 loss:    0.11223
epoch:  21 loss:    0.03978
epoch:  41 loss:    0.00793
epoch:  61 loss:    0.00005
epoch:  81 loss:    0.00154
epoch: 101 loss:    0.00958
epoch: 121 loss:    0.00029
epoch: 141 loss:    0.00657
epoch: 161 loss:    0.00015
epoch: 181 loss:    0.00226
epoch: 201 loss:    0.00000
epoch: 221 loss:    0.00003
epoch: 241 loss:    0.00025
epoch: 249 loss:    0.00020
```

## Test the model

* Use the last 12 months values from the training set to use for prediction

### Predict the future 9 months values

* Initially the test input contains 12 items
* These 12 items are used to make prediction for the next item, i.e. the first item from the test set (item 109)
* The predicted value is then appended to the test inputs list
* For the next iteration the last 12 items of the test inputs will be used
* There will be 9 iterations for the 9 predictions
* In the end the test inputs will contaion 12+9 = 21 items, out of which the last 9 items will be the predicted values for the test set.

### Denormalize the predicted values

Since the dataset for training was normalized, the predicted values are also normalized. So, we convert the normalized predicted values into actual predicted values.

### Compare the predicted values against the actual test data

* Create a list of indices for the test set: 108-116 (last 9 records)
* Plot the total number of passengers for the last 9 months, along with the predicted number of passengers for the last 9 months

<img src="last_9_months_prediction.png" alt="last_9_months_prediction" style="float: left; margin-right: 10px;" />

## Conclusion

* LSTM is one of the most widely used algorithm to solve sequence problems.
* We implemented implement LSTM with PyTorch to create a model to make future predictions using time series data.
* Our predictions are not very accurate but the algorithm was able to capture the trend that the number of passengers in the future months should be higher than the previous months with occasional fluctuations.

## Improvements

* Modify or increase the number of epochs.  
* Use more number of neurons in the LSTM layer to check if the model gives better performance.

