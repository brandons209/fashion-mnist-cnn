# Fashion MNIST CNN
## CNN trained on the Fashion MNIST data set
***
The Fashion MNIST data set contains 60,000 training images and 10,000 testing images of 10 different kinds of clothing. It is like the MNIST data set, with pictures 28x28 pixels and 10 classes. This data set is becoming the new standard for testing and teaching people about neural networks.
[LICENSE](LICENSE)
***
### Libraries
For this project, I used the Keras deep learning library with the TensorFlow backend. I also used matplotlib for graphing history from training models. TensorBoard is also used to monitor loss and accuracy during training.
### Model Architecture
This model employs two convolutional layers, followed by a max pooling layer. These layers are connected to a fully connected dense layer of 400 nodes, then to the output layer of 10 nodes, each for the different class of clothing. Dropout is also employed after the max pooling layers and the fully connected hidden layer to help with overfitting. I started with a learning rate of 0.001 and trained for 20 epochs.
### Running the Model
1. Install required libraries in requirements.txt file.
```bash
pip install -r requirements.txt
```
2. Change hyper-parameters and model structure as needed.
```python
#hyperparameters
epochs = 20
batch_size = 64
learn_rate = 0.001
```
3. (Optional) Start TensorBoard to view loss and accuracy graphs as model trains.
```bash
tensorboard --logdir=tensorboard_logs
```
4. Run the model.
```bash
python fashion-mnist.py
```
5. View history graphs with graph_history file.
```bash
python graph_history /path/to/history/file.pkl
```
### Performance
With a decent size network for this data set, I was able to score 93% validation accuracy with 100% top 5 accuracy. This data set is small and relatively simple, so it can be easy to over fit. I was trying to get the highest accuracy possible, so I am going to try to fine tune the model some more. It seems that the model converges around 0.222 validation loss, even with various tweaks and slightly different models. A better model size, perhaps with batch normalization and some other techniques, can get a higher accuracy.
