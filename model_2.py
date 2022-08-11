from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, PReLU
from pandas import read_csv
from keras import callbacks
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sequence_split import sequence_split

# Loading the dataset with header set to '0' to avoid the string
# headers in the first row
df = read_csv('energy_consumption_levels (1).csv', header=0)

# Shape of dataset:
print("Shape of df: ", df.shape, '\n')

# Get values of consumption
values = df[['consumption','hour_of_day']].values.astype('float32')

# Dataset contained hourly readings,therefore window size reflects data for a whole
# day making the number of time steps within each row 24
n_steps = 24

# Split values into samples
X, y = sequence_split(values, n_steps)

# Reshape X for lstm
X = X.reshape(X.shape[0], X.shape[1], 2)
print("x:", X.shape, "y:", y.shape, '\n')

# Initialise the training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle=False)
print("x_train:", X_train.shape, "y_train:", y_train.shape,
      "x_test:", X_test.shape, "y_test:", y_test.shape, '\n')

# Define the model
model = Sequential()

# Add LSTM with multiple inputs & outputs
# input shape is the window size plus number of features
model.add(LSTM(X.shape[1], return_sequences=True, dropout=0.2,
               input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(X.shape[1], activation=PReLU()))
model.add(Dense(X.shape[1], activation=PReLU()))
model.add(Dense(X.shape[2], activation='tanh'))

# Compile the model
model.compile(loss='mae', optimizer='adam', metrics=['mae', 'mse', 'accuracy'])

# Print summary of model
model.summary()

# Fit the model
early_stopping = callbacks.EarlyStopping(monitor="val_loss",  mode="min",
                                         patience=5, restore_best_weights=True)
fit = model.fit(X_train, y_train, epochs=500, batch_size=65, verbose=2,
                validation_data=(X, y), callbacks=[early_stopping])

# Evaluate the model
loss, mae, mse, accuracy = model.evaluate(X_test, y_test, verbose=2)
print('Loss: %f, MAE: %f, MSE: %f, Accuracy: %f' % (loss, mae, mse, accuracy*100 ))

# Plot metrics graph
plt.plot(fit.history['mae'], color='red', label="MAE")
plt.plot(fit.history['mse'], color='green', label="MSE")
plt.plot(fit.history['accuracy'], color='blue', label="Accuracy")
plt.grid(True)
plt.legend()
plt.show()

# Make prediction based on X
data = model.predict(X)

# Store the predicted power consumtption values from predictions
consumption_predictions = [i[0] for item in data for i in item]

# Get actual consumption data from the dataset
consumption_data_real = [i[3] for i in df.to_numpy().tolist()]

# X & Y axis for graph showing predicted power comsumption over next
# 72 hours and comparing against the last 72 hours from the dataset
hours = list(range(1, 72+1))
plt.rcParams['axes.formatter.useoffset'] = False
plt.plot(hours, consumption_predictions[:72], color='red', marker="o", label="Predicted")
plt.plot(hours, consumption_data_real[-72:], color='green', marker="o", label="Actual")
plt.title('Predicted power consumption - Model II', fontsize=12)
plt.xlabel('Hour count', fontsize=12)
plt.ylabel(' consumption', fontsize=12)
plt.grid(True)
plt.legend()
plt.show()

# Save the model
# model.save('model_2')




