import keras.models
from pandas import read_csv
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

# Load the models
model_1 = keras.models.load_model('model_1')
model_2 = keras.models.load_model('model_2')

# Make the predictions against the data
model_1_data = model_1.predict(X)
model_2_data = model_2.predict(X)

# Plot a graph for comparison
m1_consumption_predictions = [i[0] for item in model_1_data for i in item]
m2_consumption_predictions = [i[0] for item in model_2_data for i in item]

# Get actual consumption data from the dataset
consumption_data_real = [i[3] for i in df.to_numpy().tolist()]

# X & Y axis for graph showing predicted power comsumption over next
# 72 hours and comparing against the last 72 hours from the dataset
hours = list(range(1, 72+1))
plt.rcParams['axes.formatter.useoffset'] = False
plt.plot(hours, m1_consumption_predictions[:72], color='red', marker="o", label="Model I")
plt.plot(hours, m2_consumption_predictions[:72], color='green', marker="o", label="Model II")
plt.plot(hours, consumption_data_real[-72:], color='blue', marker="o", label="Actual")
plt.title('Predicted power consumption - Model II', fontsize=12)
plt.xlabel('Hour count', fontsize=12)
plt.ylabel(' consumption', fontsize=12)
plt.grid(True)
plt.legend()
plt.show()

