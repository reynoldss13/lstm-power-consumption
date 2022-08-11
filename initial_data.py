from pandas import read_csv
import matplotlib.pyplot as plt
from tabulate import tabulate

# Loading the dataset with header set to '0' to avoid the string
# headers in the first row
df = read_csv('energy_consumption_levels (1).csv')
print(tabulate(df[:72], headers='keys', tablefmt='fancy_grid', showindex=True))


# Retrieve the consumption values as a list.
consumption_data = [i[3] for i in df.to_numpy().tolist()]

# Plot graph detailing week of power consumption
consumption_week = consumption_data[:168]
hours_week = lst = list(range(72, 168+1))
plt.plot(hours_week, consumption_week, color='red', marker="o", label="Reading point")
plt.title('Power consumption over 1st week of January 2016', fontsize=12)
plt.xlabel('Hour count', fontsize=12)
plt.ylabel('Consumption reading', fontsize=12)
plt.grid(True)
plt.legend()
plt.show()

# Plot graph detailing power consumption for last 3 days of 2016
consumption_week = consumption_data[-72:]
hours_week = lst = list(range(1, 72+1))
plt.plot(hours_week, consumption_week, color='red', marker="o", label="Reading point")
plt.title('Power consumption over last 3 days of 2016', fontsize=12)
plt.xlabel('Hour count', fontsize=12)
plt.ylabel('Consumption reading', fontsize=12)
plt.grid(True)
plt.legend()
plt.show()




