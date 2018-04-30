import numpy as np
import pickle
import matplotlib.pyplot as plt

# Import train, validation, and test image-like GPS segments
filename = '../Mode-codes-Revised/paper2_data_for_DL_train_val_test.pickle'
with open(filename, 'rb') as f:
    Train_X, Train_Y, Val_X, Val_Y, Val_Y_ori, Test_X, Test_Y, Test_Y_ori, X_unlabeled = pickle.load(f)

# Sum the number of GPS segments for each mode and number of unlabeled trajectories.
num_labels = 5
mode_distribution = []
Train_Y_ori = np.argmax(Train_Y, axis=1)
for i in range(num_labels):
    mode_distribution.append(len(np.where(Train_Y_ori == i)[0]) + len(np.where(Val_Y_ori == i)[0])
                             + len(np.where(Test_Y_ori == i)[0]))
mode_distribution.append(len(X_unlabeled))
print(mode_distribution)

# Plot pie chart for mode distribution
labels = 'Walk', 'Bike', 'Bus', 'Driving(car&taxi)', 'Train', 'Unlabeled'
sizes = mode_distribution
colors = ['gold', 'darkorchid', 'yellowgreen', 'lightcoral', 'lightskyblue', 'gray']
explode = (0, 0, 0, 0, 0, 0.1)

plt.figure(1)
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.title('Distribution of Transportation Modes Among All GPS Segments')
plt.tight_layout()
plt.show()