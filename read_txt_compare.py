import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Read data from the text file
with open("/data/wuyue_2023/code/results_sup.txt", "r") as file:
    lines = file.readlines()

# Define a list to store data for each loss function and trial
data = []
current_loss_function = None
trial_colors = {}  # Dictionary to map trials to unique colors

# Process the data into a list of dictionaries
for line in lines:
    if line.startswith("L1loss") or line.startswith("CrossEntropy") or line.startswith("FocalLoss0.5") or line.startswith("FocalLoss2"):
        current_loss_function = line.strip().split()[0]
    else:  # Skip empty lines
        parts = line.split(":")
        if parts[0].strip().startswith("Trial"):
            trial, metrics = parts[0].strip(), parts[1].strip()
            trial_parts = trial.split()
            trial_num = int(trial_parts[1])
            epoch_num=int(trial_parts[3])
            epoch_parts = metrics.split(", ")
            epoch_parts_0=epoch_parts[0].split("=")
            epoch_parts_1=epoch_parts[1].split("=")
            epoch_parts_2=epoch_parts[2].split("=")
            epoch_parts_3=epoch_parts[3].split("=")
            loss_train = float(epoch_parts_0[1])
            acc_train = float(epoch_parts_1[1].strip('%'))
            loss_test = float(epoch_parts_2[1])
            acc_test = float(epoch_parts_3[1].strip('%'))

            # Map each trial to a unique color
            if trial_num not in trial_colors:
                trial_colors[trial_num] = plt.cm.tab20(trial_num)
            trial_color = trial_colors[trial_num]

            data.append({"LossFunction": current_loss_function, "Trial": trial_num, "Epoch": epoch_num, "LossTrain": loss_train, "AccTrain": acc_train, "LossTest": loss_test, "AccTest": acc_test})

# Create a DataFrame
df = pd.DataFrame(data)
df.to_csv('/data/wuyue_2023/code/Trail_results_sup.csv')

# output_directory = "/data/wuyue_2023/code/mean_compare"
# os.makedirs(output_directory, exist_ok=True)

# mean_loss_accuracy = {}
# for loss_function in df['LossFunction'].unique():
#     mean_loss_accuracy[loss_function] = df.groupby(['Epoch', 'LossFunction'])['LossTrain'].mean()
#     mean_loss_accuracy[loss_function + '_accuracy'] = df.groupby(['Epoch', 'LossFunction'])['AccTrain'].mean()
# mean_loss_accuracy

# fig, axs = plt.subplots(2, 1, sharex=True)

# for loss_function, ax in zip(mean_loss_accuracy.keys(), axs):
#     for i, trial in enumerate(df['Trial'].unique()):
#         ax.plot(mean_loss_accuracy[loss_function][trial], color=colors[i], label=trial)

#     ax.set_xlabel('Epoch')
#     ax.set_ylabel(f'Mean {loss_function}')

#     ax.legend()

# plt.tight_layout()
# plt.show()

# plt.savefig(os.path.join(output_directory, "compare_line.png"))
# plt.close()








