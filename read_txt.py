import pandas as pd
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
    # if line.startswith("L1loss") or line.startswith("CrossEntropy") or line.startswith("FocalLoss0.5") or line.startswith("FocalLoss2"):
    if line.startswith("l1loss"):
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
df.to_csv('/data/wuyue_2023/code/Trail_results.csv')

# Create separate figures for each loss function for Accuracy
output_directory = "/data/wuyue_2023/code/loss_function_acc_figures_sup"
os.makedirs(output_directory, exist_ok=True)

sns.set(style="whitegrid")
for loss_function, loss_data in df.groupby("LossFunction"):
    plt.figure()

    # Loss plot
    for metric, linestyle in [("LossTrain", "-"), ("LossTest", "--")]:
        for trial, trial_data in loss_data.groupby("Trial"):
            trial_color = trial_colors[trial_data["Trial"].values[0]]
            plt.plot(trial_data["Epoch"], trial_data[metric], label=f"{metric} (Trial {trial})", color=trial_color, linestyle=linestyle)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"{loss_function} - Loss")

    plt.savefig(os.path.join(output_directory, f"{loss_function}_loss.png"))
    plt.close()

sns.set(style="whitegrid")
for loss_function, loss_data in df.groupby("LossFunction"):
    plt.figure()

    # Accuracy plot
    for metric, linestyle in [("AccTrain", "-"), ("AccTest", "--")]:
        for trial, trial_data in loss_data.groupby("Trial"):
            trial_color = trial_colors[trial_data["Trial"].values[0]]
            plt.plot(trial_data["Epoch"], trial_data[metric], label=f"{metric} (Trial {trial})", color=trial_color, linestyle=linestyle)

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title(f"{loss_function} - Accuracy")

    plt.savefig(os.path.join(output_directory, f"{loss_function}_accuracy.png"))
    plt.close()

print(f"Figures saved in the 'loss_function_figures' and 'loss_function_figures_acc' directories.")







