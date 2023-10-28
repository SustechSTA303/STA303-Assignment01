import matplotlib.pyplot as plt
import json
import os
import shutil
import numpy as np
import seaborn as sns




def Plot(NUM_EPOCHS, Results_Dir, Image_Dir, epochs_to_plot):
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


    with open(Results_Dir, 'r') as file:
        results = json.load(file)

    lineStyles = ['-', '-.', '--', ':']
    if os.path.exists(Image_Dir):
        shutil.rmtree(Image_Dir)

    os.makedirs(Image_Dir)
    
#     #Test
#     fig, axes = plt.subplots(figsize=(8, 6))
#     for i, (loss_name, loss_values) in enumerate(results.items()):
#         training_loss = loss_values['training_loss']
#         x = range(1, NUM_EPOCHS + 1)
#         axes.plot(x, training_loss, label=loss_name, linestyle=lineStyles[i])
#     axes.set_xlabel('Epochs')
#     axes.set_ylabel('Training Loss')
# #     axes.set_title('Training Loss vs Epochs')
#     axes.legend()
#     axes.autoscale(enable=True, axis='both', tight=True)

#     plt.tight_layout()
#     plt.savefig(f'{Image_Dir}/test.png')
#     plt.close()
    
    # 训练损失
    fig, axes = plt.subplots(figsize=(8, 6))
    for i, (loss_name, loss_values) in enumerate(results.items()):
        training_loss = loss_values['training_loss']
        x = range(1, NUM_EPOCHS + 1)
        axes.plot(x, training_loss, label=loss_name, linestyle=lineStyles[i])
    axes.set_xlabel('Epochs')
    axes.set_ylabel('Training Loss')
#     axes.set_title('Training Loss vs Epochs')
    axes.legend()
    axes.autoscale(enable=True, axis='both', tight=True)

    plt.tight_layout()
    plt.savefig(f'{Image_Dir}/training_loss_plot.png')
    plt.close()

    # 测试损失
    fig, axes = plt.subplots(figsize=(8, 6))
    for i, (loss_name, loss_values) in enumerate(results.items()):
        testing_loss = loss_values['testing_loss']
        x = range(1, NUM_EPOCHS + 1)
        axes.plot(x, testing_loss, label=loss_name, linestyle=lineStyles[i])
    axes.set_xlabel('Epochs')
    axes.set_ylabel('Testing Loss')
#     axes.set_title('Testing Loss vs Epochs')
    axes.legend()
    axes.autoscale(enable=True, axis='both', tight=True)

    plt.tight_layout()
    plt.savefig(f'{Image_Dir}/testing_loss_plot.png')
    plt.close()

    # 训练准确
    fig, axes = plt.subplots(figsize=(8, 6))
    for i, (loss_name, loss_values) in enumerate(results.items()):
        training_acc = loss_values['training_acc']
        x = range(0, NUM_EPOCHS + 1)
        axes.plot(x, training_acc, label=loss_name, linestyle=lineStyles[i])

    axes.set_xlabel('Epochs')
    axes.set_ylabel('Training Accuracy')
#     axes.set_title('Training Accuracy vs Epochs')
    axes.legend()
    axes.autoscale(enable=True, axis='both', tight=True)
    plt.tight_layout()
    plt.savefig(f'{Image_Dir}/training_acc_plot.png')
    plt.close()

    # 测试准确
    fig, axes = plt.subplots(figsize=(8, 6))
    for i, (loss_name, loss_values) in enumerate(results.items()):
        testing_acc = loss_values['testing_acc']
        x = range(0, NUM_EPOCHS + 1)
        axes.plot(x, testing_acc, label=loss_name, linestyle=lineStyles[i])

    axes.set_xlabel('Epochs')
    axes.set_ylabel('Testing Accuracy')
#     axes.set_title('Testing Accuracy vs Epochs')
    axes.legend()
    axes.autoscale(enable=True, axis='both', tight=True)
    plt.tight_layout()
    plt.savefig(f'{Image_Dir}/testing_acc_plot.png')
    plt.close()


    # 训练时间
    fig, axes = plt.subplots(figsize=(8, 6))
    for i, (loss_name, loss_values) in enumerate(results.items()):
        training_acc = loss_values['training_acc']
        training_times = loss_values['training_times']
        axes.plot(training_times, training_acc, label=loss_name, linestyle=lineStyles[i])

    axes.set_xticks(np.arange(0, 100001, 500))  # 设置x轴刻度
    axes.set_xlabel('Time (seconds)')
    axes.set_ylabel('Training Accuracy')
#     axes.set_title('Training Accuracy vs Time')
    axes.legend()
    axes.autoscale(enable=True, axis='both', tight=True)
    plt.tight_layout()
    plt.savefig(f'{Image_Dir}/training_acc_time_plot.png')
    plt.close()

    # 测试时间
    fig, axes = plt.subplots(figsize=(8, 6))
    for i, (loss_name, loss_values) in enumerate(results.items()):
        testing_acc = loss_values['testing_acc']
        testing_times = loss_values['testing_times']
        axes.plot(testing_times, testing_acc, label=loss_name, linestyle=lineStyles[i])

    axes.set_xticks(np.arange(0, 100001, 50))  # 设置x轴刻度
    axes.set_xlabel('Time (seconds)')
    axes.set_ylabel('Testing Accuracy')
#     axes.set_title('Testing Accuracy vs Time')
    axes.legend()
    axes.autoscale(enable=True, axis='both', tight=True)
    plt.tight_layout()
    plt.savefig(f'{Image_Dir}/testing_acc_time_plot.png')
    plt.close()

    # 训练precision
    fig, axes = plt.subplots(figsize=(8, 6))
    for i, (loss_name, loss_values) in enumerate(results.items()):
        training_weighted_avg_precision = loss_values['training_weighted_avg_precision']
        x = range(0, NUM_EPOCHS + 1)
        axes.plot(x, training_weighted_avg_precision, label=loss_name, linestyle=lineStyles[i])
    axes.set_xlabel('Epochs')
    axes.set_ylabel('Training Weighted Avg Precision')
#     axes.set_title('Training Weighted Avg Precision vs Epochs')
    axes.legend()
    axes.autoscale(enable=True, axis='both', tight=True)

    plt.tight_layout()
    plt.savefig(f'{Image_Dir}/training_precision.png')
    plt.close()

    
    # 测试precision
    fig, axes = plt.subplots(figsize=(8, 6))
    for i, (loss_name, loss_values) in enumerate(results.items()):
        testing_weighted_avg_precision = loss_values['testing_weighted_avg_precision']
        x = range(0, NUM_EPOCHS + 1)
        axes.plot(x, testing_weighted_avg_precision, label=loss_name, linestyle=lineStyles[i])
    axes.set_xlabel('Epochs')
    axes.set_ylabel('Testing Weighted Avg Precision')
#     axes.set_title('Testing Weighted Avg Precision vs Epochs')
    axes.legend()
    axes.autoscale(enable=True, axis='both', tight=True)

    plt.tight_layout()
    plt.savefig(f'{Image_Dir}/testing_precision.png')
    plt.close()


    # 训练recall
    fig, axes = plt.subplots(figsize=(8, 6))
    for i, (loss_name, loss_values) in enumerate(results.items()):
        training_weighted_avg_recall = loss_values['training_weighted_avg_recall']
        x = range(0, NUM_EPOCHS + 1)
        axes.plot(x, training_weighted_avg_recall, label=loss_name, linestyle=lineStyles[i])
    axes.set_xlabel('Epochs')
    axes.set_ylabel('Training Weighted Avg recall')
#     axes.set_title('Training Weighted Avg recall vs Epochs')
    axes.legend()
    axes.autoscale(enable=True, axis='both', tight=True)

    plt.tight_layout()
    plt.savefig(f'{Image_Dir}/training_recall.png')
    plt.close()

    # 测试recall
    fig, axes = plt.subplots(figsize=(8, 6))
    for i, (loss_name, loss_values) in enumerate(results.items()):
        testing_weighted_avg_recall = loss_values['testing_weighted_avg_recall']
        x = range(0, NUM_EPOCHS + 1)
        axes.plot(x, testing_weighted_avg_recall, label=loss_name, linestyle=lineStyles[i])
    axes.set_xlabel('Epochs')
    axes.set_ylabel('Testing Weighted Avg recall')
#     axes.set_title('Testing Weighted Avg recall vs Epochs')
    axes.legend()
    axes.autoscale(enable=True, axis='both', tight=True)

    plt.tight_layout()
    plt.savefig(f'{Image_Dir}/testing_recall.png')
    plt.close()

    

    # 训练f1_score
    fig, axes = plt.subplots(figsize=(8, 6))
    for i, (loss_name, loss_values) in enumerate(results.items()):
        training_weighted_avg_f1_score = loss_values['training_weighted_avg_f1_score']
        x = range(0, NUM_EPOCHS + 1)
        axes.plot(x, training_weighted_avg_f1_score, label=loss_name, linestyle=lineStyles[i])
    axes.set_xlabel('Epochs')
#     axes.set_title('Training Weighted Avg F1 Score vs Epochs')
    axes.legend()
    axes.autoscale(enable=True, axis='both', tight=True)

    plt.tight_layout()
    plt.savefig(f'{Image_Dir}/training_f1_score.png')
    plt.close()

    # 测试f1_score
    fig, axes = plt.subplots(figsize=(8, 6))
    for i, (loss_name, loss_values) in enumerate(results.items()):
        testing_weighted_avg_f1_score = loss_values['testing_weighted_avg_f1_score']
        x = range(0, NUM_EPOCHS + 1)
        axes.plot(x, testing_weighted_avg_f1_score, label=loss_name, linestyle=lineStyles[i])
    axes.set_xlabel('Epochs')
    axes.set_ylabel('Testing Weighted Avg F1 Score')
#     axes.set_title('Testing Weighted Avg F1 Score vs Epochs')
    axes.legend()
    axes.autoscale(enable=True, axis='both', tight=True)

    plt.tight_layout()
    plt.savefig(f'{Image_Dir}/testing_f1_score.png')
    plt.close()


    # 混淆矩阵training
    for i, (loss_name, loss_values) in enumerate(results.items()):
#         epochs_to_plot = [0, 9, 49, 99]
#         epochs_to_plot = [0,1,2,3]
#         epochs_to_plot = [0, 9, 24, 49]
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        for i, epoch in enumerate(epochs_to_plot):
            confusion_matrix = np.array(loss_values['training_confusion'][epoch])
            row, col = divmod(i, 2)
            sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=axes[row, col])
            axes[row, col].set_title(f'Epoch {epoch+1}')
            axes[row, col].set_xlabel('Predicted')
            axes[row, col].set_ylabel('True')

#         plt.suptitle(f'Confusion Matrix ({loss_name})', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{Image_Dir}/training_confusion_{loss_name}.png')
        plt.close()

        # 混淆矩阵testing
    for i, (loss_name, loss_values) in enumerate(results.items()):
#         epochs_to_plot = [0, 9, 49, 99]
#         epochs_to_plot = [0,1,2,3]
#         epochs_to_plot = [0, 9, 24, 49]
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        for i, epoch in enumerate(epochs_to_plot):
            confusion_matrix = np.array(loss_values['testing_confusion'][epoch])
            row, col = divmod(i, 2)
            sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=axes[row, col])
            axes[row, col].set_title(f'Epoch {epoch+1}')
            axes[row, col].set_xlabel('Predicted')
            axes[row, col].set_ylabel('True')

#         plt.suptitle(f'Confusion Matrix ({loss_name})', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{Image_Dir}/testing_confusion_{loss_name}.png')
        plt.close()



# loss_names = results_size_train.keys()
# sample_sizes.insert(0, 0)
# fig, axes = plt.subplots(1, 2, figsize=(15, 6))
# for loss_name in loss_names:
#     training_acc = results_size_train[loss_name]
#     testing_acc = results_size_test[loss_name]

#     axes[0].plot(sample_sizes, training_acc, marker='o', label=loss_name)

#     axes[1].plot(sample_sizes, testing_acc, marker='o', label=loss_name)

# axes[0].set_title('Training Accuracy vs Sample Size')
# axes[0].set_xlabel('Sample Size')
# axes[0].set_ylabel('Training Accuracy')
# axes[0].set_xticks(sample_sizes)
# axes[0].legend()
# axes[0].autoscale(enable=True, axis='both', tight=True)

# axes[1].set_title('Testing Accuracy vs Sample Size')
# axes[1].set_xlabel('Sample Size')
# axes[1].set_ylabel('Testing Accuracy')
# axes[1].set_xticks(sample_sizes)
# axes[1].legend()
# axes[1].autoscale(enable=True, axis='both', tight=True)

# plt.tight_layout()
# plt.show()


# Plot(100, 'record/results8.json', 'images/images8',[0, 9, 49, 99])