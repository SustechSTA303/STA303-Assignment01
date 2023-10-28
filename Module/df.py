import matplotlib.pyplot as plt
import json
import pandas as pd



Results_Dir='record/results4.json'

with open(Results_Dir, 'r') as file:
    results = json.load(file)

    
for loss_name, loss_values in results.items():
    # 提取每个列表的最后一个元素
    final_training_loss = loss_values['training_loss'][-1]
    final_testing_loss = loss_values['testing_loss'][-1]
    final_training_acc = loss_values['training_acc'][-1]
    final_testing_acc = loss_values['testing_acc'][-1]
    final_training_weighted_avg_precision = loss_values['training_weighted_avg_precision'][-1]
    final_testing_weighted_avg_precision = loss_values['testing_weighted_avg_precision'][-1]
    final_training_weighted_avg_recall = loss_values['training_weighted_avg_recall'][-1]
    final_testing_weighted_avg_recall = loss_values['testing_weighted_avg_recall'][-1]
    final_training_weighted_avg_f1_score = loss_values['training_weighted_avg_f1_score'][-1]
    final_testing_weighted_avg_f1_score = loss_values['testing_weighted_avg_f1_score'][-1]

    # 将提取的最后一个元素添加到原始JSON数据中
    loss_values['final_training_loss'] = final_training_loss
    loss_values['final_testing_loss'] = final_testing_loss
    loss_values['final_training_acc'] = final_training_acc
    loss_values['final_testing_acc'] = final_testing_acc
    loss_values['final_training_weighted_avg_precision'] = final_training_weighted_avg_precision
    loss_values['final_testing_weighted_avg_precision'] = final_testing_weighted_avg_precision
    loss_values['final_training_weighted_avg_recall'] = final_training_weighted_avg_recall
    loss_values['final_testing_weighted_avg_recall'] = final_testing_weighted_avg_recall
    loss_values['final_training_weighted_avg_f1_score'] = final_training_weighted_avg_f1_score
    loss_values['final_testing_weighted_avg_f1_score'] = final_testing_weighted_avg_f1_score

# 将更新后的数据写回原始JSON文件
with open(Results_Dir, 'w') as file:
    json.dump(results, file, indent=4)    

    
with open(Results_Dir, 'r') as file:
    results = json.load(file)    
df = pd.DataFrame(results)
df.to_excel('results5.xlsx', index=True)








