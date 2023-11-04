# **Task2** 
## **发现一：**
我们发现当learning rate为1e-2时最终的准确度始终最高；leaning rate为1e-1时次之，有时候接近learning rate为1e-2时最终的准确度，有时距离learning rate为1e-2时最终的准确度较远；learning rate为1e-3时最终的准确度始终最小。
## **解释发现一：**
leaning rate为1e-1时较大，导致在优化的过程中，在最优值的附近反复多次横跳；learning rate为1e-3时太小了，需要较大的步数才能到达最优，在这个问题中，我们的步数较少，导致达不到一个比较理想的结果。而learning rate为1e-2时，步长的设置较为合理。
## **发现二：**
通过比较learning rate为1e-1和leaning rate为1e-2时的图像，发现虽然它们有时候最终的准确度接近，但learning rate为1e-2时，准确度较快达到相对稳定的状态，leaning rate为1e-1时较慢达到相对稳定的状态。
## **解释发现二：**
leaning rate为1e-1时较大，导致在优化的过程中，在最优值的附近反复多次横跳才能达到相对较优的结果，而learning rate为1e-2时，步长的设置较为合理。
## **发现三：**
无论learning rate的选取，模型在训练集上随着训练次数的增加，模型在训练集上的准确度不断升高，loss不断降低
## **解释发现三**
说明模型是收敛的
## **额外在1e-2周围测试几个learning_rate(5e-3, 2e-2, 4e-2)**
我们发现当Learning rate为2e-2时模型的表现最好，4e-2时次之。
