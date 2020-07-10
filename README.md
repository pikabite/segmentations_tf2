# Tensorflow 2.1 Segmentation pipeline implementation from scratch


---   

## Process with Inria datasets
### Inria dataset with *Modified HRNet  
inria dataset (https://project.inria.fr/aerialimagelabeling/leaderboard/)  
HRNet (https://arxiv.org/pdf/1908.07919.pdf)  
*Modify HRNet's layers for satellite image processing
  
| Input image | Ground truth | Segmentation result |  
![inria hrnet result 1](./outputs/inria_subject4/chicago4_97.png)  
![inria hrnet result 2](./outputs/inria_subject4/chicago4_153.png)  
![inria hrnet result 3](./outputs/inria_subject4/chicago4_114.png)  
![inria hrnet result 3](./outputs/inria_subject4/chicago4_248.png)  

Foreground IOU : 75.63% (without post-processing)
