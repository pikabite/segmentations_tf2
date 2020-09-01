# Tensorflow 2.1 Segmentation pipeline implementation from scratch

---

## Process with Cityscape datasets
### Cityscape data with HRNet

cityscape dataset (https://www.cityscapes-dataset.com/)  
HRNet (https://arxiv.org/pdf/1908.07919.pdf)  

HRNetV2W48 model  
mIOU 70.31% (without pre-trained backbone)  
  
| Input image | Ground truth | Segmentation result |  
![cityscape hrnet result 1](./outputs/cityscape_hrnet/frankfurt_000001_079206_leftImg8bit.png)  
![cityscape hrnet result 2](./outputs/cityscape_hrnet/frankfurt_000001_044658_leftImg8bit.png)  
![cityscape hrnet result 3](./outputs/cityscape_hrnet/munster_000001_000019_leftImg8bit.png)  
![cityscape hrnet result 3](./outputs/cityscape_hrnet/munster_000051_000019_leftImg8bit.png)  



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
