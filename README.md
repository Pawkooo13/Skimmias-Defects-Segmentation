# Skimmias-Defects-Segmentation

The project involves developing a model for detecting various defects in Skimmia Rubella flowers. It includes an analysis of the characteristic features of individual flower defects to determine appropriate preprocessing techniques. This analysis will directly influence the selection of the most suitable model and its effectiveness. The project implements segmentation models such as FCN (Fully Convolutional Neural Network) and U-Net. Detecting individual flower damages enables appropriate responses and helps prevent the issue from spreading on a larger scale. Such preventive action can protect the company from significant financial losses.

| Bitten | Burnt |
|-------------|-------------|
|  ![Image_39](https://github.com/user-attachments/assets/e37b159c-cbf8-4f09-9050-8506f4aa575d) | ![Image_151](https://github.com/user-attachments/assets/9833d3c9-0178-4972-932f-817e70376b8d) |

The dataset consists of over 300 images depicting plants with various defects such as bite marks and burns, along with their corresponding masks. It also includes several dozen hard-mining examples (images of plants without defects) to evaluate the impact of such data on the training of segmentation models. The models were tested on two datasets - the raw dataset and an extended version containing additional images of plants without defects - to evaluate whether the models would learn to detect defects more effectively by ignoring irrelevant elements such as leaves or pots.

| Image | Mask |
|---------|------|
| ![Image_14](https://github.com/user-attachments/assets/507d878c-113e-4185-a14c-affa9943c8a2) | ![Image_14-mask](https://github.com/user-attachments/assets/e902e1ba-2704-4af6-aac6-c1db9c03ad9b) |

```
Clone repository using  `git clone`
```
```
Create virtual environment using `python -m venv venv`
```
```
Activate virtual envitonemnt using `.\venv\Scripts\activate`
```
```
Insall requirements using `pip install -r requirements.txt`
```
```
Download data and place them into "data" directory
```
```
Write  `dvc pull` in terminal
```
```
Write `dvc repro` in terminal to start the pipeline 
```
