# SRGAN
This is a SRGAN Code reproduction based on pytorch using python.
This code is based on paper [*Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network*](https://arxiv.org/pdf/1609.04802v1.pdf), with the help of [this blog](https://blog.csdn.net/NikkiElwin/article/details/112910957?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522163238627116780265454371%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=163238627116780265454371&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2%7Eall%7Etop_positive%7Edefault-1-112910957.first_rank_v2_pc_rank_v29&utm_term=srgan+pytorch&spm=1018.2226.3001.4187).

Realize the super-resolution reconstruction of a 24×24 image to 96×96.

# Software Environment
- Python 3.8.8 (default, Apr 13 2021, 15:08:03) [MSC v.1916 64 bit (AMD64)] :: Anaconda, Inc. on win32
- pytorch                   1.7.1           py3.8_cuda110_cudnn8_0  
- opencv                    4.0.1            py38h2a7c758_0
- numpy                     1.20.1           py38h34a8a5c_0
- torchvision               0.8.2                py38_cu110
- matplotlib                3.3.4            py38haa95532_0
- tensorboard               2.6.0                    pypi_0 

# Test Dataset
[Anime avatar dataset](https://drive.google.com/uc?id=1IGrTr308mGAaCKotpkkm8wTKlWs9Jq-p) 

# Structure 
+ model.py: Network model of Generator and Discriminator
+ srgan.py: Training SRGAN network
+ srgan_eval.py: Evaulate the network performance
+ logs: Tensorboard logs visualizing the training process (check by instruction 'tensorboard --logdir=logs') 
+ data: Images dataset

# Results Demonstration
## The visualization of training process are shown below: 
![result1](https://github.com/huzihan0525/SRGAN/blob/main/images/loss_result.png)
![result1](https://github.com/huzihan0525/SRGAN/blob/main/images/image_result.png)

## Some of the training results are shown below:
![result1](https://github.com/huzihan0525/SRGAN/blob/main/images/training_result.png)

## Some of the test results are shown below: (arbitrary anime avatar images from Internet)
![result1](https://github.com/huzihan0525/SRGAN/blob/main/images/result1.png)
![result2](https://github.com/huzihan0525/SRGAN/blob/main/images/result2.png)
![result3](https://github.com/huzihan0525/SRGAN/blob/main/images/result3.png)
![result4](https://github.com/huzihan0525/SRGAN/blob/main/images/result4.png)



