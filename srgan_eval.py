from model import *
import torch
import cv2
import matplotlib.pyplot as plt
import torchvision
import numpy as np


img_path = 'images\\test4.jpg'
NetG = NetworkG()
NetG.load_state_dict(torch.load('model\\netG_epoch_22000.pth'))
original_img = cv2.imread(img_path)
original_img = original_img[:, :, [2, 1, 0]]
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Resize((24, 24))])
img = transform(original_img)
img = torch.reshape(img, (1, 3, 24, 24))
output = NetG(img)
img = np.transpose(torch.reshape(img, (3, 24, 24)).detach().numpy(), (1, 2, 0))
bicubic_img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_CUBIC)
output = np.transpose(torch.reshape(output, (3, 96, 96)).detach().numpy(), (1, 2, 0))

plt.figure(facecolor='none')
plt.subplot(1, 4, 1)
plt.imshow(img)
plt.title("Low-Resolution Image Input")
plt.subplot(1, 4, 2)
plt.imshow(bicubic_img)
plt.title("After Bicubic")
plt.subplot(1, 4, 3)
plt.imshow(output)
plt.title("After SRGAN")
plt.subplot(1, 4, 4)
plt.imshow(original_img)
plt.title("Original Image")
plt.show()

