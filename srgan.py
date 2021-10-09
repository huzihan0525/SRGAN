import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import cv2
from model import *
from torch.utils.tensorboard import SummaryWriter


EPOCHS = 10


class MyDataSet(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.img_list = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_path = os.path.join(self.root_dir, img_name)
        to_tensor = torchvision.transforms.ToTensor()
        resize = torchvision.transforms.Resize((24, 24))
        hr_img = cv2.imread(img_path)
        hr_img = to_tensor(hr_img)
        lr_img = resize(hr_img)
        return lr_img, hr_img


data_root = './data'
data_set = MyDataSet(data_root)
data_loader = DataLoader(data_set, batch_size=32)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 构造迭代器并取出其中一个样本
data_iter = iter(data_loader)
test_imgs, _ = data_iter.next()
test_imgs = test_imgs.to(device)  # test_imgs的用处是为了可视化生成对抗的结果

# 创建网络
netG = NetworkG()
netD = NetworkD()
netG.to(device)
netD.to(device)

# 定义优化器
optimizerG = torch.optim.Adam(netG.parameters())
optimizerD = torch.optim.Adam(netD.parameters())

# 构造损失函数
lossF = nn.MSELoss().to(device)

# 构造VGG损失中的网络模型
vgg = torchvision.models.vgg16(pretrained=True).to(device)
lossNetwork = nn.Sequential(*list(vgg.features)[:31]).eval()
for param in lossNetwork.parameters():
    param.requires_grad = False  # 让VGG停止学习

total_step = 0
writer = SummaryWriter('logs')
writer.add_images('original_img', test_imgs)
print("Starting Training loop...")
for epoch in range(EPOCHS):
    netD.train()
    netG.train()
    for i, imgs in enumerate(data_loader, 0):
        img1, img2 = imgs
        lr_img = img1.to(device)  # 低分辨率图片
        original_img = img2.to(device)  # 原图片

        fake_img = netG(lr_img).to(device)

        # training D
        netD.zero_grad()
        realOut = netD(original_img).mean()
        fakeOut = netD(fake_img).mean()
        dLoss = 1 - realOut + fakeOut
        dLoss.backward(retain_graph=True)

        # training G
        netG.zero_grad()
        gLossSR = lossF(fake_img, original_img)
        gLossGAN = 0.001 * torch.mean(1 - fakeOut)
        gLossVGG = 0.006 * lossF(lossNetwork(fake_img), lossNetwork(original_img))
        gLoss = gLossSR + gLossGAN + gLossVGG
        gLoss.backward()

        optimizerD.step()
        optimizerG.step()

        if total_step % 50 == 0:
            print('[%d/%d][%d/%d]\t Loss_D: %.4f\t Loss_G: %.4f'
                  % (epoch+1, EPOCHS, i, len(data_loader), dLoss.item(), gLoss.item()))
            writer.add_scalars("loss", {'D_loss': dLoss.item(),
                                        'G_loss': gLoss.item()}, total_step)
            with torch.no_grad():
                fake_imgs = netG(test_imgs).detach().cpu()
                writer.add_images('generate_img', fake_imgs, total_step)

        if total_step % 200 == 0:
            torch.save(netG.state_dict(), 'model/netG_epoch_%d.pth' % total_step)
            torch.save(netD.state_dict(), 'model/netD_epoch_%d.pth' % total_step)

        total_step += 1

writer.close()

