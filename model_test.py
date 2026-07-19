import torch
from torchvision.datasets import ImageFolder
import torchvision
import torch.utils.data as Data
from torchvision import transforms
from model import GoogLeNet, Inception
from PIL import Image

def test_data_process():
    test_data = ImageFolder(
        root='./data/test',  # 直接指向 train 文件夹（里面要有 dog 和 cat 子文件夹）
        transform=transforms.Compose([
            transforms.Resize((224, 224)),  # 统一缩放到 224x224
            transforms.RandomHorizontalFlip(),  # 新增：随机左右翻转（增强数据，防止过拟合）
            transforms.ToTensor(),  # 转成张量
            transforms.Normalize(  # 新增：标准化（让模型训练更稳定）
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    )

    test_dataloader = Data.DataLoader(dataset=test_data,
                                       batch_size=1,
                                       shuffle=True,
                                       num_workers=2)

    return test_dataloader

def test_model_process(model, test_dataloader):

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)

    test_acc = 0.0
    test_num = 0
    test_corrects = 0

    with torch.no_grad():
        for test_data_x, test_data_y in test_dataloader:
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)
            model.eval()

            output = model(test_data_x)
            pre_label = torch.argmax(output, dim=1)

            test_corrects += torch.sum(pre_label == test_data_y)
            test_num += test_data_x.size(0)

    test_acc = test_corrects / test_num
    print("测试准确率：", test_acc.item())

if __name__ == "__main__":
    model = GoogLeNet(Inception)
    model.load_state_dict(torch.load("model.pth"))

    test_dataloader = test_data_process()
    #test_model_process(model, test_dataloader)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    classes = ['猫', '狗']
    with torch.no_grad():
        # for b_x, b_y in test_dataloader:
        #     b_x = b_x.to(device)
        #     b_y = b_y.to(device)
        #     model.eval()
        #
        #     output = model(b_x)
        #     pre_label = torch.argmax(output, dim=1)
        #     result = pre_label.iteme()
        #     label = b_y.item()
        #
        #     print("预测值：", classes[result],"--------", "真实值：", classes[label])
        image = Image.open("./1.png")
        transforms = transforms.Compose([transforms.Lambda(lambda img: img.convert('RGB')),
                                        transforms.Resize((224,224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])
        image = transforms(image)
        image = image.unsqueeze(0)
        image = image.to(device)
        model.eval()
        output = model(image)
        ptr_label = torch.argmax(output, dim=1)
        result = ptr_label.item()
        print(classes[result])