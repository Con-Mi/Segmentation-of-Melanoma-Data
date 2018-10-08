from data_loader import Melanoma_Train_Validation_DataLoader

train_loader, validation_loader = Melanoma_Train_Validation_DataLoader(batch_size = 1)

for i, sample in enumerate(validation_loader):
    img, label_img = sample
    print(img.size(2))
    print(label_img.size(2))
