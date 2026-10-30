from torchvision import datasets

dataset = datasets.ImageFolder("./EuroSAT")
print(dataset.classes)
print(len(dataset.classes))
