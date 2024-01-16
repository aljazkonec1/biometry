import torch
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image
import os
import numpy as np

wd = os.path.join(os.getcwd(), 'datasets', 'ears', 'images-cropped')
img_dir = os.path.join(wd, 'test')

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model = resnet50(weights=None)
pretrained = torch.load('best_model.pt')
del pretrained['fc.weight']
del pretrained['fc.bias'] 

model.load_state_dict(pretrained, strict=False)

# model.eval()
# probi se visje rezati, mogoce -3 
feature_extractor = torch.nn.Sequential(*list(model.children())[:-3])
feature_extractor.eval()

image_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]

for image_path in image_files:
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    image = Image.open(os.path.join(img_dir, image_path)).convert('RGB')
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)


    with torch.no_grad():
        features = feature_extractor(input_batch)
    features = features[0, :, 0, 0]
    
    # print(max(features),min(features))
    print(features)
    # print(features.shape)
    # print(type(features))
    features = features.numpy()
    np.savetxt(os.path.join(wd, "test_resnet_3", image_name+ '.csv'), features, delimiter=",")


    # torch.save(features, os.path.join(wd, "test_resnet", image_name + ".pt"))