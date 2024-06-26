import cv2
import torch
import fractions
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions

def lcm(a, b):
    return abs(a * b) / fractions.gcd(a, b) if a and b else 0

def face_transformation(pic_a_path, pic_b_path, number='0',output_path="./output/"):
    transformer_Arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transformer = transforms.Compose([
        transforms.ToTensor(),
    ])

    detransformer = transforms.Compose([
        transforms.Normalize([0, 0, 0], [1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize([-0.485, -0.456, -0.406], [1, 1, 1])
    ])

    opt = TestOptions().parse()

    torch.nn.Module.dump_patches = True
    model = create_model(opt)
    model.eval()

    with torch.no_grad():
        img_a = Image.open(pic_a_path).convert('RGB')
        img_a = transformer_Arcface(img_a)
        img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])

        img_b = Image.open(pic_b_path).convert('RGB')
        img_b = transformer(img_b)
        img_att = img_b.view(-1, img_b.shape[0], img_b.shape[1], img_b.shape[2])

        img_id = img_id.cuda()
        img_att = img_att.cuda()

        img_id_downsample = F.interpolate(img_id, size=(112,112))
        latend_id = model.netArc(img_id_downsample)
        latend_id = latend_id.detach().to('cpu')
        latend_id = latend_id/np.linalg.norm(latend_id, axis=1, keepdims=True)
        latend_id = latend_id.to('cuda')

        img_fake = model(img_id, img_att, latend_id, latend_id, True)

        row1, row2, row3 = None, None, None

        for i in range(img_id.shape[0]):
            if i == 0:
                row1 = img_id[i]
                row2 = img_att[i]
                row3 = img_fake[i]
            else:
                row1 = torch.cat([row1, img_id[i]], dim=2)
                row2 = torch.cat([row2, img_att[i]], dim=2)
                row3 = torch.cat([row3, img_fake[i]], dim=2)

        full = row3.detach()
        full = full.permute(1, 2, 0)
        output = full.to('cpu')
        output = np.array(output)
        output = output[..., ::-1]
        output = output * 255

        cv2.imwrite(output_path + f'result{number}.jpg', output)
        return str(output_path + f'result{number}.jpg')

# # Example usage:
# face_transformation("path/to/pic_a.jpg", "path/to/pic_b.jpg", "path/to/output/")
