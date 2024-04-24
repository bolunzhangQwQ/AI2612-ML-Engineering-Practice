import torch
import torch.nn.functional as F
import cv2
import numpy as np
from swap_two_image import face_transformation
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def calculate_cosine_distance(source_img, result_img, ArcFace_model, device):
    cos_loss = torch.nn.CosineSimilarity()

    imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)

    source_img = (source_img - imagenet_mean) / imagenet_std
    source_img = source_img.unsqueeze(0)
    source_img = F.interpolate(source_img, size=(112, 112), mode='bicubic')
    source_id = ArcFace_model(source_img)
    source_id = F.normalize(source_id, p=2, dim=1)

    result_img = (result_img - imagenet_mean) / imagenet_std
    result_img = result_img.unsqueeze(0)
    result_img = F.interpolate(result_img, size=(112, 112), mode='bicubic')
    result_id = ArcFace_model(result_img)
    result_id = F.normalize(result_id, p=2, dim=1)

    cos_dis = 1 - cos_loss(source_id, result_id)

    return cos_dis.item()


def evaluate_face_swap(model, source_images, target_images):
    total_cosine_distance = 0.0

    for i in range(len(source_images)):
        source_img_path= source_images[i]
        target_img_path= target_images[i]

        source_img= preprocess_image(source_img_path)
        target_img= preprocess_image(target_img_path)

        cos_distance_source_to_target = calculate_cosine_distance(source_img, target_img, model, device)
        cos_distance_target_to_source = calculate_cosine_distance(target_img, source_img, model, device)

        total_cosine_distance += (cos_distance_source_to_target + cos_distance_target_to_source) / 2


    average_cosine_distance = total_cosine_distance / len(source_images)

    return average_cosine_distance

def preprocess_image(img_path):
    image = cv2.imread(img_path)
    image = torch.tensor(image)
    image = image.permute(2, 0, 1)
    #print(np.shape(image))
    return image#torch.zeros((3, 112, 112)) 


def main():
    source_folder = "./FaceImageTest/"
    num_images = 5  # Replace with the actual number of images in your dataset

    source_images = [os.path.join(source_folder, f"source_{i:02d}.jpg") for i in range(num_images)]
    target_images = [os.path.join(source_folder, f"target_{i:02d}.jpg") for i in range(num_images)]

    path_to_arcface_ckpt = "./arcface_model/arcface_checkpoint.tar"
    ArcFace_model = torch.load(path_to_arcface_ckpt, map_location=torch.device("cpu"))
    ArcFace_model.eval()
    ArcFace_model.requires_grad_(False)


    # # Evaluate face swap
    # average_cosine_distance = evaluate_face_swap(ArcFace_model, source_images, target_images)

    # print(f"Average Cosine Distance: {average_cosine_distance}")

    # Perform source-target swapping
    for num in range(num_images):
        source_img_path = source_images[num]
        target_img_path = target_images[num]
        # Perform face swap using face_transformation function
        swapped_img_0= face_transformation(source_img_path, target_img_path,number=str(num)+'0') #Face from source
        swapped_img_0=preprocess_image(swapped_img_0)
        swapped_img_1= face_transformation(target_img_path, source_img_path,number=str(num)+'1') #Face from target
        swapped_img_1=preprocess_image(swapped_img_1)
        # # Load images
        source_img = preprocess_image(source_img_path)
        target_img = preprocess_image(target_img_path)
        # Calculate cosine distance with the swapped image
        cos_distance_swapped_0= calculate_cosine_distance(swapped_img_0, source_img, ArcFace_model, device)
        cos_distance_swapped_1= calculate_cosine_distance(swapped_img_1, target_img, ArcFace_model, device)
        print(f"Cosine Distance after swapping target_{num:02d} with source_{num:02d}: {cos_distance_swapped_0}")
        print(f"Cosine Distance after swapping source_{num:02d} with target_{num:02d} : {cos_distance_swapped_1}")
if __name__ == '__main__':
    main()
