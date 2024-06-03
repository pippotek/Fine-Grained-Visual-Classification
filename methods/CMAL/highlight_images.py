import torch
from torchvision import transforms
from methods.CMAL.basic_conv import *


# Highlight images
def highlight_im(images, attention_map, attention_map2, attention_map3, theta=0.5, padding_ratio=0.1):
    images = images.clone()
    attention_map = attention_map.clone().detach()
    attention_map2 = attention_map2.clone().detach()
    attention_map3 = attention_map3.clone().detach()

    batches, _, imgH, imgW = images.size()

    for batch_index in range(batches):
        image_tmp = images[batch_index]
        map_tpm = attention_map[batch_index].unsqueeze(0).unsqueeze(0)
        map_tpm = torch.nn.functional.interpolate(map_tpm, size=(imgH, imgW)).squeeze()
        map_tpm = (map_tpm - map_tpm.min()) / (map_tpm.max() - map_tpm.min() + 1e-6)


        map_tpm2 = attention_map2[batch_index].unsqueeze(0).unsqueeze(0)
        map_tpm2 = torch.nn.functional.interpolate(map_tpm2, size=(imgH, imgW)).squeeze()
        map_tpm2 = (map_tpm2 - map_tpm2.min()) / (map_tpm2.max() - map_tpm2.min() + 1e-6)

        map_tpm3 = attention_map3[batch_index].unsqueeze(0).unsqueeze(0)
        map_tpm3 = torch.nn.functional.interpolate(map_tpm3, size=(imgH, imgW)).squeeze()
        map_tpm3 = (map_tpm3 - map_tpm3.min()) / (map_tpm3.max() - map_tpm3.min() + 1e-6)

        map_tpm = (map_tpm + map_tpm2 + map_tpm3)
        map_tpm = (map_tpm - map_tpm.min()) / (map_tpm.max() - map_tpm.min() + 1e-6)
        map_tpm = map_tpm >= theta

        nonzero_indices = torch.nonzero(map_tpm, as_tuple=False)
        height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
        height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
        width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
        width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)

        image_tmp = image_tmp[:, height_min:height_max, width_min:width_max].unsqueeze(0)
        image_tmp = torch.nn.functional.interpolate(image_tmp, size=(imgH, imgW)).squeeze()

        images[batch_index] = image_tmp

    return images

def show_image(inputs):
    inputs = inputs.squeeze()
    to_pil_image = transforms.ToPILImage()
    img = to_pil_image(inputs.cpu())
    img.show()

def attention_im(images, attention_map, theta=0.5, padding_ratio=0.1):
    images = images.clone()
    attention_map = attention_map.clone().detach()
    batches, _, imgH, imgW = images.size()

    for batch_index in range(batches):
        image_tmp = images[batch_index]
        map_tpm = attention_map[batch_index].unsqueeze(0).unsqueeze(0)
        map_tpm = torch.nn.functional.interpolate(map_tpm, size=(imgH, imgW)).squeeze()
        map_tpm = (map_tpm - map_tpm.min()) / (map_tpm.max() - map_tpm.min() + 1e-6)
        map_tpm = map_tpm >= theta
        nonzero_indices = torch.nonzero(map_tpm, as_tuple=False)
        height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
        height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
        width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
        width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)

        image_tmp = image_tmp[:, height_min:height_max, width_min:width_max].unsqueeze(0)
        image_tmp = torch.nn.functional.interpolate(image_tmp, size=(imgH, imgW)).squeeze()

        images[batch_index] = image_tmp

    return images