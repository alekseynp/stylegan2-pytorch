from PIL import Image
import torch
import torch.autograd as autograd
from torchvision import transforms
import numpy as np

from processing.blur import GaussianBlur, RandomGaussianBlur


img = Image.open('blur.png').convert('RGB')


def show_blur(size, sig):
    gb = GaussianBlur(size, sig, n_channels=3).cuda()

    img_tensor_batch = torch.stack([transforms.ToTensor()(img)])

    output = gb(img_tensor_batch.cuda())

    output = output.data.cpu().numpy()[0]
    output = np.round(255.0 * np.transpose(output, axes=(1, 2, 0))).astype(np.uint8)

    Image.fromarray(output).save('{}_{}.png'.format(size, sig))

show_blur(3, 3)
show_blur(5, 5)

for size in [3,6]:
    for sig in [1.0, 2.0, 3.0]:
        show_blur(size, sig)




rgb = RandomGaussianBlur(0.5, 3, 1.0, n_channels=3).cuda()

img_tensor_batch = torch.stack([transforms.ToTensor()(img)])
img_tensor_batch = img_tensor_batch.repeat(16, 1, 1, 1)

output = rgb(img_tensor_batch.cuda()).cpu().numpy()

for i in range(16):
    o = np.round(255.0 * np.transpose(output[i], axes=(1, 2, 0))).astype(np.uint8)
    Image.fromarray(o).save('{}.png'.format(i))