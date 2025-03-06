import os
import json
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision import models
from PIL import Image
import numpy as np

# Set device
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

# ---------------------------
# Load ImageNet Class Index and Build Mappings
# ---------------------------
class_idx = json.load(open("./imagenet_class_index.json"))
synset2idx = {v[0]: int(k) for k, v in class_idx.items()}
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]

# ---------------------------
# Define Transform for Inception v3
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
])

# ---------------------------
# Custom Dataset for Non-Standard ImageNet Folder
# ---------------------------
class CustomImageNetDataset(Data.Dataset):
    """
    Expects all images in a single folder.
    Filenames are assumed to start with the synset (e.g., "n04579432_whistle.jpg").
    """
    def __init__(self, root, transform=None, synset2idx=None):
        self.root = root
        self.transform = transform
        self.synset2idx = synset2idx
        self.image_files = [f for f in os.listdir(root) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, index):
        filename = self.image_files[index]
        filepath = os.path.join(self.root, filename)
        image = Image.open(filepath).convert('RGB')
        if self.transform:
            image = self.transform(image)
        synset = filename.split('_')[0]
        label = self.synset2idx[synset]
        return image, label

# ---------------------------
# Utility Function: Display Image
# ---------------------------
def imshow(img, title):
    npimg = img.numpy()
    plt.figure(figsize=(5,5))
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.title(title)
    plt.axis('off')
    plt.show()

# ---------------------------
# GaLoreProjector and 8-bit Quantization Functions
# ---------------------------
import torch.nn.functional as F

class GaLoreProjector:
    def __init__(self, rank, verbose=False, update_proj_gap=10, scale=1.0, proj_type='std',
                 quant=False, group_size=-1, n_bit=8, cos_threshold=0.4, gamma_proj=2, queue_size=5):
        self.rank = rank
        self.verbose = verbose
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.ortho_matrix = None
        self.ortho_matrix_scales = None
        self.ortho_matrix_zeros = None
        self.ortho_matrix_shape = None
        self.proj_type = proj_type

        self.quant = quant
        self.quant_group_size = group_size
        self.quant_n_bit = n_bit

        self.past_ortho_vector = None
        self.queue_size = queue_size
        self.queue = []
        self.cos_threshold = cos_threshold
        self.gamma_proj = gamma_proj
        self.svd_count = 0

    def project(self, full_rank_grad, iter):
        # full_rank_grad is expected to be a 2D tensor.
        assert full_rank_grad.dim() == 2, "Expected a 2D tensor for projection."
        if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                float_ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='right')
                self.svd_count += 1
                if self.past_ortho_vector is not None:
                    if len(self.queue) == self.queue_size:
                        self.queue.pop(0)
                    self.queue.append(F.cosine_similarity(self.past_ortho_vector,
                        float_ortho_matrix[:1, :].clone().flatten(), dim=0).item())
                    if len(self.queue) == self.queue_size and sum(self.queue)/self.queue_size >= self.cos_threshold:
                        self.update_proj_gap = int(self.update_proj_gap * self.gamma_proj)
                self.past_ortho_vector = float_ortho_matrix[:1, :].clone().flatten()
                if self.quant:
                    self.ortho_matrix, self.ortho_matrix_scales, self.ortho_matrix_zeros, self.ortho_matrix_shape = \
                        self._quantize(float_ortho_matrix, q_group_size=self.quant_group_size, n_bit=self.quant_n_bit)
                else:
                    self.ortho_matrix = float_ortho_matrix
            if self.quant:
                float_ortho_matrix = self.ortho_matrix_scales * (
                    self.ortho_matrix.to(self.ortho_matrix_scales.dtype) - self.ortho_matrix_zeros)
                float_ortho_matrix = float_ortho_matrix.reshape(self.ortho_matrix_shape)
            else:
                float_ortho_matrix = self.ortho_matrix
            low_rank_grad = torch.matmul(full_rank_grad, float_ortho_matrix.t())
        else:
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                float_ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='left')
                self.svd_count += 1
                if self.past_ortho_vector is not None:
                    if len(self.queue) == self.queue_size:
                        self.queue.pop(0)
                    self.queue.append(F.cosine_similarity(self.past_ortho_vector,
                        float_ortho_matrix[:, :1].clone().flatten(), dim=0).item())
                    if len(self.queue) == self.queue_size and sum(self.queue)/self.queue_size >= self.cos_threshold:
                        self.update_proj_gap = int(self.update_proj_gap * self.gamma_proj)
                self.past_ortho_vector = float_ortho_matrix[:, :1].clone().flatten()
                if self.quant:
                    self.ortho_matrix, self.ortho_matrix_scales, self.ortho_matrix_zeros, self.ortho_matrix_shape = \
                        self._quantize(float_ortho_matrix, q_group_size=self.quant_group_size, n_bit=self.quant_n_bit)
                else:
                    self.ortho_matrix = float_ortho_matrix
            if self.quant:
                float_ortho_matrix = self.ortho_matrix_scales * (
                    self.ortho_matrix.to(self.ortho_matrix_scales.dtype) - self.ortho_matrix_zeros)
                float_ortho_matrix = float_ortho_matrix.reshape(self.ortho_matrix_shape)
            else:
                float_ortho_matrix = self.ortho_matrix
            low_rank_grad = torch.matmul(float_ortho_matrix.t(), full_rank_grad)
        return low_rank_grad

    def project_back(self, low_rank_grad):
        if self.proj_type == 'std':
            if self.quant:
                float_ortho_matrix = self.ortho_matrix_scales * (
                    self.ortho_matrix.to(self.ortho_matrix_scales.dtype) - self.ortho_matrix_zeros)
                float_ortho_matrix = float_ortho_matrix.reshape(self.ortho_matrix_shape)
            else:
                float_ortho_matrix = self.ortho_matrix
            if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
                full_rank_grad = torch.matmul(low_rank_grad, float_ortho_matrix)
            else:
                full_rank_grad = torch.matmul(float_ortho_matrix, low_rank_grad)
        return full_rank_grad * self.scale

    def get_orthogonal_matrix(self, weights, rank, type):
        if weights.dtype != torch.float:
            float_data = False
            original_type = weights.dtype
            original_device = weights.device
            matrix = weights.float()
        else:
            float_data = True
            matrix = weights
        U, s, Vh = torch.linalg.svd(matrix, full_matrices=False)
        if type == 'right':
            B = Vh[:rank, :]
            if not float_data:
                B = B.to(original_device).type(original_type)
            return B
        elif type == 'left':
            A = U[:, :rank]
            if not float_data:
                A = A.to(original_device).type(original_type)
            return A
        elif type == 'full':
            A = U[:, :rank]
            B = Vh[:rank, :]
            if not float_data:
                A = A.to(original_device).type(original_type)
                B = B.to(original_device).type(original_type)
            return [A, B]
        else:
            raise ValueError('type should be left, right or full')

    def _quantize(self, w, q_group_size=-1, n_bit=8):
        org_w_shape = w.shape
        if q_group_size > 0:
            if w.nelement() < q_group_size:
                q_group_size = w.nelement()
            assert w.nelement() % q_group_size == 0, f"Total elements {w.nelement()} not divisible by {q_group_size}"
            w = w.reshape(-1, q_group_size)
        assert w.dim() == 2
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2 ** n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
        assert torch.isnan(scales).sum() == 0
        assert torch.isnan(w).sum() == 0
        w = torch.clamp(torch.round(w / scales) + zeros, min_int, max_int).to(torch.uint8)
        return w, scales, zeros, org_w_shape

def quantize_8bit(tensor, n_bit=8):
    orig_shape = tensor.shape
    tensor_flat = tensor.view(-1, tensor.shape[-1])
    max_val = tensor_flat.amax(dim=1, keepdim=True)
    min_val = tensor_flat.amin(dim=1, keepdim=True)
    max_int = 2 ** n_bit - 1
    scales = (max_val - min_val).clamp(min=1e-5) / max_int
    zeros = (-torch.round(min_val / scales)).clamp(0, max_int)
    tensor_quant = torch.clamp(torch.round(tensor_flat / scales) + zeros, 0, max_int).to(torch.uint8)
    tensor_quant = tensor_quant.view(orig_shape)
    return tensor_quant, scales, zeros

def dequantize_8bit(tensor_quant, scales, zeros):
    orig_shape = tensor_quant.shape
    tensor_flat = tensor_quant.view(-1, tensor_quant.shape[-1]).to(torch.float32)
    tensor_dequant = (tensor_flat - zeros) * scales
    tensor_dequant = tensor_dequant.view(orig_shape)
    return tensor_dequant

# ---------------------------
# Adaptive PGD Attack (Dynamic Projection) with Targeting Option
# ---------------------------
def adaptive_pgd_attack_8bit(model, images, labels, eps=0.3, alpha=2/255, iters=40, device='cuda',
                             targeted=False, target_label=None):
    """
    Runs the adaptive (dynamic projection) PGD attack.
    If targeted=True and target_label is provided, the attack minimizes the loss for the target label,
    and the update is reversed (i.e. subtracting the gradient sign).
    """
    images = images.clone().detach().to(device)
    labels = labels.to(device)
    loss_fn = nn.CrossEntropyLoss()
    ori_images = images.clone().detach()
    projector = GaLoreProjector(rank=10, update_proj_gap=10, scale=1.0, proj_type='std',
                                quant=True, group_size=3, n_bit=8, cos_threshold=0.4, gamma_proj=2, queue_size=5)
    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)
        model.zero_grad()
        if targeted and target_label is not None:
            target_tensor = torch.full_like(labels, target_label)
            loss = loss_fn(outputs, target_tensor)
        else:
            loss = loss_fn(outputs, labels)
        loss.backward()
        grad = images.grad
        grad_flat = grad.view(grad.size(0), -1)
        grad_qg = projector.project(grad_flat, i)
        grad_qg_quant, scales, zeros = quantize_8bit(grad_qg, n_bit=8)
        grad_qg_dequant = dequantize_8bit(grad_qg_quant, scales, zeros)
        update_full = projector.project_back(grad_qg_dequant)
        update_full = update_full.view_as(images)
        if targeted and target_label is not None:
            adv_images = images - alpha * update_full.sign()
        else:
            adv_images = images + alpha * update_full.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, 0, 1).detach()
    return images

# ---------------------------
# Main Routine: Process 10 Images One by One (with Optional Targeting)
# ---------------------------
def main():
    dataset = CustomImageNetDataset(root='./imagenet-sample-images', transform=transform, synset2idx=synset2idx)
    num_images = 10
    model = models.inception_v3(pretrained=True).to(device)
    model.eval()
    
    targeted_attack = True
    target_class = "goldfish"
    if targeted_attack:
        try:
            target_idx = idx2label.index(target_class)
        except ValueError:
            print(f"Target class '{target_class}' not found. Using untargeted attack.")
            targeted_attack = False
            target_idx = None
    else:
        target_idx = None

    for i in range(num_images):
        image, label = dataset[i]
        image = image.unsqueeze(0)
        label_tensor = torch.tensor([label])
        true_label = idx2label[label]
        
        image = image.to(device)
        with torch.no_grad():
            output = model(image)
        orig_pred = output.argmax(dim=1).item()
        orig_pred_label = idx2label[orig_pred]
        print(f"Image {i+1}: True label: {true_label}, Original prediction: {orig_pred_label}")
        if targeted_attack:
            print(f"Image {i+1}: Targeted attack intended to push toward: {target_class}")
        
        adv_image = adaptive_pgd_attack_8bit(model, image, label_tensor, eps=0.3, alpha=2/255,
                                             iters=40, device=device, targeted=targeted_attack, target_label=target_idx)
        with torch.no_grad():
            output_adv = model(adv_image)
        adv_pred = output_adv.argmax(dim=1).item()
        adv_pred_label = idx2label[adv_pred]
        print(f"Image {i+1}: Adversarial prediction: {adv_pred_label}")
        
        grid_orig = torchvision.utils.make_grid(image.cpu(), normalize=True)
        grid_adv = torchvision.utils.make_grid(adv_image.cpu(), normalize=True)
        # Rotate images 90 degrees clockwise
        orig_np = np.rot90(torch.transpose(grid_orig, 0, 2).numpy(), k=-1)
        adv_np = np.rot90(torch.transpose(grid_adv, 0, 2).numpy(), k=-1)
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.title(f"Original\nPred: {orig_pred_label}")
        plt.imshow(orig_np)
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.title(f"Adversarial\nPred: {adv_pred_label}")
        plt.imshow(adv_np)
        plt.axis('off')
        plt.show()
    
if __name__ == '__main__':
    main()
