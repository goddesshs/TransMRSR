import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import log10, sqrt
from torch.optim import lr_scheduler
import scipy.io as sio
import pdb
import cv2
import mmcv
def get_nonlinearity(name):
    """Helper function to get non linearity module, choose from relu/softplus/swish/lrelu"""
    if name == 'relu':
        return nn.ReLU(inplace=True)
    elif name == 'softplus':
        return nn.Softplus()
    elif name == 'swish':
        return Swish(inplace=True)
    elif name == 'lrelu':
        return nn.LeakyReLU()


class Swish(nn.Module):
    def __init__(self, inplace=False):
        """The Swish non linearity function"""
        super().__init__()
        self.inplace = True

    def forward(self, x):
        if self.inplace:
            x.mul_(F.sigmoid(x))
            return x
        else:
            return x * F.sigmoid(x)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_scheduler(optimizer, opts, last_epoch=-1):
    if 'lr_policy' not in opts or opts.lr_policy == 'constant':
        scheduler = None
    elif opts.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opts.step_size,
                                        gamma=opts.gamma, last_epoch=last_epoch)
    elif opts.lr_policy == 'lambda':
        def lambda_rule(ep):
            lr_l = 1.0 - max(0, ep - opts.epoch_decay) / float(opts.n_epochs - opts.epoch_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=last_epoch)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opts.lr_policy)
    return scheduler

def get_recon_loss(opts):
    loss = None
    if opts['recon'] == 'L2':
        loss = nn.MSELoss()
    elif opts['recon'] == 'L1':
        loss = nn.L1Loss()

    return loss

def rmse(org_img: np.ndarray, pred_img: np.ndarray, max_p: int = 4095, input_order: str='CHW') -> float:
    """
    Root Mean Squared Error
    Calculated individually for all bands, then averaged
    """
    # _assert_image_shapes_equal(org_img, pred_img, "RMSE")
    org_img = reorder_image(org_img, input_order=input_order)
    pred_img = reorder_image(pred_img, input_order=input_order)
    org_img = org_img.astype(np.float32)
    
    # if image is a gray image - add empty 3rd dimension for the .shape[2] to exist
    if org_img.ndim == 2:
        org_img = np.expand_dims(org_img, axis=-1)
    
    rmse_bands = []
    for i in range(org_img.shape[2]):
        dif = np.subtract(org_img[:, :, i], pred_img[:, :, i])
        m = np.mean(np.square(dif))
        s = np.sqrt(m)
        rmse_bands.append(s)

    return np.mean(rmse_bands)
# def rmse(sr_image, gt_image, peak_signal=255):

#     mse = np.mean((sr_image - gt_image)**2)
#     return np.sqrt(mse)

def psnr(sr_image, gt_image):
    # assert sr_image.size(0) == gt_image.size(0) == 1
    print('img_size: ', sr_image.size(0), gt_image.size(0))
    peak_signal = 255

    mse = (sr_image - gt_image).pow(2).mean().item()
    if mse==0:
        return 0
    return 10 * log10(peak_signal ** 2 / mse)


def _ssim(img1, img2):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1, img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: ssim result.
    """

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def reorder_image(img, input_order='HWC'):
    """Reorder images to 'HWC' order.

    If the input_order is (h, w), return (h, w, 1);
    If the input_order is (c, h, w), return (h, w, c);
    If the input_order is (h, w, c), return as it is.

    Args:
        img (ndarray): Input image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            If the input image shape is (h, w), input_order will not have
            effects. Default: 'HWC'.

    Returns:
        ndarray: reordered image.
    """

    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    if len(img.shape) == 2:
        img = img[..., None]
        return img
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img

def ssim(img1, img2, crop_border=0, input_order='HWC', convert_to=None):
    """Calculate SSIM (structural similarity).

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edges of an image. These
            pixels are not involved in the SSIM calculation. Default: 0.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        convert_to (str): Whether to convert the images to other color models.
            If None, the images are not altered. When computing for 'Y',
            the images are assumed to be in BGR order. Options are 'Y' and
            None. Default: None.

    Returns:
        float: ssim result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are different: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    # if isinstance(convert_to, str) and convert_to.lower() == 'y':
    #     img1, img2 = img1.astype(np.float32), img2.astype(np.float32)
    #     img1 = mmcv.bgr2ycbcr(img1 / 255., y_only=True) * 255.
    #     img2 = mmcv.bgr2ycbcr(img2 / 255., y_only=True) * 255.
    #     img1 = np.expand_dims(img1, axis=2)
    #     img2 = np.expand_dims(img2, axis=2)
    # elif convert_to is not None:
    #     raise ValueError('Wrong color model. Supported values are '
    #                      '"Y" and None')

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, None]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, None]

    ssims = []
    for i in range(img1.shape[2]):
        ssims.append(_ssim(img1[..., i], img2[..., i]))
    return np.array(ssims).mean()

def mse(sr_image, gt_image):
    assert sr_image.size(0) == gt_image.size(0) == 1

    mse = (sr_image - gt_image).pow(2).mean().item()

    return mse


''''
K-Space
'''
def data_consistency(k, k0, mask, noise_lvl=None):
    """
    k    - input in k-space [b,w,h,2] need to [b,2,w,h]
    k0   - initially sampled elements in k-space
    dc_mask - corresponding nonzero location
    """
    k = k.permute(0,3,1,2)
    v = noise_lvl
    if v:  # noisy case
        out = (1 - mask) * k + mask * (k + v * k0) / (1 + v)
    else:  # noiseless case
        out = (1 - mask) * k + mask * k0
    return out


class DataConsistencyInKspace_I(nn.Module):
    """ Create data consistency operator

    Warning: note that FFT2 (by the default of torch.fft) is applied to the last 2 axes of the input.
    This method detects if the input tensor is 4-dim (2D data) or 5-dim (3D data)
    and applies FFT2 to the (nx, ny) axis.

    """

    def __init__(self, noise_lvl=None):
        super(DataConsistencyInKspace_I, self).__init__()
        self.noise_lvl = noise_lvl

    def forward(self, *input, **kwargs):
        return self.perform(*input)

    def perform(self, x, k0, mask):
        """
        x    - input in image domain, of shape (n, 2, nx, ny)
        k0   - initially sampled elements in k-space
        dc_mask - corresponding nonzero location
        """

        if x.dim() == 4: # input is 2D
            x = x.permute(0, 2, 3, 1) #[n,w,h,2]
        else:
            raise ValueError("error in data consistency layer!")

        k = fft2(x)
        # out = data_consistency(k, k0, dc_mask.repeat(1, 1, 1, 2), self.noise_lvl)
        out = data_consistency(k, k0, mask, self.noise_lvl)
        x_res = ifft2(out) #[b,2,w,h]

        if x.dim() == 4:
            # x_res = x_res.permute(0, 3, 1, 2)
            x_res = x_res
        else:
            raise ValueError("Iuput dimension is wrong, it has to be a 2D input!")

        return x_res, out


class DataConsistencyInKspace_K(nn.Module):
    """ Create data consistency operator

    Warning: note that FFT2 (by the default of torch.fft) is applied to the last 2 axes of the input.
    This method detects if the input tensor is 4-dim (2D data) or 5-dim (3D data)
    and applies FFT2 to the (nx, ny) axis.

    """

    def __init__(self, noise_lvl=None):
        super(DataConsistencyInKspace_K, self).__init__()
        self.noise_lvl = noise_lvl

    def forward(self, *input, **kwargs):
        return self.perform(*input)

    def perform(self, k, k0, mask):
        """
        k    - input in frequency domain, of shape (n, 2, nx, ny)
        k0   - initially sampled elements in k-space
        dc_mask - corresponding nonzero location
        """

        if k.dim() == 4:  # input is 2D [b,2,w,h]
            k = k.permute(0, 2, 3, 1) #[b,w,h,2]
        else:
            raise ValueError("error in data consistency layer!")

        out = data_consistency(k, k0, mask, self.noise_lvl) #[b,2,w,h]
        x_res = ifft2(out) #[b,2,w,h]
        # ========
        # ks_net_fin_out = x_res.cpu().detach().numpy()
        # sio.savemat('ks_net_fin_out.mat', {'data': ks_net_fin_out});
        # ========

        if k.dim() == 4:
            # x_res = x_res.permute(0, 3, 1, 2)
            x_res = x_res
        else:
            raise ValueError("Iuput dimension is wrong, it has to be a 2D input!")

        return x_res, out

# Basic functions / transforms
def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.
    Args:
        data (np.array): Input numpy array
    Returns:
        torch.Tensor: PyTorch version of data
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)
    return torch.from_numpy(data)


def fft2(data):
    """
    Apply centered 2 dimensional Fast Fourier Transform.
    input 2, w, h
    output w, h, 2
    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.[w,h,2]
    Returns:
        torch.Tensor: The FFT of the input.
    """
    # data = data.permute(1,2,0) #[w,h,2]

    # assert data.size(-1) == 2
    torch.fft.fftshift()
    data = ifftshift(data, dim=(-3, -2))
    data = torch.fft(data, 2, normalized=False)
    data = fftshift(data, dim=(-3, -2))

    # data = data.permute(2,0,1) #[2,w,h]
    return data

def fft2_net(data):
    """
    Apply centered 2 dimensional Fast Fourier Transform.
    input b,2, w, h need to b,w, h, 2
    output b,2, w, h
    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.[w,h,2]
    Returns:
        torch.Tensor: The FFT of the input.
    """
    # data = data.permute(1,2,0) #[w,h,2]

    # assert data.size(-1) == 2
    # print(data.size())
    data = data.permute(0, 2, 3, 1) #[b,w,h,2]
    data = ifftshift(data, dim=(-3, -2))
    data = torch.fft(data, 2, normalized=False)
    data = fftshift(data, dim=(-3, -2))
    data = data.permute(0, 3, 1, 2) #[b,2,w,h]
    # data = data.permute(2,0,1) #[2,w,h]
    return data


def ifft2(data):
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.
    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.
    Returns:
        torch.Tensor: The IFFT of the input [b,2,w,h].
    """
    # assert data.size(-1) == 2
    data = data.permute(0,2,3,1)#[0,w,h,2]
    data = ifftshift(data, dim=(-3, -2))
    data = torch.ifft(data, 2, normalized=False)
    data = fftshift(data, dim=(-3, -2))
    data = data.permute(0, 3, 1, 2)  # [0,2,w,h]
    return data


def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)


def fftshift(x, dim=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)


def ifftshift(x, dim=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return roll(x, shift, dim)


def complex_abs(data):
    """
    Compute the absolute value of a complex valued input tensor.
    Args:
        data (torch.Tensor): A complex valued tensor, where the size of the final dimension
            should be 2.
    Returns:
        torch.Tensor: Absolute value of data
    """
    assert data.size(-1) == 2
    return (data ** 2).sum(dim=-1).sqrt()


def complex_abs_eval(data):
    assert data.size(1) == 2
    return (data[:, 0:1, :, :] ** 2 + data[:, 1:2, :, :] ** 2).sqrt()


def to_spectral_img(data):
    """
    Compute the spectral images of a kspace data
    with keeping each column for creation of one spectral image
    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.
    Returns:
        torch.Tensor: The IFFT of the input.
    """
    assert data.size(-1) == 2

    spectral_vol = torch.zeros([data.size(-2), data.size(-2), data.size(-2)])

    for i in range(data.size(-2)):
        kspc1 = torch.zeros(data.size())
        kspc1[:, i, :] = data[:, i, :]
        img1 = ifft2(kspc1)
        img1_abs = complex_abs(img1)

        spectral_vol[i, :, :] = img1_abs

    return spectral_vol


def tensor_transform_reverse(image):
    assert image.dim() == 4
    moco_input = torch.zeros(image.size()).type_as(image)
    moco_input[:,0,:,:] = image[:,0,:,:] * 0.229 + 0.485
    moco_input[:,1,:,:] = image[:,1,:,:] * 0.224 + 0.456
    moco_input[:,2,:,:] = image[:,2,:,:] * 0.225 + 0.406
    return moco_input