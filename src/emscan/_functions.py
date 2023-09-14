from functools import reduce
from itertools import chain
from math import ceil, floor
from operator import mul

import healpy
import mrcfile
import numpy as np
import torch
from morphosamplers.sampler import sample_subvolumes
from scipy.signal.windows import gaussian
from scipy.spatial.transform import Rotation
from torch.fft import fftn, fftshift, ifftn, ifftshift
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import rotate


def normalize(img, dim=None):
    """Normalize images to std=1 and mean=0."""
    if not torch.is_complex(img):
        img = img.to(torch.float32)
    img -= img.mean(dim=dim, keepdim=True)
    img /= img.std(dim=dim, keepdim=True)
    return img


def ft_and_shift(img, dim=None):
    return fftshift(fftn(fftshift(img, dim=dim), dim=dim), dim=dim)


def ift_and_shift(img, dim=None):
    return ifftshift(ifftn(ifftshift(img, dim=dim), dim=dim), dim=dim)


def rotations(img, degree_range, center=None):
    """Generate a number of rotations from an image and a list of angles.

    degree range: iterable of degrees (counterclockwise).
    """
    if center is None:
        center = list(np.array(img.shape) // 2)

    rot = None
    for angle in degree_range:
        if torch.is_complex(img):
            real = rotate(
                img.real[None],
                angle,
                center=center,
                interpolation=InterpolationMode.BILINEAR,
            )[0]
            imag = rotate(
                img.imag[None],
                angle,
                center=center,
                interpolation=InterpolationMode.BILINEAR,
            )[0]
            rot = angle, real + (1j * imag)
        else:
            rot = (
                angle,
                rotate(
                    img[None],
                    angle,
                    center=center,
                    interpolation=InterpolationMode.BILINEAR,
                )[0],
            )
        yield rot


def coerce_ndim(img, ndim):
    """Coerce image dimensionality if smaller than ndim."""
    if img.ndim > ndim:
        raise ValueError(f"image has too high dimensionality ({img.ndim})")
    while img.ndim < ndim:
        img = img[None]
    return img


def rotated_projection_fts(ft, healpix_order=2):
    """Generate rotated projections fts of a map."""
    # TODO: sinc function to avoid edge artifacts

    nside = healpy.order2nside(healpix_order)
    npix = healpy.nside2npix(nside)
    # only half the views are needed, cause projection symmetry
    angles = healpy.pix2ang(nside, np.arange(npix // 2))
    angles = np.stack(angles).T
    rot = Rotation.from_euler("xz", angles)
    pos = np.array(ft.shape) / 2
    # get an even size grid that ensures we include everything
    grid_size = int(np.ceil(np.linalg.norm(ft.shape)))
    grid_size += grid_size % 2
    grid_shape = (grid_size, grid_size, 1)
    slices = sample_subvolumes(
        np.array(ft), positions=pos, orientations=rot, grid_shape=grid_shape
    ).squeeze()

    return torch.from_numpy(slices)


def gaussian_window(shape, sigmas=1, device=None):
    """Generate a gaussian_window of the given shape and sigmas."""
    sigmas = np.broadcast_to(sigmas, len(shape))
    windows = [gaussian(n, s) for n, s in zip(shape, sigmas)]
    tensors = [torch.tensor(w, device=device) for w in np.ix_(*windows)]
    return reduce(mul, tensors)


def resize(ft, target_shape, dim=None):
    """Bin or unbin image(s) ft by fourier cropping or padding."""
    if dim is None:
        dim = tuple(range(ft.ndim))

    target_shape = np.array(target_shape)
    if np.all(target_shape <= ft.shape):
        ft_resized = crop_to(ft, target_shape, dim=dim)
        # needed for edge artifacts
        cropped_shape = np.array(ft_resized.shape)
        window = gaussian_window(cropped_shape, cropped_shape / 3)
        ft_resized *= window
        return ft_resized
    elif np.all(target_shape >= ft.shape):
        return pad_to(ft, target_shape, dim=dim)
    else:
        raise NotImplementedError("cannot pad and crop at the same time")


def crop_to(img, target_shape, dim=None):
    if dim is None:
        dim = tuple(range(img.ndim))
    edge_crop = ((np.array(img.shape) - np.array(target_shape)) // 2).astype(int)
    crop_slice = tuple(
        slice(edge_crop[d], -edge_crop[d]) if d in dim else slice(None)
        for d in range(img.ndim)
    )
    return img[crop_slice]


def pad_to(img, target_shape, dim=None):
    if dim is None:
        dim = tuple(range(img.ndim))
    edge_pad = ((np.array(target_shape) - np.array(img.shape)) / 2).astype(int)
    pad_left = tuple(floor(edge_pad[d]) if d in dim else 0 for d in range(img.ndim))
    pad_right = tuple(ceil(edge_pad[d]) if d in dim else 0 for d in range(img.ndim))
    padding = tuple(chain.from_iterable(zip(pad_right, pad_left)))[::-1]
    return torch.nn.functional.pad(img, padding)


def rescale_ft(ft, px_size_in, px_size_out, dim=None):
    """Rescale an image's ft given before/after pixel sizes."""
    ratio = np.array(px_size_in) / np.array(px_size_out)
    if np.allclose(ratio, 1):
        return ft
    target_shape = np.round(np.array(ft.shape) * ratio / 2) * 2
    return resize(ft, target_shape, dim=dim)


def correlate_rotations(img_ft, proj_fts, angle_step=5):
    """Fast cross correlation of all elements of projections and the image.

    Performs also rotations. Input fts must be fftshifted+fftd+fftshifted.
    """
    shape1 = tuple(img_ft.shape[1:])
    shape2 = tuple(proj_fts.shape[1:])
    if not np.allclose(shape1, shape2):
        raise RuntimeError(
            f"correlate requires the same shape, got {shape1} and {shape2}"
        )

    img_autocc = torch.abs(ift_and_shift(img_ft * img_ft.conj()))
    proj_autocc = torch.abs(ift_and_shift(proj_fts * proj_fts.conj(), dim=(1, 2)))
    denoms = torch.sqrt(img_autocc.amax()) * torch.sqrt(proj_autocc.amax(dim=(1, 2)))

    best_ccs = torch.zeros(len(proj_fts), device=proj_fts.device)
    for _, img_ft_rot in rotations(img_ft, range(0, 360, angle_step)):
        ccs = torch.abs(ift_and_shift(img_ft_rot[None] * proj_fts.conj(), dim=(1, 2)))
        max_norm_ccs = torch.amax(ccs, dim=(1, 2)) / denoms
        best_ccs = torch.amax(torch.stack([max_norm_ccs, best_ccs]), dim=0)
    return best_ccs


def compute_cc(cls_path, agg_path, device=None, bin_resolution=4):
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    with torch.cuda.device(device):
        with mrcfile.mmap(cls_path) as mrc:
            class_data = torch.tensor(mrc.data)
            class_px_size = mrc.voxel_size.x.item()

        class_data = normalize(coerce_ndim(class_data, 3), dim=(1, 2))
        entry_data = torch.load(agg_path, map_location=device)

        class_ft = ft_and_shift(class_data, dim=(1, 2))
        class_ft = rescale_ft(class_ft, class_px_size, bin_resolution, dim=(1, 2))
        class_ft = pad_to(class_ft, entry_data.shape, dim=(1, 2))

        corr_values = {}
        for cls_idx, cls in enumerate(class_ft):
            ccs = correlate_rotations(cls, entry_data)
            for i, entry in enumerate(entry_data.entries):
                start = i * entry_data.entry_stride
                stop = (i + 1) * entry_data.entry_stride
                corr_values.setdefault(cls_idx, {})[entry] = torch.max(
                    ccs[start:stop]
                ).item()
    return corr_values
