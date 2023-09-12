from functools import reduce
from operator import mul

import healpy
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
    img -= img.mean(dim=dim, keepdim=True)
    img /= img.std(dim=dim, keepdim=True)
    return img


def rotations(img, degree_range, center=None):
    """Generate a number of rotations from an image and a list of angles.

    degree range: iterable of degrees (counterclockwise).
    """
    if center is None:
        center = list(np.array(img.shape) // 2)

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
            yield angle, real + (1j * imag)
        else:
            yield angle, rotate(
                img[None],
                angle,
                center=center,
                interpolation=InterpolationMode.BILINEAR,
            )[0]


def coerce_ndim(img, ndim):
    """Coerce image dimensionality if smaller than ndim."""
    if img.ndim > ndim:
        raise ValueError(f"image has too high dimensionality ({img.ndim})")
    while img.ndim < ndim:
        img = img[None]
    return img


def rotated_projections(img, healpix_order=2, dtype=None):
    """Generate rotated projections of a map."""
    # TODO: sinc function to avoid edge artifacts
    if dtype is not None:
        img = img.astype(dtype)

    ft = fftshift(fftn(fftshift(img)))

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
        ft, positions=pos, orientations=rot, grid_shape=grid_shape
    ).squeeze()

    return ifftshift(
        ifftn(ifftshift(slices, axes=(1, 2)), axes=(1, 2)), axes=(1, 2)
    ).real


def gaussian_window(shape, sigmas=1):
    """Generate a gaussian_window of the given shape and sigmas."""
    sigmas = np.broadcast_to(sigmas, len(shape))
    windows = [gaussian(n, s) for n, s in zip(shape, sigmas)]
    tensors = [torch.tensor(w, device="cuda") for w in np.ix_(*windows)]
    return reduce(mul, tensors)


def fourier_resize(img, target_shape, dim=None):
    """Bin or unbin image(s) by fourier cropping or padding."""
    if dim is None:
        dim = tuple(range(img.ndim))

    ft = fftshift(fftn(img, dim=dim), dim=dim)

    target_shape = np.array(target_shape)
    if np.all(target_shape <= img.shape):
        edge_crop = ((img.shape - target_shape) // 2).astype(int)
        crop_slice = tuple(
            slice(edge_crop[d], -edge_crop[d]) if d in dim else slice(None)
            for d in range(img.ndim)
        )
        ft_resized = ft[crop_slice]
        cropped_shape = np.array(ft_resized.shape)
        # needed for edge artifacts
        window = gaussian_window(cropped_shape, cropped_shape / 4)
        ft_resized *= window
    elif np.all(target_shape >= img.shape):
        edge_pad = ((target_shape - img.shape) // 2).astype(int)
        padding = tuple(
            edge_pad[d] if d in dim else 0 for d in np.arange(img.ndim).repeat(2)
        )
        ft_resized = torch.nn.functional.pad(ft, padding)
    else:
        raise NotImplementedError("cannot pad and crop at the same time")

    return ifftn(ifftshift(ft_resized, dim=dim), dim=dim).real


def rescale(img, px_size_in, px_size_out, dim=None):
    """Rescale an image given before/after pixel sizes."""
    ratio = px_size_in / px_size_out
    target_shape = np.round(np.array(img.shape) * ratio / 2) * 2
    return fourier_resize(img, target_shape, dim=dim)


def match_px_size(img1, img2, px_size1, px_size2, dim=None):
    """Match two images' pixel sizes by binning the one with higher resolution."""
    ratio = px_size1 / px_size2
    if ratio > 1:
        return img1, rescale(img2, px_size2, px_size1, dim=dim), px_size1
    else:
        return rescale(img1, px_size1, px_size2, dim=dim), img2, px_size2


def cumsum_nD(img, dim=None):
    """Calculate ndimensional cumsum."""
    import torch

    dim = list(range(img.ndim)) if dim is None else list(dim)
    out = img.cumsum(dim[0])
    for d in dim[1:]:
        torch.cumsum(out, dim=d, out=out)
    return out


def correlate_rotations(img, features, angle_step=5):
    """Fast cross correlation of all elements of features and the images.

    Performs also rotations. Input fts must be fftshifted+fftd+fftshifted.
    """
    features = coerce_ndim(features, 3)

    shape1 = np.array(img.shape[1:])
    shape2 = np.array(features.shape[1:])

    edges = (np.abs(shape1 - shape2) // 2).astype(int)
    center_slice = tuple(slice(e, -e) if e else slice(None) for e in edges)

    if np.all(shape1 >= shape2):
        img = img[center_slice]
    elif np.all(shape1 <= shape2):
        features = features[:, *center_slice]
    else:
        raise ValueError("weird shapes")

    img = normalize(img)
    features = normalize(features, dim=(1, 2))

    img_ft = fftshift(fftn(fftshift(img)))
    feat_fts = fftshift(fftn(fftshift(features, dim=(1, 2)), dim=(1, 2)), dim=(1, 2))

    img_ft_conj = img_ft.conj()
    img_autocc = torch.abs(ifftshift(ifftn(ifftshift(img_ft * img_ft_conj))))
    feat_autocc = torch.abs(
        ifftshift(
            ifftn(ifftshift(feat_fts * feat_fts.conj(), dim=(1, 2)), dim=(1, 2)),
            dim=(1, 2),
        )
    )
    norm_denominators = torch.sqrt(img_autocc.amax()) * torch.sqrt(
        feat_autocc.amax(dim=(1, 2))
    )

    for feat_ft, denom in zip(feat_fts, norm_denominators):
        best_cc = 0
        for _, feat_ft_rot in rotations(feat_ft, range(0, 360, angle_step)):
            cc = torch.abs(ifftshift(ifftn(ifftshift(img_ft_conj * feat_ft_rot))))
            best_cc = max(torch.amax(cc / denom), best_cc)
        yield best_cc.item()
