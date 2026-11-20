import pytest
import torch

from emscan._functions import (
    _ncc,
    coerce_ndim,
    crop_or_pad_from_px_sizes,
    crop_to,
    ft_and_shift,
    gaussian_window,
    ift_and_shift,
    normalize,
    pad_to,
    resize,
    rotations,
)


def test_normalize():
    img = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = normalize(img, dim=1, inplace=False)

    assert torch.allclose(out.mean(dim=1), torch.zeros(2), atol=1e-6)
    assert torch.allclose(out.std(dim=1), torch.ones(2), atol=1e-6)
    assert torch.equal(img, torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

    img_copy = img.clone()
    assert normalize(img_copy, dim=1) is img_copy
    assert not torch.isnan(normalize(torch.tensor([1.0, float("nan"), 3.0]))).any()


def test_ft_ift_roundtrip():
    img = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    ft = ft_and_shift(img)

    assert torch.is_complex(ft)
    assert torch.allclose(ift_and_shift(ft).real, img, atol=1e-5)

    expected_dc = torch.zeros((4, 4), dtype=torch.complex64)
    expected_dc[2, 2] = 16
    assert torch.allclose(ft_and_shift(torch.ones_like(img)), expected_dc, atol=1e-5)


def test_coerce_ndim():
    img = torch.ones((4, 4))

    assert coerce_ndim(img, 2) is img
    assert coerce_ndim(img, 4).shape == (1, 1, 4, 4)
    with pytest.raises(ValueError):
        coerce_ndim(img, 1)


def test_gaussian_window():
    window = gaussian_window((5, 7))

    assert window.shape == (5, 7)
    assert window.max() == 1
    assert window.min() == 0
    assert torch.allclose(window, window.flip(0).flip(1))
    assert gaussian_window((5, 7), sigmas=(1, 2)).shape == (5, 7)


def test_crop_to_preserves_center():
    img = torch.arange(25, dtype=torch.float32).reshape(5, 5)

    cropped = crop_to(img, (3, 3))
    assert cropped.shape == (3, 3)
    assert torch.equal(cropped, img[1:4, 1:4])
    assert torch.equal(crop_to(img, (3, 3), dim=(1,)), img[:, 1:4])


def test_pad_to_preserves_center():
    img = torch.arange(9, dtype=torch.float32).reshape(3, 3)

    padded = pad_to(img, (5, 5))
    assert padded.shape == (5, 5)
    assert torch.equal(padded[1:4, 1:4], img)
    assert padded.sum() == img.sum()
    assert torch.equal(pad_to(img, (5, 5), dim=(1,)), torch.nn.functional.pad(img, (1, 1)))


def test_resize():
    ft = torch.ones((8, 8))

    assert resize(ft, (8, 8)) is ft
    assert resize(ft, (4, 4)).shape == (4, 4)
    assert resize(ft, (16, 16)).shape == (16, 16)
    with pytest.raises(NotImplementedError):
        resize(ft, (4, 16))


def test_crop_or_pad_from_px_sizes():
    ft = torch.ones((8, 8))

    assert crop_or_pad_from_px_sizes(ft, 2, 4).shape == (4, 4)
    assert crop_or_pad_from_px_sizes(ft, 4, 2).shape == (16, 16)
    assert crop_or_pad_from_px_sizes(ft, (4, 4), (4, 4)) is ft


def test_rotations():
    img = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    rotated = dict(rotations(img, [0, 90]))

    assert list(rotated) == [0, 90]
    assert all(r.shape == img.shape for r in rotated.values())
    assert torch.allclose(rotated[0], img * gaussian_window(img.shape))
    assert not torch.allclose(rotated[90], rotated[0])

    complex_img = img + 1j * img.flip(0)
    complex_rot = dict(rotations(complex_img, [0]))[0]
    assert torch.is_complex(complex_rot)
    assert torch.allclose(complex_rot, complex_img * gaussian_window(img.shape))


def test_ncc_finds_matching_projection():
    generator = torch.Generator().manual_seed(0)
    window = gaussian_window((16, 16))
    projections = torch.randn(3, 16, 16, generator=generator) * window
    projections = normalize(projections, dim=(1, 2), inplace=False)
    classes = torch.stack((projections[1], projections[0]))

    result = _ncc(classes, projections, angle_step=90)

    assert result["indices"] == {0: 1, 1: 0}
    assert result["angles"] == {0: 0.0, 1: 0.0}
    assert min(result["values"].values()) > 0.9
