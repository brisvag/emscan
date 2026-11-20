import gc
import inspect
import os
import re
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import lru_cache, reduce
from itertools import chain
from math import ceil, floor
from operator import mul
from pathlib import Path

import mrcfile
import numpy as np
import pandas as pd
import requests
import sh
import torch
import xmltodict
from morphosamplers.sampler import sample_subvolumes
from rich import print
from scipy.signal.windows import gaussian
from scipy.spatial.transform import Rotation
from torch.fft import fftn, fftshift, ifftn, ifftshift
from torch.multiprocessing import set_start_method
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import rotate

BIN_RESOLUTION = 4
GOLDEN_RATIO = (1 + np.sqrt(5)) / 2


def _print_tensors(depth=2):
    """Print info about tensor variables locally and nonlocally."""
    print("=" * 80)
    for obj in gc.get_objects():
        frame = inspect.currentframe()
        frames = []
        for _i in range(depth):
            frame = frame.f_back
            frames.append(frame)
        try:
            if torch.is_tensor(obj) or (
                hasattr(obj, "data") and torch.is_tensor(obj.data)
            ):
                for gobj, gobjid in globals().items():
                    if gobjid is obj and gobj not in ("obj", "gobjid"):
                        print("GLOBAL:\t", gobj, obj.size(), obj.data_ptr())
                for frame in frames:
                    fname = frame.f_code.co_name
                    locs = frame.f_locals
                    for gobj, gobjid in locs.items():
                        if gobjid is obj and gobj not in ("obj", "gobjid"):
                            print(f"{fname}():\t", gobj, obj.size(), obj.data_ptr())
        except:  # noqa
            pass
    print("=" * 80)


def normalize(img, dim=None, inplace=True):
    """Normalize images to std=1 and mean=0."""
    if not torch.is_complex(img):
        img = img.to(torch.float32, copy=not inplace)
    elif not inplace:
        img = img.copy()
    img -= img.mean(dim=dim, keepdim=True)
    img /= img.std(dim=dim, keepdim=True)
    if img.isnan().any():
        img = torch.nan_to_num(img, out=img)
    return img


def ft_and_shift(img, dim=None):
    # first move the image to be center of the origin (ifftshift), which improves
    # interpolation artifacts later (ftshift is equivalent to multiplying
    # with a checkerboard, which messes up interpolation real bad)
    return fftshift(fftn(ifftshift(img, dim=dim), dim=dim), dim=dim)


def ift_and_shift(img, dim=None):
    # undo the above (note the order is the same cause it's actually inverted)
    return fftshift(ifftn(ifftshift(img, dim=dim), dim=dim), dim=dim)


def rotate_by(arr, ang, center=None):
    if center is None:
        # torchvision expects the center as (x, y), i.e. (col, row)
        center = list(np.array(arr.shape)[::-1] // 2)

    return rotate(
        arr[None],
        float(ang),
        center=center,
        interpolation=InterpolationMode.BILINEAR,
    )[0]


def rotations(img, degree_range, center=None):
    """Generate a number of rotations from an image and a list of angles.

    degree range: iterable of degrees (counterclockwise).
    """
    gauss = gaussian_window(img.shape, device=img.device)

    for angle in degree_range:
        if torch.is_complex(img):
            real = rotate_by(img.real, angle, center)
            imag = rotate_by(img.imag, angle, center)
            rot = real + (1j * imag)
            del real, imag
        else:
            rot = rotate_by(img, angle, center)
        yield angle, rot * gauss
        del rot


def coerce_ndim(img, ndim):
    """Coerce image dimensionality if smaller than ndim."""
    if img.ndim > ndim:
        raise ValueError(f"image has too high dimensionality ({img.ndim})")
    while img.ndim < ndim:
        img = img[None]
    return img


@lru_cache
def _rotations_hemisphere_fibonacci():
    # http://extremelearning.com.au/how-to-evenly-distribute-points-on-a-
    # sphere-more-effectively-than-the-canonical-fibonacci-lattice/
    # Note: not using epsilon we ensure we get the Z axis, which is a common
    # aligment for maps. Packing is slightly worse, but still better than healpix

    N = 300  # THIS IS FIXED ONCE GENERATED THE DB!
    i = np.arange(N)[: N // 2]  # only take half cause projection symmetry
    phi = 2 * np.pi * i / GOLDEN_RATIO
    theta = np.arccos(1 - 2 * i / (N - 1))
    angles = np.stack([phi, theta], axis=1)
    return Rotation.from_euler("ZX", angles)


def rotated_projection_fts(ft):
    """Generate rotated projections fts of a map."""
    # TODO: sinc function to avoid edge artifacts

    rot = _rotations_hemisphere_fibonacci()
    pos = np.array(ft.shape) // 2  # using // is needed to center ctf for odd images
    grid_size = (ft.shape[0], ft.shape[0], 1)
    slices = sample_subvolumes(
        np.array(ft), positions=pos, orientations=rot, grid_shape=grid_size, fill_value=0,
    ).squeeze()

    return torch.from_numpy(slices)


def gaussian_window(shape, sigmas=None, device=None):
    """Generate a gaussian_window of the given shape and sigmas, ensuring it goes to zero."""
    if sigmas is None:
        sigmas = np.array(shape)
    sigmas = np.broadcast_to(sigmas, len(shape))
    windows = [gaussian(n, s) for n, s in zip(shape, sigmas, strict=True)]
    mins = np.array([min(w) for w in windows])
    # true minimum of the separable product, so the window reaches exactly 0
    zero = np.prod(mins)
    tensors = [torch.tensor(w, device=device) for w in np.ix_(*windows)]
    window = reduce(mul, tensors)
    window -= zero
    window = torch.clip(window, 0, 1)
    window /= window.max()
    return window


def crop_to(img, target_shape, dim=None):
    if dim is None:
        dim = tuple(range(img.ndim))
    # ensure we round the cropping correctly to keeo the center in the right place
    edge_crop = (np.array(img.shape) - np.array(target_shape)) / 2
    crop_left = []
    crop_right = []
    for d in range(img.ndim):
        if d not in dim:
            crop_left.append(None)
            crop_right.append(None)
            continue
        # rounding must depend on the parity of the dimension being cropped
        if img.shape[d] % 2:
            left, right = floor(edge_crop[d]), ceil(edge_crop[d])
        else:
            left, right = ceil(edge_crop[d]), floor(edge_crop[d])
        crop_left.append(left or None)
        crop_right.append(-right or None)
    crop_slice = tuple(
        slice(*crops) for crops in zip(crop_left, crop_right, strict=True)
    )
    return img[crop_slice]


def pad_to(img, target_shape, dim=None, value=0):
    if dim is None:
        dim = tuple(range(img.ndim))
    # ensure we round the padding correctly to keeo the center in the right place
    edge_pad = (np.array(target_shape) - np.array(img.shape)) / 2
    pad_left = []
    pad_right = []
    for d in range(img.ndim):
        if d not in dim:
            pad_left.append(0)
            pad_right.append(0)
            continue
        # rounding must depend on the parity of the dimension being padded
        if img.shape[d] % 2:
            left, right = ceil(edge_pad[d]), floor(edge_pad[d])
        else:
            left, right = floor(edge_pad[d]), ceil(edge_pad[d])
        pad_left.append(left)
        pad_right.append(right)
    padding = tuple(chain.from_iterable(zip(pad_right, pad_left, strict=True)))[::-1]
    padded = torch.nn.functional.pad(img, padding, value=value)
    return padded


def resize(ft, target_shape, dim=None):
    """Bin or unbin image(s) ft by fourier cropping or padding."""
    if dim is None:
        dim = tuple(range(ft.ndim))

    target_shape = np.array(target_shape)
    input_shape = np.array(ft.shape)
    target_shape_dims = target_shape[list(dim)]
    input_shape_dims = input_shape[list(dim)]
    if np.all(target_shape_dims == input_shape_dims):
        return ft
    elif np.all(target_shape_dims < input_shape_dims):
        return crop_to(ft, target_shape, dim=dim)
    elif np.all(target_shape_dims > input_shape_dims):
        return pad_to(ft, target_shape, dim=dim)
    else:
        raise NotImplementedError("cannot pad and crop at the same time")


def crop_or_pad_from_px_sizes(ft, px_size_in, px_size_out, dim=None):
    """Rescale an image's ft given before/after pixel sizes."""
    ratio = np.array(px_size_in) / np.array(px_size_out)
    target_shape = np.round(np.array(ft.shape) * ratio / 2) * 2
    dim = tuple(range(ft.ndim)) if dim is None else dim
    return resize(ft, target_shape, dim=dim)


def _ncc(S, L, angle_step):
    """Fast cross correlation of all elements of projections and the image.

    Performs also rotations and mirroring to cover all orientations.

    Based on eq (9) from:
    https://w.imagemagick.org/docs/AcceleratedTemplateMatchingUsingLocalStatisticsAndFourierTransforms.pdf
    Optimized to reduce operations (FFTs are precomputed and images are pre-normalized).
    Some stuff is also easier because L[0] and S are forced to have the same shape if S is big.
    """
    # see link above for what is L, LL, S, U, and the equation
    corr_results = {"values": {}, "indices": {}, "angles": {}}

    if S.shape[-1] < L.shape[-1]:
        # crop L
        L = crop_to(L, S.shape, dim=(1, 2))
        L = normalize(L, dim=(1, 2), inplace=False)
    else:
        # cropping, so assume "same size" of S and L
        S = crop_to(S, L.shape, dim=(1, 2))
        S = normalize(S, dim=(1, 2), inplace=False)

    L_ = ft_and_shift(L, dim=(1, 2))
    L_conj = L_.conj()
    LL_ = ft_and_shift(L**2, dim=(1, 2))
    N = S[0].nelement()
    # TODO this is just zeros with N in the center
    U_ = ft_and_shift(torch.ones_like(S[0], device=S.device))

    def _cc(ft1, ft2, ft2_is_conj=False):
        conj = ft2 if ft2_is_conj else ft2.conj()
        return ift_and_shift(ft1 * conj, dim=(1, 2)).real

    denom_inner = (N * _cc(U_, LL_)) - (_cc(U_, L_) ** 2)
    denom = torch.sqrt(torch.clamp_min(denom_inner, 0))
    n_proj = len(L)

    del L, L_, LL_, U_

    def ncc(S_):
        nonlocal denom, L_conj
        # NOTE: this assume S is ft of normalized image (std=1, mean=0)
        # NOTE: L_conj is precomputed to save computations!
        return _cc(S_, L_conj, ft2_is_conj=True) / denom

    for cls_idx, cls in enumerate(S):
        best_ccs = torch.zeros(n_proj, device=S.device)
        best_ang = torch.zeros(n_proj, device=S.device)

        for angle, cls_rot in rotations(cls, np.arange(0, 360, angle_step)):
            # normalize to avoid needing std and mean in ncc calculation
            cls_rot = normalize(cls_rot)
            # also correlate transposed image, because we only have half a sphere of projections
            for cls_flip, flip_sign in ((cls_rot, 1), (cls_rot.T, -1)):
                S_ = ft_and_shift(cls_flip)
                ccs = ncc(S_).amax(dim=(1, 2))
                best_ccs, replaced = torch.max(torch.stack([best_ccs, ccs]), dim=0)
                best_ang[replaced == 1] = angle * flip_sign
                del S_, ccs
            del cls_rot

        best_cc, best_idx = best_ccs.max(dim=0)
        corr_results["values"][cls_idx] = best_cc.item()
        corr_results["indices"][cls_idx] = best_idx.item()
        corr_results["angles"][cls_idx] = best_ang[best_idx.item()].item()
        del cls, best_ccs, best_cc, best_ang
    del L_conj, S, denom
    return corr_results


_CLASS_CACHE: dict = {}


def compute_ncc(cls_path, entry_path, angle_step=5, device=None, fraction=1):
    """Small wrapper for _ncc to load entry in the proper gpu.

    Class data is loaded once per worker process and reused across entries, so only a
    small key crosses the process boundary instead of the full tensor.
    """
    key = (str(cls_path), device, fraction)
    if key not in _CLASS_CACHE:
        _CLASS_CACHE[key] = load_class_data(cls_path, device=device, fraction=fraction)
    S = _CLASS_CACHE[key]
    L = torch.load(entry_path, map_location=S.device, weights_only=True)
    return _ncc(S, L, angle_step)


def load_class_data(cls_path, device=None, fraction=1):
    with mrcfile.mmap(cls_path) as mrc:
        class_data = coerce_ndim(torch.tensor(mrc.data, device=device), 3)
        class_px_size = mrc.voxel_size.x.item()

    class_data = normalize(class_data, dim=(1, 2))
    class_data *= gaussian_window(class_data.shape[1:], device=device)
    class_data = normalize(class_data, dim=(1, 2))

    # crop/pad ft to match pixel size
    class_ft = ft_and_shift(class_data, dim=(1, 2))

    class_ft = crop_or_pad_from_px_sizes(
        class_ft, class_px_size, BIN_RESOLUTION, dim=(1, 2)
    )
    class_data = ift_and_shift(class_ft, dim=(1, 2)).real

    if fraction < 1:
        cropped_shape = (np.array(class_data.shape[1:]) * fraction).round().astype(int)
        class_data = crop_to(
            class_data, (class_data.shape[0], *cropped_shape), dim=(1, 2)
        )
        class_data *= gaussian_window(class_data.shape[1:], device=device)

    class_data = normalize(class_data, dim=(1, 2))
    del class_ft
    return class_data


def get_entries_to_process(prog, db_path, log, overwrite):
    db = pd.read_csv(db_path / "database_summary.csv", sep="\t", index_col=0)
    task = prog.add_task(description="Selecting valid entries...", total=len(db))

    to_download = []
    to_extract = []
    to_project = []
    for entry_id in db.index:
        prog.update(task, advance=1)
        proj_path = db_path / f"{entry_id:04}.pt"
        map_path = db_path / f"emd_{entry_id:04}.map"
        gz_path = db_path / f"emd_{entry_id:04}.map.gz"

        # only projection should ever need overwriting
        if proj_path.exists() and not overwrite:
            log.info(f"{entry_id:04} was already projected")
        elif map_path.exists():
            log.info(f"{entry_id:04} was already extracted")
            to_project.append(map_path)
        elif gz_path.exists():
            log.info(f"{entry_id:04} was already downloaded")
            to_extract.append(gz_path)
            to_project.append(map_path)
        else:
            log.info(f"{entry_id:04} is new")
            to_download.append(entry_id)
            to_extract.append(gz_path)
            to_project.append(map_path)

    log.warning(f"Will download {len(to_download)}.")
    log.warning(f"Will extract {len(to_extract)}.")
    log.warning(f"Will project {len(to_project)}.")
    return to_download, to_extract, to_project


def download_headers(prog, db_path, dry_run):
    if dry_run:
        print(
            f"Will update header list (currently have {len(list(db_path.glob('*.xml')))} headers)."
        )
        return

    task = prog.add_task(description="Downloading headers...", total=None)

    rsync = sh.rsync.bake("-rltpvzhu", "--info=progress2")

    def done(cmd, success, exit_code):
        prog.update(task, advance=1, refresh=True)

    full_list_url = "https://ftp.ebi.ac.uk/pub/databases/emdb/status/latest/emdb_released_holdings.json"

    response = requests.get(full_list_url, timeout=60)
    response.raise_for_status()

    entries = response.json().keys()

    failed_download = []
    for entry_id in entries:
        sync_path = f"rsync.ebi.ac.uk::pub/databases/emdb/structures/{entry_id}/header/{entry_id.lower()}-v30.xml"
        try:
            rsync(sync_path, db_path, _done=done)
        except sh.ErrorReturnCode:
            failed_download.append(entry_id)

    return failed_download


def fetch_maps(*, prog, db_path, to_download, to_extract, dry_run, threads, failed):
    """Download and extract maps, yielding map paths as soon as they're ready.

    Failed entries are appended to failed and skipped.
    """
    if dry_run:
        print(f"Will download {len(to_download)} maps")
        print(f"Will extract {len(to_extract)} maps")
        return iter(())

    dl_task = prog.add_task(description="Downloading maps...", total=len(to_download))
    ex_task = prog.add_task(description="Extracting...", total=len(to_extract))

    rsync = sh.rsync.bake("-rltpvzhu", "--info=progress2")
    download_ids = set(to_download)

    def fetch(gz_path):
        entry_id = int(re.search(r"\d+", gz_path.stem).group())
        if entry_id in download_ids:
            sync_path = f"rsync.ebi.ac.uk::pub/databases/emdb/structures/EMD-{entry_id:04}/map/emd_{entry_id:04}.map.gz"
            try:
                rsync(sync_path, db_path)
            except sh.ErrorReturnCode:
                failed.append(entry_id)
                return None
            prog.update(dl_task, advance=1)
        sh.gzip("-d", str(gz_path))
        prog.update(ex_task, advance=1)
        return gz_path.with_suffix("")

    def ready():
        with ThreadPoolExecutor(max_workers=threads) as pool:
            futures = [pool.submit(fetch, gz) for gz in to_extract]
            for fut in as_completed(futures):
                map_path = fut.result()
                if map_path is not None:
                    yield map_path

    return ready()


def _project_map(map_path, proj_path, save_as_mrc=False):
    with mrcfile.open(map_path) as mrc:
        img = torch.from_numpy(mrc.data.astype(np.float32, copy=True))
        px_size = mrc.voxel_size.x.item()

    # apply gaussian window to avoid edge issues
    img = normalize(img)
    img *= gaussian_window(img.shape)

    # pad if needed
    if not np.all(img.shape[0] == np.array(img.shape)):
        # not square, pad to square before projecting
        img = pad_to(img, [np.max(img.shape)] * 3)

    img = normalize(img)

    ft = crop_or_pad_from_px_sizes(ft_and_shift(img), px_size, BIN_RESOLUTION)
    # # gaussian filter in fourier space as well to help with rotations and whatnot
    # ft *= gaussian_window(ft.shape)
    proj_ft = rotated_projection_fts(ft)
    proj = ift_and_shift(proj_ft, dim=(1, 2)).real
    proj = normalize(proj, dim=(1, 2))

    if save_as_mrc:
        with mrcfile.new(proj_path, np.asarray(proj), overwrite=True) as mrc:
            mrc.voxel_size = BIN_RESOLUTION
    else:
        torch.save(proj, proj_path)
    return proj_path


def _project_map_file(map_path):
    """Project a map into its sibling <entry_id>.pt projection file."""
    entry_id = re.search(r"\d+", map_path.stem).group()
    return _project_map(map_path, map_path.with_name(f"{entry_id}.pt"))


def project_maps(*, prog, maps, log, dry_run, threads, keep_maps, total):
    if dry_run:
        print(f"Will project {total} maps")
        return

    task = prog.add_task(description="Projecting...", total=total)

    torch.set_default_dtype(torch.float32)
    set_start_method("spawn", force=True)

    errors = []
    with ProcessPoolExecutor(
        max_workers=threads, initializer=os.nice, initargs=(10,)
    ) as pool:
        results = {}

        def finish(fut):
            map_path = results.pop(fut)
            if fut.exception():
                err = fut.exception()
                err.add_note(map_path.stem)
                errors.append(err)
                log.warning(f"failed projecting {map_path.stem}: {err}")
            else:
                proj_path = fut.result()
                log.info(f"finished projecting {proj_path.stem}")
                if not keep_maps:
                    # keep the map on failure so a retry need not re-download it
                    map_path.unlink(missing_ok=True)
            prog.update(task, advance=1)

        # drain finished projections while the download/extract stream still feeds us
        for map_path in maps:
            for fut in tuple(results):
                if fut.done():
                    finish(fut)
            results[pool.submit(_project_map_file, map_path)] = map_path
        for fut in as_completed(results):
            finish(fut)


def generate_db_summary(prog, db_path, log, dry_run):
    entries = list(Path(db_path).glob("*.xml"))
    task = prog.add_task(
        description="Generating database summary...", total=len(entries)
    )

    rows = {}
    bad = 0
    for xml in entries:
        prog.update(task, advance=1)
        with open(xml, "rb") as f:
            data = xmltodict.parse(f)
        entry_id = int(re.search(r"emd-(\d+)", xml.stem).group(1))
        shape = [int(i) for i in data["emd"]["map"]["dimensions"].values()]
        px = [float(v["#text"]) for v in data["emd"]["map"]["pixel_spacing"].values()]
        size = int(data["emd"]["map"]["@size_kbytes"])
        structure_det = data["emd"]["structure_determination_list"][
            "structure_determination"
        ]
        # resolution
        methods = (
            "singleparticle",
            "subtomogram_averaging",
            "crystallography",
            "helical",
            "tomography",
        )
        if any(f"{method}_processing" in structure_det for method in methods):
            pass
        else:
            log.info(f"{entry_id:04} is bad (method = {structure_det['method']})")
            continue

        errors = []
        for method in methods:
            try:
                processing = structure_det[f"{method}_processing"]
                if isinstance(processing, list):
                    processing = processing[0]  # ugly but ok
                resolution = float(
                    processing["final_reconstruction"]["resolution"]["#text"]
                )
                break
            except Exception as e:
                errors.append(e)
                continue
        else:
            bad += 1
            log.info(
                f"{entry_id:04} is bad:\n"
                + ", ".join(f"{e.__class__.__name__}({e})" for e in errors)
            )
            resolution = np.nan

        rows[entry_id] = [*shape, *px, size, resolution]

    db = pd.DataFrame.from_dict(
        rows,
        orient="index",
        columns=["x", "y", "z", "px_x", "px_y", "px_z", "size_kb", "resolution"],
    )
    db = db.astype(
        {
            "x": int,
            "y": int,
            "z": int,
            "px_x": float,
            "px_y": float,
            "px_z": float,
            "size_kb": int,
            "resolution": float,
        }
    )

    shape = db[["x", "y", "z"]]
    max_dim = shape.max(axis=1)
    size_mb = db["size_kb"] / 1e3
    # convert shape to square to check for sizes (will have to happen in processing)
    volume_when_cube = max_dim**3
    volume_padded_GB = volume_when_cube * 4 / 1e9
    volume_nm = volume_when_cube * db["px_x"] ** 3 / 1e3  # nm^3
    rescale_ratio = (db["px_x"] / BIN_RESOLUTION) ** 3
    volume_rescaled_GB = volume_padded_GB * rescale_ratio

    too_big_or_small = (
        (volume_nm < 1e3) | (volume_padded_GB > 1) | (volume_rescaled_GB > 1)
    )
    too_heavy = size_mb > 400
    anisotropic = (db.px_x != db.px_y) | (db.px_x != db.px_z)

    # discard bad stuff
    to_discard = too_big_or_small | too_heavy | anisotropic
    db = db[~to_discard]

    log.warning(
        f"Found {bad} entries ({100*bad/len(entries):.2f}% of headers) with no single particle info."
    )
    log.warning(
        f"{to_discard.sum()} entries were discarded because too heavy, big, or small to process."
    )
    db.sort_index(inplace=True)
    if not dry_run:
        db.to_csv(
            db_path / "database_summary.csv",
            sep="\t",
            header=True,
            index=True,
            na_rep="NaN",
        )
    return db
