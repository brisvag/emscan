import gc
import inspect
import re
from functools import partial
from itertools import chain
from math import ceil, floor
from pathlib import Path
from threading import Semaphore

import healpy
import mrcfile
import numpy as np
import pandas as pd
import sh
import torch
import xmltodict
from morphosamplers.sampler import sample_subvolumes
from rich import print
from scipy.spatial.transform import Rotation
from torch.fft import fftn, fftshift, ifftn, ifftshift
from torch.multiprocessing import set_start_method
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import rotate

BIN_RESOLUTION = 4
HEALPIX_ORDER = 2


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


def normalize(img, dim=None):
    """Normalize images to std=1 and mean=0."""
    if not torch.is_complex(img):
        img = img.to(torch.float32)
    img -= img.mean(dim=dim, keepdim=True)
    img /= img.std(dim=dim, keepdim=True)
    img = torch.nan_to_num(img, out=img)
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
            del real, imag
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
        del rot


def coerce_ndim(img, ndim):
    """Coerce image dimensionality if smaller than ndim."""
    if img.ndim > ndim:
        raise ValueError(f"image has too high dimensionality ({img.ndim})")
    while img.ndim < ndim:
        img = img[None]
    return img


def rotated_projection_fts(ft):
    """Generate rotated projections fts of a map."""
    # TODO: sinc function to avoid edge artifacts

    nside = healpy.order2nside(HEALPIX_ORDER)
    npix = healpy.nside2npix(nside)
    # only half the views are needed, cause projection symmetry
    angles = healpy.pix2ang(nside, np.arange(npix // 2))
    angles = np.stack(angles).T
    rot = Rotation.from_euler("xz", angles)
    pos = np.array(ft.shape) // 2  # // needed to center ctf for odd images
    grid_size = (ft.shape[0], ft.shape[0], 1)
    slices = sample_subvolumes(
        np.array(ft), positions=pos, orientations=rot, grid_shape=grid_size
    ).squeeze()

    return torch.from_numpy(slices)


def crop_to(img, target_shape, dim=None):
    if dim is None:
        dim = tuple(range(img.ndim))
    edge_crop = (np.array(img.shape) - np.array(target_shape)) / 2
    crop_left = tuple(
        floor(edge_crop[d]) or None if d in dim else None for d in range(img.ndim)
    )
    crop_right = tuple(
        -ceil(edge_crop[d]) or None if d in dim else None for d in range(img.ndim)
    )
    crop_slice = tuple(
        slice(*crops) for crops in zip(crop_left, crop_right, strict=True)
    )
    return img[crop_slice]


def pad_to(img, target_shape, dim=None):
    if dim is None:
        dim = tuple(range(img.ndim))
    edge_pad = (np.array(target_shape) - np.array(img.shape)) / 2
    pad_left = tuple(floor(edge_pad[d]) if d in dim else 0 for d in range(img.ndim))
    pad_right = tuple(ceil(edge_pad[d]) if d in dim else 0 for d in range(img.ndim))
    padding = tuple(chain.from_iterable(zip(pad_right, pad_left, strict=True)))[::-1]
    padded = torch.nn.functional.pad(img, padding)
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


def correlate_rotations(img_ft, proj_fts, angle_step=5):
    """Fast cross correlation of all elements of projections and the image.

    Performs also rotations. Input fts must be fftshifted+fftd+fftshifted.
    """
    shape1 = tuple(img_ft.shape)
    shape2 = tuple(proj_fts.shape[1:])
    if not np.allclose(shape1, shape2):
        raise RuntimeError(
            f"correlate requires the same shape, got {shape1} and {shape2}"
        )

    img_autocc = torch.abs(ift_and_shift(img_ft * img_ft.conj()))
    proj_autocc = torch.abs(ift_and_shift(proj_fts * proj_fts.conj(), dim=(1, 2)))
    denoms = torch.sqrt(img_autocc.amax()) * torch.sqrt(proj_autocc.amax(dim=(1, 2)))
    del img_autocc, proj_autocc

    best_ccs = torch.zeros(len(proj_fts), device=proj_fts.device)
    for _, img_ft_rot in rotations(img_ft, range(0, 360, angle_step)):
        ccs = (
            torch.abs(
                ift_and_shift(img_ft_rot[None] * proj_fts.conj(), dim=(1, 2))
            ).amax(dim=(1, 2))
            / denoms
        )
        best_ccs = torch.amax(torch.stack([ccs, best_ccs]), dim=0)
        del ccs
    del img_ft, proj_fts
    return best_ccs


def compute_cc(class_data, entry_path, device=None):
    corr_values = {}
    entry_data = torch.load(entry_path, map_location=device)

    # crop/pad in real space to match target shape
    class_data = resize(class_data, entry_data.shape, dim=(1, 2))
    # go back to ft
    class_ft = ft_and_shift(class_data, dim=(1, 2))
    del class_data

    for cls_idx, cls in enumerate(class_ft):
        ccs = correlate_rotations(cls, entry_data)
        corr_values[cls_idx] = ccs.max().item()
        del ccs
    del entry_data
    return corr_values


def load_class_data(cls_path, device=None):
    with mrcfile.mmap(cls_path) as mrc:
        class_data = torch.tensor(mrc.data, device=device)
        class_px_size = mrc.voxel_size.x.item()

    class_data = normalize(coerce_ndim(class_data, 3), dim=(1, 2))

    # crop/pad ft to match pixel size
    class_ft = ft_and_shift(class_data, dim=(1, 2))
    class_ft = crop_or_pad_from_px_sizes(
        class_ft, class_px_size, BIN_RESOLUTION, dim=(1, 2)
    )
    class_data = ift_and_shift(class_ft, dim=(1, 2))
    del class_ft
    return class_data


def rsync_with_progress(prog, task_desc, remote_path, local_path, dry_run):
    if dry_run:
        print(
            f"Will update header list (currently have {len(list(local_path.glob('*.xml')))} headers)."
        )
        return

    task = prog.add_task(description=task_desc, start=False)
    percent = re.compile(r"(\d+)%")

    def _process_output(task, line):
        if match := percent.search(line):
            prog.start_task(task)
            prog.update(task, completed=int(match.group(1)))

    rsync = sh.rsync.bake("-rltpvzhu", "--info=progress2")
    proc = rsync(
        remote_path,
        local_path,
        _out=partial(_process_output, task),
        _err=print,
        _bg=True,
    )

    proc.wait()


def get_valid_entries(prog, db_path, log, dry_run):
    entries = sorted(db_path.glob("emd-*.xml"))
    task = prog.add_task(description="Getting list of maps...", total=len(entries))
    to_download = []
    too_big_or_small = 0
    too_heavy = 0
    to_remove = 0
    for entry_xml in entries:
        prog.update(task, advance=1)
        entry_id = re.search(r"emd-(\d+)", entry_xml.stem).group(1)

        with open(entry_xml, "rb") as f:
            entry_metadata = xmltodict.parse(f)

        px_size = np.array(
            [
                float(v["#text"])
                for v in entry_metadata["emd"]["map"]["pixel_spacing"].values()
            ]
        )
        shape = np.array(
            [int(i) for i in entry_metadata["emd"]["map"]["dimensions"].values()]
        )
        shape_square = np.array([np.max(shape)] * 3)

        size_mb = float(entry_metadata["emd"]["map"]["@size_kbytes"]) / 1e3
        # convert shape to square to check for sizes (will have to happen in processing)
        volume_padded_GB = np.prod(shape_square) * 32 / 1e9
        volume_nm = np.prod(np.array(shape_square) * px_size) / 1e3  # nm^3
        proj_path = db_path / f"{entry_id}.pt"
        map_path = db_path / f"emd_{entry_id}.map"
        gz_path = db_path / f"emd_{entry_id}.map.gz"
        if volume_nm > 3e6 or volume_nm < 1e3 or volume_padded_GB > 2:
            # rule of thumb: too big or small to be worth working with
            # discarding less than 5%, so should be good, but we save a lot of issues and bandwidth
            log.info(f"{entry_id} is too big or small ({volume_nm:.2} nm^3), skipping")
            too_big_or_small += 1
            remove = False
            for pth in (proj_path, map_path, gz_path):
                if pth.exists():
                    remove = True
                    log.info(f"{pth.name} exists and will be removed")
                if not dry_run:
                    pth.unlink(missing_ok=True)
            to_remove += remove
            continue
        if size_mb > 300:
            # 300MB is above the 83 percentile (could reduce maybe, but good start)
            log.info(f"{entry_id}'s file is too heavy ({int(size_mb)} MB), skipping")
            too_heavy += 1
            remove = False
            for pth in (proj_path, map_path, gz_path):
                if pth.exists():
                    remove = True
                    log.info(f"{pth.name} exists and will be removed")
                if not dry_run:
                    pth.unlink(missing_ok=True)
            to_remove += remove
            continue

        img_path = db_path / f"emd_{entry_id}.map"
        gz_path = db_path / f"emd_{entry_id}.map.gz"

        if img_path.exists() or gz_path.exists():
            log.info(f"{entry_id} was already extracted or downloaded")
            continue
        else:
            to_download.append(entry_id)

    log.warn(f"Will download {len(to_download)}. Will remove {to_remove}.")
    log.warn(f"Skipping {too_big_or_small} too big/small and {too_heavy} too heavy.)")
    log.warn(
        f"Totaling {(len(entries) - too_big_or_small - too_heavy) / len(entries) * 100}% of the emdb."
    )
    return to_download


def download_maps(prog, db_path, to_download, dry_run):
    if dry_run:
        print(f"Will download {len(to_download)} maps")
        return
    task = prog.add_task(description="Downloading...", total=len(to_download))

    rsync = sh.rsync.bake("-rltpvzhu", "--info=progress2")

    pool = Semaphore(10)

    def done(cmd, success, exit_code):
        pool.release()
        prog.update(task, advance=1, refresh=True)

    def run_rsync(source_path):
        pool.acquire()
        return rsync(source_path, db_path, _bg=True, _done=done)

    procs = []
    for entry_id in to_download:
        sync_path = f"rsync.ebi.ac.uk::pub/databases/emdb/structures/EMD-{entry_id}/map/emd_{entry_id}.map.gz"
        procs.append(run_rsync(sync_path))

    try:
        for p in procs:
            p.wait()
    except sh.ErrorReturnCode:
        for p in procs:
            p.kill()


def extract_maps(prog, db_path, dry_run):
    gz_paths = sorted(db_path.glob("*.map.gz"))
    if dry_run:
        print(f"Will extract {len(gz_paths)} maps")
        return
    task = prog.add_task(description="Extracting...", total=len(gz_paths))

    for gz_path in gz_paths:
        sh.gzip("-d", str(gz_path))
        prog.update(task, advance=1)


def _project_map(map_path, proj_path):
    with mrcfile.open(map_path) as mrc:
        data = mrc.data.astype(np.float32, copy=True)
        px_size = mrc.voxel_size.x.item()

    img = normalize(torch.from_numpy(data))
    if not np.all(data.shape[0] == np.array(data.shape)):
        # not square, pad to square before projecting
        img = pad_to(img, [np.max(data.shape)] * 3)
    ft = crop_or_pad_from_px_sizes(ft_and_shift(img), px_size, BIN_RESOLUTION)
    proj = rotated_projection_fts(ft)

    torch.save(proj, proj_path)
    return proj_path


def project_maps(prog, db_path, overwrite, log, dry_run):
    maps = list(db_path.glob("*.map"))
    task = prog.add_task(
        description="Checking existing projections...", total=len(maps)
    )

    maps_to_project = []
    projections = []
    exist = 0
    for m in maps:
        prog.update(task, advance=1)
        proj = db_path / (re.search(r"\d+", m.stem).group() + ".pt")
        if not overwrite and proj.exists():
            exist += 1
            continue
        maps_to_project.append(m)
        projections.append(proj)

    if dry_run:
        print(
            f"Will project {len(maps_to_project)} maps"
            + ("" if overwrite else f", skipping {exist} already existing.")
        )
        return

    task = prog.add_task(description="Projecting...", total=len(maps_to_project))

    torch.set_default_dtype(torch.float32)
    set_start_method("spawn", force=True)

    # single-threaded for testing
    errors = []
    for m, p in zip(maps_to_project, projections, strict=True):
        try:
            _project_map(m, p)
            log.info(f"finished projecting {p.stem}")
        except Exception as e:
            e.add_note(m.stem)
            errors.append(e)
            log.warn(f"failed projecting {p.stem}")
        prog.update(task, advance=1)
    if errors:
        raise ExceptionGroup("Some projections failed", errors)

    # with Pool(processes=os.cpu_count() // 2, initializer=os.nice, initargs=(10,)) as pool:
    #     results = [
    #         pool.apply_async(_project_map, (m, p)) for m, p in zip(maps, projections, strict=True)
    #     ]
    #     while len(results):
    #         sleep(0.1)
    #         for res in tuple(results):
    #             if res.ready():
    #                 proj_path = res.get()
    #                 log.info(f"finished projecting {proj_path.stem}")
    #                 prog.update(task, advance=1)
    #                 results.pop(results.index(res))
    #


def _parse_headers(db_path):
    df = pd.DataFrame(columns=["x", "y", "z", "px_x", "px_y", "px_z", "size_kb"])
    df.astype(
        {
            "x": int,
            "y": int,
            "z": int,
            "px_x": float,
            "px_y": float,
            "px_z": float,
            "size_kb": int,
        }
    )
    for xml in Path(db_path).glob("*.xml"):
        with open(xml, "rb") as f:
            data = xmltodict.parse(f)
        entry_id = int(re.search(r"emd-(\d+)", xml.stem).group(1))
        shape = [int(i) for i in data["emd"]["map"]["dimensions"].values()]
        px = [float(v["#text"]) for v in data["emd"]["map"]["pixel_spacing"].values()]
        size = int(data["emd"]["map"]["@size_kbytes"])
        df.loc[entry_id] = [*shape, *px, size]
    return df
