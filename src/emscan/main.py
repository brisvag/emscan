import click

import emscan


@click.command(
    context_settings={"help_option_names": ["-h", "--help"], "show_default": True},
)
@click.argument(
    "classes", type=click.Path(exists=True, dir_okay=False, resolve_path=True)
)
@click.option(
    "-l", "--update-list", is_flag=True, help="Whether to update the list of entries."
)
@click.option(
    "-p",
    "--update-projections",
    is_flag=True,
    help="Whether to update the projection database.",
)
@click.option(
    "-q",
    "--emdb-query",
    type=str,
    default="",
)
@click.option(
    "--emdb-save-path",
    type=click.Path(dir_okay=True, file_okay=False),
    default="~/.emdb_projections/",
    help="Where to save the database of projections.",
)
@click.option(
    "-o",
    "--output",
    default="./correlation_output.json",
    type=click.Path(dir_okay=False, file_okay=True),
    help="Output json file with the cc data",
)
@click.option("-v", "--verbose", count=True)
@click.option("-f", "--overwrite", is_flag=True, help="overwrite output if exists")
@click.version_option(version=emscan.__version__)
def cli(
    classes,
    update_list,
    update_projections,
    emdb_query,
    emdb_save_path,
    correlation_output,
    verbose,
    overwrite,
):
    """Find emdb entries similar to the given 2D classes."""
    import json
    import logging
    import re
    from pathlib import Path
    from time import sleep

    import mrcfile
    import numpy as np
    import sh
    import torch
    import xmltodict
    from rich.logging import RichHandler
    from rich.progress import Progress
    from torch.multiprocessing import Pool, set_start_method

    from emscan._functions import (
        compute_cc,
        crop_or_pad_from_px_sizes,
        ft_and_shift,
        load_class_data,
        normalize,
        rotated_projection_fts,
    )

    logging.basicConfig(
        level=40 - max(verbose * 10, 0),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler()],
    )
    log = logging.getLogger("emscan")

    output = Path(correlation_output).expanduser().resolve()
    if output.exists() and not overwrite:
        raise click.UsageError("Output already exists. Pass -f to overwrite.")

    bin_resolution = 4
    healpix_order = 2

    emdb_save_path = Path(emdb_save_path).expanduser().resolve()

    rsync = sh.rsync.bake("-rltpvzhu", "--info=progress2")

    if update_list:
        print("Updating header database...")
        rsync(
            "rsync.ebi.ac.uk::pub/databases/emdb/structures/*/header/*v30.xml",
            emdb_save_path,
        )

    entries = sorted(emdb_save_path.glob("*.xml"))

    torch.set_default_tensor_type(torch.FloatTensor)

    with Progress(disable=False) as prog:
        if update_projections:
            for entry_xml in prog.track(
                entries, description="Updating projection database..."
            ):
                entry_id = re.search(r"emd-(\d+)", entry_xml.stem).group(1)

                entry_pt = emdb_save_path / f"{entry_id}.pt"
                if entry_pt.exists() and not overwrite:
                    log.warn(f"{entry_id} projections exist, skipping")
                    continue

                with open(entry_xml, "rb") as f:
                    entry_metadata = xmltodict.parse(f)

                px_size = np.array(
                    [
                        float(v["#text"])
                        for v in entry_metadata["emd"]["map"]["pixel_spacing"].values()
                    ]
                )
                shape = np.array(
                    [
                        int(i)
                        for i in entry_metadata["emd"]["map"]["dimensions"].values()
                    ]
                )
                volume = np.prod(np.array(shape) * px_size) / 1e3  # nm^3
                if volume > 1e6:
                    # rule of thumb: too big to be worth working with
                    log.warn(f"{entry_id} is too big, skipping")
                    continue
                if not np.allclose(shape, shape[0]):
                    # only deal with square for now
                    log.warn(f"{entry_id} is not square, skipping")
                    continue

                img_path = emdb_save_path / f"emd_{entry_id}.map"

                if not img_path.exists():
                    log.info(f"downloading {entry_id}")
                    gz_name = f"emd_{entry_id}.map.gz"
                    sync_path = f"rsync.ebi.ac.uk::pub/databases/emdb/structures/EMD-{entry_id}/map/{gz_name}"
                    rsync(sync_path, emdb_save_path)
                    sh.gzip("-d", str(emdb_save_path / gz_name))

                log.info(f"projecting {entry_id}")
                with mrcfile.open(img_path) as mrc:
                    data = mrc.data.astype(np.float32, copy=True)

                img = normalize(torch.from_numpy(data))
                ft = crop_or_pad_from_px_sizes(
                    ft_and_shift(img), px_size, bin_resolution
                )
                proj = rotated_projection_fts(ft, healpix_order=healpix_order)

                log.info(f"Saving tensor: {entry_id}")
                torch.save(proj, entry_pt)

                img_path.unlink()

        entries = sorted(emdb_save_path.glob("*.pt"))

        corr_values = {}

        set_start_method("spawn", force=True)
        devices = torch.cuda.device_count()

        task = prog.add_task(description="Correlating...")
        with Pool(processes=devices) as pool:
            # pre-load class data for each gpu
            cls_data = {
                device: load_class_data(
                    classes, device=device, bin_resolution=bin_resolution
                )
                for device in range(devices)
            }

            results = [
                pool.apply_async(
                    compute_cc,
                    (cls_data[(device := idx % devices)], entry, f"cuda:{device}"),
                )
                for idx, entry in enumerate(entries)
            ]
            while len(results):
                sleep(0.1)
                for res in tuple(results):
                    if res.ready():
                        entry_path, cc_dict = res.get()
                        for cls_idx, cc in cc_dict.items():
                            corr_values.setdefault(cls_idx, {})[entry_path.stem] = cc
                        prog.update(task, advance=100 / len(entries))
                        results.pop(results.index(res))

        with open(correlation_output, "w+") as f:
            json.dump(corr_values, f)


if __name__ == "__main__":
    cli()
