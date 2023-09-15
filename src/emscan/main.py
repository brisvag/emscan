import click

import emscan


@click.group(
    name="emscan",
    context_settings={"help_option_names": ["-h", "--help"], "show_default": True},
)
@click.version_option(version=emscan.__version__)
@click.option(
    "--db-path",
    type=click.Path(dir_okay=True, file_okay=False),
    default="~/.emdb_projections/",
    help="Where to save/find the database of projections.",
)
@click.option(
    "-f", "--overwrite", is_flag=True, help="Overwrite outputs if they exist."
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Verbosity level (can be passed multiple times).",
)
@click.pass_context
def cli(ctx, db_path, overwrite, verbose):
    """Scan EMDB entries for similar 2D classes by comparing projections."""
    import logging
    from pathlib import Path

    from rich.logging import RichHandler

    ctx.ensure_object(dict)

    ctx.obj["db_path"] = Path(db_path).expanduser().resolve()
    ctx.obj["overwrite"] = overwrite

    logging.basicConfig(
        level=40 - max(verbose * 10, 0),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler()],
    )
    log = logging.getLogger("emscan")
    ctx.obj["log"] = log


@cli.command()
@click.option("-l", "--update-list", is_flag=True, help="Update the list of entries.")
@click.option(
    "-p",
    "--update-projections",
    is_flag=True,
    help="Update the projection database (takes a long time!).",
)
@click.pass_context
def gen_db(
    ctx,
    update_list,
    update_projections,
):
    """Generate the projection database."""
    if not update_list and not update_projections:
        raise click.UsageError("Must provide at least -l or -p.")

    import re

    import mrcfile
    import numpy as np
    import sh
    import torch
    import xmltodict
    from rich.progress import Progress

    from emscan._functions import (
        crop_or_pad_from_px_sizes,
        ft_and_shift,
        normalize,
        rotated_projection_fts,
        rsync_with_progress,
    )

    db_path = ctx.obj["db_path"]
    overwrite = ctx.obj["overwrite"]
    log = ctx.obj["log"]

    BIN_RESOLUTION = 4
    HEALPIX_ORDER = 2
    EMDB_HEADERS = "rsync.ebi.ac.uk::pub/databases/emdb/structures/*/header/*v30.xml"

    torch.set_default_tensor_type(torch.FloatTensor)

    rsync = sh.rsync.bake("-rltpvzhu", "--info=progress2")

    with Progress() as prog:
        if update_list:
            task = prog.add_task(description="Updating header database...", start=False)
            rsync_with_progress(prog, task, EMDB_HEADERS, db_path)

        if update_projections:
            task = prog.add_task(description="Updating projection database...")
            entries = sorted(db_path.glob("*.xml"))
            for entry_xml in entries:
                entry_id = re.search(r"emd-(\d+)", entry_xml.stem).group(1)

                entry_pt = db_path / f"{entry_id}.pt"
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

                img_path = db_path / f"emd_{entry_id}.map"

                if not img_path.exists():
                    log.info(f"downloading {entry_id}")
                    gz_name = f"emd_{entry_id}.map.gz"
                    sync_path = f"rsync.ebi.ac.uk::pub/databases/emdb/structures/EMD-{entry_id}/map/{gz_name}"
                    rsync(sync_path, db_path)
                    sh.gzip("-d", str(db_path / gz_name))

                log.info(f"projecting {entry_id}")
                with mrcfile.open(img_path) as mrc:
                    data = mrc.data.astype(np.float32, copy=True)

                img = normalize(torch.from_numpy(data))
                ft = crop_or_pad_from_px_sizes(
                    ft_and_shift(img), px_size, BIN_RESOLUTION
                )
                proj = rotated_projection_fts(ft, healpix_order=HEALPIX_ORDER)

                log.info(f"Saving tensor: {entry_id}")
                torch.save(proj, entry_pt)

                img_path.unlink()

                prog.update(task, advance=100 / len(entries))


@cli.command()
@click.argument(
    "classes", type=click.Path(exists=True, dir_okay=False, resolve_path=True)
)
@click.option(
    "-o",
    "--output",
    default="./output.csv",
    type=click.Path(dir_okay=False, file_okay=True),
    help="Output json file with the cc data",
)
@click.pass_context
def scan(
    ctx,
    classes,
    output,
):
    """Find emdb entries similar to the given 2D classes."""
    from pathlib import Path
    from time import sleep

    import pandas as pd
    import torch
    from rich.progress import Progress
    from torch.multiprocessing import Pool, set_start_method

    from emscan._functions import (
        compute_cc,
        load_class_data,
    )

    db_path = ctx.obj["db_path"]
    overwrite = ctx.obj["overwrite"]
    # log = ctx.obj['log']
    output = Path(output).expanduser().resolve()

    if output.exists() and not overwrite:
        raise click.UsageError("Output already exists. Pass -f to overwrite.")

    BIN_RESOLUTION = 4

    entries = sorted(db_path.glob("*.pt"))

    set_start_method("spawn", force=True)
    devices = torch.cuda.device_count()

    with Progress() as prog:
        task = prog.add_task(description="Correlating...")
        with Pool(processes=devices) as pool:
            # pre-load class data for each gpu
            cls_data = {
                device: load_class_data(
                    classes, device=device, bin_resolution=BIN_RESOLUTION
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
            corr_values = {}
            while len(results):
                sleep(0.1)
                for res in tuple(results):
                    if res.ready():
                        entry_path, cc_dict = res.get()
                        for cls_idx, cc in cc_dict.items():
                            corr_values.setdefault(cls_idx, {})[entry_path.stem] = cc
                        prog.update(task, advance=100 / len(entries))
                        results.pop(results.index(res))

    df = pd.DataFrame(corr_values)
    df.index.name = "entry"
    df.to_csv(output, sep="\t")


@cli.command()
@click.argument(
    "correlation_results",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
)
@click.option("-g", "--class-group", multiple=True, type=str)
@click.pass_context
def show(ctx, correlation_results, class_group):
    """Parse correlation results and show related emdb entries and plots."""
    import pandas as pd

    ctx.obj["db_path"]
    ctx.obj["overwrite"]
    # log = ctx.obj['log']

    df = pd.read_csv(correlation_results, sep="\t", index_col="entry")

    for group in class_group:
        cols = group.split(",")
        mean = df[cols].mean(axis=1)
        df = df.drop(columns=cols)
        df[group] = mean

    df_top = pd.DataFrame()
    df_top.index.name = "rank"
    for col in df:
        top_10_idx = df[col].sort_values()[::-1].index[:10]
        top_10 = df[col].loc[top_10_idx].reset_index()
        df_top[[f"{col}_entry", f"{col}_cc"]] = top_10
    print(df_top)


if __name__ == "__main__":
    cli()
