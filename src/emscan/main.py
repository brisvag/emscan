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
@click.option("-d", "--dry-run", is_flag=True)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Verbosity level (can be passed multiple times).",
)
@click.pass_context
def cli(ctx, db_path, overwrite, verbose, dry_run):
    """Scan EMDB entries for similar 2D classes by comparing projections."""
    import logging
    from pathlib import Path

    from rich.logging import RichHandler

    ctx.ensure_object(dict)

    ctx.obj["db_path"] = Path(db_path).expanduser().resolve()
    ctx.obj["overwrite"] = overwrite
    ctx.obj["dry_run"] = dry_run

    logging.basicConfig(
        level=40 - max(verbose * 10, 0),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler()],
    )
    log = logging.getLogger("emscan")
    ctx.obj["log"] = log


@cli.command()
@click.option(
    "-l", "--list", "list_", is_flag=True, help="Update the list of entry headers."
)
@click.option(
    "-m",
    "--maps",
    is_flag=True,
    help="Download the emdb maps from the available headers.",
)
@click.option(
    "-p",
    "--projections",
    is_flag=True,
    help="Generate projections for all the available maps.",
)
@click.pass_context
def gen_db(
    ctx,
    list_,
    maps,
    projections,
):
    """Generate the projection database."""
    if not list_ and not maps and not projections:
        raise click.UsageError("Must provide at least -l or -p or -m.")

    from rich.progress import Progress

    from emscan._functions import (
        download_maps,
        extract_maps,
        get_valid_entries,
        project_maps,
        rsync_with_progress,
    )

    db_path = ctx.obj["db_path"]
    overwrite = ctx.obj["overwrite"]
    log = ctx.obj["log"]
    dry_run = ctx.obj["dry_run"]

    EMDB_HEADERS = "rsync.ebi.ac.uk::pub/databases/emdb/structures/*/header/*v30.xml"

    with Progress(disable=False) as prog:
        if list_:
            rsync_with_progress(
                prog,
                task_desc="Updating header database...",
                remote_path=EMDB_HEADERS,
                local_path=db_path,
                dry_run=dry_run,
            )

        if maps:
            to_download = get_valid_entries(prog=prog, db_path=db_path, log=log)
            download_maps(
                prog=prog, db_path=db_path, to_download=to_download, dry_run=dry_run
            )
            extract_maps(prog=prog, db_path=db_path, dry_run=dry_run)

        if projections:
            project_maps(
                prog=prog,
                db_path=db_path,
                overwrite=overwrite,
                log=log,
                dry_run=dry_run,
            )


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

    entries = sorted(db_path.glob("*.pt"))

    set_start_method("spawn", force=True)
    devices = torch.cuda.device_count()

    with Progress() as prog:
        task = prog.add_task(description="Correlating...")
        with Pool(processes=devices) as pool:
            # pre-load class data for each gpu
            cls_data = {
                device: load_class_data(classes, device=device)
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
    import napari
    import numpy as np
    import pandas as pd
    import torch
    from rich import print

    from emscan._functions import ift_and_shift, pad_to

    db_path = ctx.obj["db_path"]
    ctx.obj["overwrite"]
    # log = ctx.obj['log']

    df = pd.read_csv(correlation_results, sep="\t", index_col="entry")

    for group in class_group:
        cols = group.split(",")
        best = df[cols].max(axis=1)
        df = df.drop(columns=cols)
        df[group] = best

    df_top = pd.DataFrame()
    df_top.index.name = "rank"
    for col in df:
        top_10_idx = df[col].sort_values()[::-1].index[:10]
        top_10 = df[col].loc[top_10_idx].reset_index()
        df_top[[f"{col}_entry", f"{col}_cc"]] = top_10

    print(df_top)
    v = napari.Viewer()
    v.grid.enabled = True

    uniq_entries = np.unique(np.ravel(df_top.iloc[:, ::2]))
    imgs = []
    for entry in uniq_entries:
        ft = torch.load(db_path / f"{entry:04}.pt")
        imgs.append(ift_and_shift(ft, dim=(1, 2)))
    max_size = np.max([img.shape for img in imgs], axis=0)
    for img in imgs:
        img = np.array(pad_to(img, max_size)).real
        v.add_image(img, name=entry)

    napari.run()


if __name__ == "__main__":
    cli()
