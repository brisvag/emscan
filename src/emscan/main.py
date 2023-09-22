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
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from pathlib import Path

    import pandas as pd
    import torch
    from rich.progress import Progress
    from torch.multiprocessing import set_start_method

    from emscan._functions import (
        compute_cc,
        load_class_data,
    )

    db_path = ctx.obj["db_path"]
    overwrite = ctx.obj["overwrite"]
    log = ctx.obj["log"]
    output = Path(output).expanduser().resolve()

    entries = sorted(db_path.glob("*.pt"))

    add_header = True
    exist = []
    if output.exists():
        if overwrite:
            log.warn(f"{output.name} exists and will be overwritten.")
        else:
            exist = pd.read_csv(output, sep="\t", index_col=0).index
            log.warn(
                f"{output.name} already esists: ({len(exist)}/{len(entries)}) entries already processed."
            )

            entries = [entry for entry in entries if int(entry.stem) not in exist]
            add_header = False

    set_start_method("spawn", force=True)
    devices = torch.cuda.device_count()

    errors = []
    with Progress() as prog:
        task = prog.add_task(description="Correlating...", total=len(entries))
        # with Pool(processes=devices) as pool:
        with ProcessPoolExecutor(max_workers=devices) as pool:
            # pre-load class data for each gpu
            cls_data = {
                device: load_class_data(classes, device=device)
                for device in range(devices)
            }

            futures = {
                pool.submit(
                    compute_cc,
                    cls_data[(device := idx % devices)],
                    entry,
                    f"cuda:{device}",
                ): entry
                for idx, entry in enumerate(entries)
            }
            for fut in as_completed(futures):
                entry_id = futures[fut].stem
                if fut.exception():
                    errors.append(fut.exception())
                    log.warn(f"failed correlating {entry_id}")
                else:
                    cc_dict = fut.result()
                    log.info(f"finished correlating to {entry_id}")
                    df = pd.DataFrame(cc_dict, index=pd.Index([entry_id], name="entry"))
                    df.to_csv(output, sep="\t", header=add_header, index=True, mode="a")
                    add_header = False  # so we only add once
                prog.update(task, advance=1)

    if errors:
        raise ExceptionGroup("Some correlations failed", errors)


@cli.command()
@click.argument(
    "correlation_results",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
)
@click.option("-g", "--class-group", multiple=True, type=str)
@click.option("-n", "--top-n", default=30, type=int, help="How many top hits to show.")
@click.pass_context
def show(ctx, correlation_results, class_group, top_n):
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
        top_n_idx = df[col].sort_values()[::-1].index[:top_n]
        top_n_entries = df[col].loc[top_n_idx].reset_index()
        df_top[[f"{col}_entry", f"{col}_cc"]] = top_n_entries

    print(df_top)
    v = napari.Viewer()
    v.grid.enabled = True
    v.grid.stride = -1

    uniq_entries = np.unique(np.ravel(df_top.iloc[:, ::2]))
    imgs = {}
    for entry in uniq_entries:
        ft = torch.load(db_path / f"{entry:04}.pt")
        imgs[entry] = ift_and_shift(ft, dim=(1, 2))
    max_size = np.max([img.shape for img in imgs], axis=0)
    for entry, img in imgs.items():
        img = np.array(pad_to(img, max_size)).real
        v.add_image(img, name=entry)

    napari.run()


if __name__ == "__main__":
    cli()
