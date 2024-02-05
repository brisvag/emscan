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
@click.option(
    "-t",
    "--threads",
    default=0,
    help="How many threads to use for projection. If <= 0, guess a good number.",
)
@click.pass_context
def gen_db(
    ctx,
    list_,
    maps,
    projections,
    threads,
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
            to_download = get_valid_entries(
                prog=prog, db_path=db_path, log=log, dry_run=dry_run
            )
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
                threads=threads,
            )


@cli.command()
@click.argument(
    "classes", type=click.Path(exists=True, dir_okay=False, resolve_path=True)
)
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, file_okay=True),
    help="Output json file with the cc data. [default: <CLASSES_NAME>.csv]",
)
@click.option(
    "--fraction",
    type=float,
    default=1,
)
@click.pass_context
def scan(
    ctx,
    classes,
    output,
    fraction,
):
    """Find emdb entries similar to the given 2D classes."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from pathlib import Path

    import pandas as pd
    import torch
    from rich import print
    from rich.progress import Progress
    from torch.multiprocessing import set_start_method

    from emscan._functions import (
        compute_ncc,
        load_class_data,
    )

    db_path = ctx.obj["db_path"]
    overwrite = ctx.obj["overwrite"]
    log = ctx.obj["log"]

    if output is None:
        output = Path(classes).with_suffix(".csv")
    else:
        output = Path(output).expanduser().resolve()

    print("Loading database list...")
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
                device: load_class_data(
                    classes, device=f"cuda:{device}", fraction=fraction
                )
                for device in range(devices)
            }

            futures = {
                pool.submit(
                    compute_ncc,
                    cls_data[idx % devices],
                    entry,
                ): entry
                for idx, entry in enumerate(entries)
            }
            for fut in as_completed(futures):
                entry_id = futures[fut].stem
                if fut.exception():
                    err = fut.exception()
                    err.add_note(entry_id)
                    errors.append(err)
                    log.warn(f"failed correlating {entry_id}: {err}")
                else:
                    cc_dict = fut.result()
                    log.info(f"finished correlating to {entry_id}")
                    for k, v in cc_dict.items():
                        if k != "values":
                            out = output.with_stem(f"{output.stem}_{k}")
                        else:
                            out = output
                        if out.exists() and overwrite:
                            out.unlink()
                        df = pd.DataFrame(v, index=pd.Index([entry_id], name="entry"))
                        df.to_csv(
                            out, sep="\t", header=add_header, index=True, mode="a"
                        )
                    add_header = False  # so we only add once
                    overwrite = False
                prog.update(task, advance=1)

    if errors:
        raise ExceptionGroup("Some correlations failed", errors)


@cli.command()
@click.argument(
    "correlation_results",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
)
@click.option(
    "-g",
    "--class-group",
    multiple=True,
    type=str,
    help="comma separated list of indices to use and group. All if empty.",
)
@click.option("-n", "--top-n", default=30, type=int, help="How many top hits to show.")
@click.option(
    "-c",
    "--class-image",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    help="Image that was used for the correlation.",
)
@click.option(
    "--fraction",
    type=float,
    default=1,
)
@click.pass_context
def show(ctx, correlation_results, class_group, top_n, class_image, fraction):
    """Parse correlation results and show related emdb entries and plots."""
    import webbrowser
    from inspect import cleandoc

    import mrcfile
    import napari
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import torch
    from rich import print

    from emscan._functions import load_class_data, pad_to

    db_path = ctx.obj["db_path"]
    ctx.obj["overwrite"]

    df = pd.read_csv(correlation_results, sep="\t", index_col="entry")

    df_selected = pd.DataFrame()
    selected = []
    if not class_group:
        df_selected["all"] = df.mean(axis=1)
    else:
        for group in class_group:
            cols = group.split(",")
            selected.extend([int(c) for c in cols])
            mean = df[cols].mean(axis=1)
            df_selected[group] = mean

    df_selected.dropna(how="any", inplace=True)

    fig = px.histogram(df_selected, x=df_selected.columns, nbins=50)
    fig.show()

    df_top = pd.DataFrame()
    df_top.index.name = "rank"
    for col in df_selected:
        top_n_idx = df_selected[col].sort_values()[::-1].index[:top_n]
        top_n_entries = df_selected[col].loc[top_n_idx].reset_index()
        df_top[[f"{col}_entry", f"{col}_cc"]] = top_n_entries

    print(df_top)
    v = napari.Viewer()

    if df_selected.shape[1] == 1:
        # preserve order
        uniq_entries = pd.unique(df_top.iloc[:, 0])
    else:
        uniq_entries = np.unique(np.ravel(df_top.iloc[:, ::2]))

    imgs = {}
    if class_image is not None:
        classes_data = load_class_data(class_image, device="cpu", fraction=fraction)
        for i, d in enumerate(classes_data):
            if not selected or i in selected:
                imgs[f"class {i}"] = d[None]
    for entry in uniq_entries:
        imgs[entry] = torch.load(db_path / f"{entry:04}.pt")
    max_size = np.max([img.shape for img in imgs.values()], axis=0)
    for entry, img in imgs.items():
        img = np.array(pad_to(img, max_size, dim=(1, 2))).real.squeeze()
        v.add_image(img, name=f"{entry:04}", interpolation2d="spline36")

    def get_correct_entry(viewer, event):
        for lay in reversed(viewer.layers):
            shift = lay._translate_grid[1:]
            pos = np.array(event.position)[1:]
            if np.all(pos >= shift):
                return lay.name
        return None

    def open_entry(viewer, event):
        if event.modifiers:
            return
        entry = get_correct_entry(viewer, event)
        if entry is not None:
            webbrowser.open(f"https://www.ebi.ac.uk/emdb/EMD-{entry}")

    def make_proj_mrc(viewer, event):
        if "Control" not in event.modifiers:
            return
        entry = get_correct_entry(viewer, event)
        if entry is not None:
            filename = f"emdb_{entry}_projections.mrc"
            with mrcfile.new(filename, data=np.array(imgs[int(entry)]).real) as mrc:
                mrc.voxel_size = 4
                mrc.set_image_stack()
            napari.utils.notifications.show_info(f"Saved projections as {filename}")

    def open_stack(viewer, event):
        if "Shift" not in event.modifiers:
            return
        entry = get_correct_entry(viewer, event)
        v_ = napari.Viewer()
        for sl in imgs[int(entry)]:
            v_.add_image(np.array(sl.real), interpolation2d="spline36")
        v_.grid.enabled = True

    print(
        cleandoc(
            """
            TIPS:
            - double click an image to open its emdb page
            - ctrl+click to create projection stack
            - shift+click to open stack exploded in new window
            """
        )
    )

    v.mouse_double_click_callbacks.append(open_entry)
    v.mouse_drag_callbacks.append(make_proj_mrc)
    v.mouse_drag_callbacks.append(open_stack)

    v.grid.enabled = True
    v.grid.stride = -1
    napari.run()


@cli.command()
@click.argument(
    "entries",
    nargs=-1,
    type=str,
)
@click.option(
    "-c",
    "--class-image",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    help="Image to load to compare to.",
)
@click.pass_context
def view_entries(ctx, entries, class_image):
    """Display projections for the requested entries."""
    import napari
    import numpy as np
    import torch

    from ._functions import load_class_data, pad_to

    db_path = ctx.obj["db_path"]

    v = napari.Viewer()

    imgs = {}
    if class_image is not None:
        classes_data = load_class_data(class_image, device="cpu")
        for i, d in enumerate(classes_data):
            imgs[f"class {i}"] = d[None]
    for entry in entries:
        imgs[entry] = torch.load(db_path / f"{entry:04}.pt")
    max_size = np.max([img.shape for img in imgs.values()], axis=0)
    for entry, img in imgs.items():
        img = np.array(pad_to(img, max_size, dim=(1, 2))).real.squeeze()
        v.add_image(img, name=f"{entry:04}", interpolation2d="spline36")

    v.grid.enabled = True
    v.grid.stride = -1
    napari.run()


@cli.command()
@click.argument(
    "map_file",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
)
@click.pass_context
def project(ctx, map_file):
    """Generate rotated templates from the given map."""
    from pathlib import Path

    from ._functions import _project_map

    overwrite = ctx.obj["overwrite"]

    map_file = Path(map_file)
    proj = map_file.with_stem(map_file.stem + "_proj")
    if proj.exists() and not overwrite:
        raise FileExistsError(proj)

    _project_map(map_file, proj, save_as_mrc=True)


if __name__ == "__main__":
    cli()
