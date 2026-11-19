# emscan

[![License](https://img.shields.io/pypi/l/emscan.svg?color=green)](https://github.com/brisvag/emscan/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/emscan.svg?color=green)](https://pypi.org/project/emscan)
[![Python Version](https://img.shields.io/pypi/pyversions/emscan.svg?color=green)](https://python.org)
[![CI](https://github.com/brisvag/emscan/actions/workflows/ci.yml/badge.svg)](https://github.com/brisvag/emscan/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/brisvag/emscan/branch/main/graph/badge.svg)](https://codecov.io/gh/brisvag/emscan)

emscan is a command line and GUI tool that can be used to cross-correlate 2D projections of unkown objects against the entirety[^1] of the [EMDB](https://www.ebi.ac.uk/emdb). For example, it can be used to find existing homologous (also partial) complexes of unknown particle picks from a cell-extract shotgun cryo-em experiment.

[^1]: not _quite_: some entries are (clearly) discarded in a few specific cases; some parameters allow tweaking this selection.

## _Disclaimers_

- emscan requires a _large_ amount of space to run. While you won't store the whole EMDB (entries are stored downsampled, pre-projected and with limited rotational resolution), it will still require several TBs of space.
- the first runs are much much slower than the usual run: downloading and generating the full database might require days. However, scanning individual projections against the full database should only take hours.

## Installation

Install from pypi via pip (ideally in a new virtual environment):

```sh
pip install emscan
```

To verify the installation and print out the usage, run:

```sh
emscan -h
```

or for a specific subcommand:

```sh
emscan gen-db -h
```

## Usage

Using emscan requires a 3 step process:
1. Database generation (downloading, extracting, projecting). This requires the longest time and biggest space on dist.
2. Scanning (comparing the input 2D image against the whole database). Faster than generation but still a heavy job.
3. Inspecting results (visually comparing best hits and investigating the most promising ones). Very fast.

Note that all need to know _where_ your database lives. By default it will be put into your home directory under `~/.emdb_projections`, but you might want to provide a different path via `--db-path` to _all commands_.

To do so, the 3 main commands are:

```sh
emscan --db-path <some_path> gen-db -lmp
emscan --db-path <some_path> scan some_map.mrc
emscan --db-path <some_path> show some_map.mrc
```

Read the help messages for more information on parameters.
