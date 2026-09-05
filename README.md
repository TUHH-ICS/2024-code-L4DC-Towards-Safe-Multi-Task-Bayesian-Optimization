# Towards Safe Multi-Task Bayesian Optimization

Reference implementation and supplementary material for the L4DC 2024 paper **“Towards Safe Multi-Task Bayesian Optimization.”** The experiments use related, inexpensive tasks to accelerate safe Bayesian optimization of an expensive primary task while accounting for uncertainty in the learned task correlations.

[Published paper (PMLR)](https://proceedings.mlr.press/v242/lubsen24a.html) · [arXiv preprint](https://arxiv.org/abs/2312.07281) · [Supplementary PDF](paper/Safe_Multi-Task_Bayesian_Optimization_Git.pdf)

## Repository layout

| Path | Purpose |
| --- | --- |
| `code/test_run.py` | Small interactive example with online posterior plots |
| `code/run_N2.py` | Two-laser experiments corresponding to Figure 3a |
| `code/run_N5.py` | Five-laser SaMSBO experiment corresponding to Figure 3b |
| `code/plot.py` | Recreates the comparison figure from the bundled result files |
| `code/bo/`, `code/model/`, `code/cov/` | Optimization loop and multi-task GP implementation |
| `code/plant/` | Laser-chain models and control utilities |
| `code/data_paper/` | Initial conditions and result data used in the paper |

## Requirements

The code was developed with Python 3.10.12 on Ubuntu 22.04.3 LTS. A virtual environment is recommended.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`slycot` requires a working BLAS/LAPACK and Fortran toolchain when a compatible binary wheel is unavailable. The plotting script also uses LaTeX text rendering, so a LaTeX installation is required to reproduce the publication typography.

## Running the experiments

Run the scripts from the `code` directory so that their relative data paths resolve correctly:

```bash
cd code

# Short visual demonstration
python test_run.py

# Generate two- and five-laser experiment data
python run_N2.py
python run_N5.py

# Use a truncated confidence parameter in either experiment
python run_N2.py trunc
python run_N5.py trunc
```

The full optimization experiments are computationally expensive. Experiment settings such as the disturbance level, number of evaluations, and selected controller type are documented near the top of each entry-point script.

Generated result files are written to `code/data/`. To recreate the comparison figure from the data shipped with the repository, run:

```bash
python plot.py
```

The figure is saved as `code/figures/comparison.pdf`.

## Citation

```bibtex
@inproceedings{lubsen2024towards,
  title     = {Towards Safe Multi-Task Bayesian Optimization},
  author    = {L\"{u}bsen, Jannis and Hespe, Christian and Eichler, Annika},
  booktitle = {Proceedings of the 6th Annual Learning for Dynamics and Control Conference},
  series    = {Proceedings of Machine Learning Research},
  volume    = {242},
  pages     = {839--851},
  year      = {2024},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v242/lubsen24a.html}
}
```

## License

This project is licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE).
