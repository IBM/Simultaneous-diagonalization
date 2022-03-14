# Circuit optimization of Hamiltonian simulation by simultaneous diagonalization of Pauli clusters

This code is provided to allow reproducibility of the results in the [paper](https://doi.org/10.22331/q-2020-09-12-322):

E. van den Berg, K. Temme, "Circuit optimization of Hamiltonian simulation by simultaneous diagonalization of Pauli clusters," Quantum 4, 322 (2020). 

The `cache` directory contains all pre-computed results. Files in this directory can be deleted and will be regenerated as needed. The `Hamiltonian` directory contains data files (Pauli lists and coefficients) kindly provided by Antonio Mezzacaco along with derived partitions; see also the relevant [publication](https://doi.org/10.1038/s41467-020-15724-9) or [preprint](https://arxiv.org/abs/1909.12852): Kenny Choo, Antonio Mezzacapo, and Giuseppe Carleo "Fermionic neural-network states for ab-initio electronic structure", Nature Communications volume 11, Article number: 2368 (2020).

The code has a number of external dependences. It uses the [NetworkX](https://networkx.org/) and [Matplotlib](https://matplotlib.org/) python packages and calls `pdfcrop` to crop the generated figures. The `cl.py` file contains routines for Paulis, tableau operations, and diagonalization; `cl_methods.py` generates exponentiation circuits; `cl_qiskit.py` provides code to generate [Qiskit](https://qiskit.org/) circuits; and `cl_chemistry.py` contains routines for dealing with the chemistry data. (Running the `experiment_chemistry.py` code may generate warnings for missing files, these correspond to partitions that took too long to generate.)

| Script | Generates tables and figures |
| ------ | ---------------------------- |
| generate_figure2a.py | Figure 2a |
| generate_figure2b.py | Figure 2b |
| generate_figure2c.py | Figure 2c |
| generate_figure9.py | Figure 9a (fig/Figure_9a.pdf) |
| | Figure 9b (fig/Figure_9b.pdf) |
| | Figure 9c (fig/Figure_9c.pdf) |
| generate_figure_paritioning.py | Figure 11 (fig/Figure_partition_\*.pdf))|
| experiment_basic.py | Figure 10 (left) (fig/Figure_basic_cnot2.pdf) |
| | Figure 10 (center) (fig/Figure_basic_single2.pdf) |
| | Figure 10 (right) (fig/Figure_basic_depth2.pdf) |
| | Table 1 (tables/table_experiment_basic.tex) |
| experiment_nonsquare.py | Table 2 (tables/table_experiment_nonsquare.tex) |
| generate_table_molecules.py | Table 3 (tables/table_molecules.tex) |
| experiment_chemistry.py | Table 4 (tables/table_simulate_sequential.tex) |
| | Table 5 (tables/table_simulate_other_cnot.tex) |
| | Table 6 (tables/table_simulate_other_depth.tex) |

