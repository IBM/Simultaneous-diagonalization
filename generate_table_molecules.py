# Copyright 2022 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file is part of the code to reproduce the results in the paper:
# E. van den Berg and Kristan Temme, "Circuit optimization of Hamiltonian
# simulation by simultaneous diagonalization of Pauli clusters," Quantum 4,
# p. 322, 2020. https://doi.org/10.22331/q-2020-09-12-322

import cl
import cl_chemistry as clchem
import statistics
from pathlib import Path

strategies = ['largest_first','independent_set','sequential']
description= ['largest first','independent set','sequential']

entries = clchem.index_molecules()
entries.sort()

max_paulis = 200000

cl.ensureDirExists('tables')
filename = "tables/table_molecules.tex"
print("Writing %s . . ." % filename)
with open(filename,"w") as fp :
   fp.write("\\begin{tabular}{|llrrl%s|}\n" % ("|rrr" * len(strategies)))
   fp.write("\\multicolumn{1}{l}{Mol.} & basis & \\# & paulis & \multicolumn{1}{r}{rep.} & %s\\\\\n" %
            "&".join("\\multicolumn{3}{c}{%s}" % d for d in description))
   fp.write("\\hline\n")

   for entry in entries :
      for (idx,instance) in enumerate(entry[1]) :
         instance = instance[1]
         paulis = instance.Hamiltonian()

         if (len(paulis) > max_paulis) :
            continue

         # Molecule information
         s = ""
         if (idx == 0) :
            s += "%s & %s & %d & %d " % (instance.molecule_latex, instance.basis,
                                         instance.nQubits, len(paulis))
         else :
            s += "&&&"

         # Add encoding
         s += "& %s" % instance.encoding_abbrv

         # Add coloring results
         for strategy in strategies :
            partition = instance.coloring(strategy, compute=False)
            if (partition) :
               sizes = [len(p) for p in partition]
               s += "& %d & %d & %d" % (len(partition),statistics.median(sizes),max(sizes))
            else :
               s += "& -- & -- & --"

         # Add newline
         s += "\\\\"

         fp.write("%s\n" % s)
         if (idx == len(entry[1])-1) :
            fp.write("\\hline\n")

   fp.write("\\end{tabular}\n")
