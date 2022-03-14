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

import os
import cl
import cl_qiskit as clq
import cl_methods as clm
import numpy as np
import math

import matplotlib.pyplot as plt
import matplotlib.colors as colors


# ======================================================================
# Methods for benchmarking
# ======================================================================

def _apply_diag_info(T,R) :
   # Circuit for diagonalization
   RC0 = clq.RecordCircuit(T.n)
   R.sendTo(RC0)
   C0 = RC0.circuit; #C0 = clq.qiskit.compiler.transpile(RC0.circuit, optimization_level=2)

   # Circuit without optimization
   RC1 = clq.RecordCircuit(T.n+1); RC1.singleRz = True
   clm.circuitDiagExponentiateGenerate(RC1, R, T, np.ones(T.m), funOrderZ=None)
   C1 = RC1.circuit; #C1 = clq.qiskit.compiler.transpile(RC1.circuit, optimization_level=2)

   # Circuit with optimization
   RC2 = clq.RecordCircuit(T.n+1); RC2.singleRz = True
   clm.circuitDiagExponentiateGenerate(RC2, R, T, np.ones(T.m), funOrderZ=cl.orderZ)
   C2 = RC2.circuit; #C2 = clq.qiskit.compiler.transpile(RC2.circuit, optimization_level=2)

   # Gather information
   c0 = C0.count_ops()
   c1 = C1.count_ops()
   c2 = C2.count_ops()
   return (c0.get('cx',0),sum([value for (key,value) in c0.items()])-c0.get('cx',0), C0.depth(),
           c1.get('cx',0),sum([value for (key,value) in c1.items()])-c1.get('cx',0), C1.depth(),
           c2.get('cx',0),sum([value for (key,value) in c2.items()])-c2.get('cx',0), C2.depth())


def _apply_direct_info(R0,R1) :
   # Circuits
   RC0 = clq.RecordCircuit(R0.nQubits); RC0.singleRz = True
   R0.sendTo(RC0)
   #C0 = RC0.circuit
   C0 = clq.qiskit.compiler.transpile(RC0.circuit, optimization_level=2)

   RC1 = clq.RecordCircuit(R1.nQubits); RC1.singleRz = True
   R1.sendTo(RC1)
   #C1 = RC1.circuit
   C1 = clq.qiskit.compiler.transpile(RC1.circuit, optimization_level=2)

   # Gather information
   c0 = C0.count_ops()
   c1 = C1.count_ops()
   return (0,0,0,
           c0.get('cx',0),sum([value for (key,value) in c0.items()])-c0.get('cx',0), C0.depth(),
           c1.get('cx',0),sum([value for (key,value) in c1.items()])-c1.get('cx',0), C1.depth())

def _apply_method_cnot(T, blocksizes) :
   R = cl.RecordOperations(T.n)
   blocksize = clm.circuitDiagonalize_CNot(T,R,blocksizes)
   return _apply_diag_info(T,R) + (blocksize,)

def _apply_method_generic(M, method) :
   T = cl.Tableau(M)
   R = cl.RecordOperations(T.n)
   clm.circuitDiagonalize(T,R,method)
   return _apply_diag_info(T,R) + (None,)


# User callable methods - simultaneous diagonalization
def apply_method_cnot(M) :
   T = cl.Tableau(M)
   return _apply_method_cnot(T,[])

def apply_method_cnot_log2(M) :
   T = cl.Tableau(M)
   return _apply_method_cnot(T,[int(math.ceil(math.log2(T.n)))])

def apply_method_cnot_best(M) :
   T = cl.Tableau(M)
   return _apply_method_cnot(T,range(1,T.n+1))

def apply_method_cz(M) :
   return _apply_method_generic(M, cl.zeroX_algorithm1_cz)

def apply_method_greedy0(M) :
   return _apply_method_generic(M, cl.zeroX_algorithm1_greedy0)

def apply_method_greedy1(M) :
   return _apply_method_generic(M, cl.zeroX_algorithm1_greedy1)

def apply_method_greedy2(M) :
   return _apply_method_generic(M, cl.zeroX_algorithm1_greedy2)

def apply_method_gmc(M) :
   return _apply_method_generic(M, cl.zeroX_algorithm2)

def apply_method_csw_cz(M) :
   return _apply_method_generic(M, cl.zeroX_algorithm3a)

def apply_method_csw_cnot(M) :
   return _apply_method_generic(M, cl.zeroX_algorithm3b)


# User callable methods - direct exponentiation
def apply_method_direct(M) :
   T0 = cl.Tableau(M)
   R0 = cl.RecordOperations(T0.n+1)
   clm.circuitDirectExponentiation(R0, cl.matrix_to_pauli(M), np.ones(T0.m), optimized=True)

   index = cl.orderPaulis(M)
   Mbar = M[index,:]
   T1 = cl.Tableau(Mbar)
   R1 = cl.RecordOperations(T1.n+1)
   clm.circuitDirectExponentiation(R1, cl.matrix_to_pauli(Mbar), np.ones(T1.m), optimized=True)

   return _apply_direct_info(R0,R1) + (None,)


# ======================================================================
# Experiment setup
# ======================================================================

methods = [("gmc",           apply_method_gmc,       "gmc"),
           ("cz",            apply_method_cz,        "cz"),
           ("cnot",          apply_method_cnot,      "cnot"),
           ("cnot_log2",     apply_method_cnot_log2, "cnot-log2"),
           ("cnot_best",     apply_method_cnot_best, "cnot-best"),
           ("greedy1",       apply_method_greedy1,   "greedy-1"),
           ("greedy2",       apply_method_greedy2,   "greedy-2"),
           ("original",      apply_method_direct,    "direct")]
#           ("csw_cz",        apply_method_csw_cz,    "csw-cz"),
#           ("csw_cnot",      apply_method_csw_cnot,  "csw-cnot")]


# ======================================================================
# Generate all required results
# ======================================================================
prefix = 'experiment_basic_'
for method in methods :
   name = prefix + method[0]
   function = method[1]
   R = cl.ResultCache(name)
   for n in range(3,26) :
      for index in range(20) :
         key = (n,index)
         if (key not in R) :
            print("Method %s, problem size %d-%d" % (name, n, index))
            M = cl.create_basic_problem(n,index)
            value = function(M)
            R[(n,index)] = value
   R.flush()


# Get all the results
cache = [cl.ResultCache(prefix + method[0]) for method in methods]


# ======================================================================
# Collect information about the best block size for the cnot method
# ======================================================================

for filterSize in [(0,1000), (5,5), (10,10),(20,20), (25,25), (20,100)] :
   cacheCNotBest = cache[4]
   counts = {}; nEqual = 0; n = 0
   for (key,value) in cacheCNotBest.data.items() :
      if ((key[0] < filterSize[0]) or (key[0] > filterSize[1])) :
         continue

      bestsizes = value[-1]
      if (len(bestsizes) == 0) : bestsizes = [0]

      v = bestsizes[-1]
      c = counts.get(v,0)
      counts[v] = c+1
      n += 1

      logn = int(math.ceil(math.log2(key[0])))
      if (v >= logn) :
         nEqual += 1


   print("--------------------------------------------------")
   print("Size filter = %s (greater than or equal to log2(n) : %.2f)" % (str(filterSize), (nEqual*100.) / n))
   print("--------------------------------------------------")
   for (key, value) in counts.items() :
      print("%2d  %.2f%%" % (key, (value * 100.) / n))


# ======================================================================
# Table with results
# ======================================================================

if (True) :
   cl.ensureDirExists('tables')

   filename = "tables/table_experiment_basic.tex"
   print("Writing %s . . ." % filename)
   with open(filename,"w") as fp :
      fp.write("\\begin{tabular}{|l|l|rrr|rrrr|rrrr|}\n")
      fp.write("\\hline\n")
      fp.write("$n$ & Algorithm & \\multicolumn{3}{c|}{Circuit $\\mathcal{U}$} "
               "& \\multicolumn{4}{c|}{Simulation} & \\multicolumn{4}{c|}{Simulation (optimized)}\\\\\n")
      fp.write("&&    &    &    & \multicolumn{2}{c}{\\cnot{}} &     &    & \multicolumn{2}{c}{\\cnot{}} &     &  \\\\\n")
      fp.write("&& \\cnot{} & single & depth & total  & exp.   & single & depth & total  & exp.   & single & depth\\\\\n")
      fp.write("\\hline\n")

      for n in [5,10,15,20,25] :
         # Determine best values
         block = np.zeros((len(methods), 11))
         for (idx,method) in enumerate(methods) :
            values = [0] * 9
            for index in range(20) :
               result = cache[idx][(n,index)]
               values = [v+r for (v,r) in zip(values,result[:9])]
            values = [round(v/20.) for v in values]
            block[idx,:3] = values[:3]

            for [i,offset] in enumerate([3,6]) :
               block[idx,3+i*4] = values[offset]
               block[idx,4+i*4] = values[offset]-2*values[0]
               block[idx,5+i*4] = values[offset+1]
               block[idx,6+i*4] = values[offset+2]

         # Convert empty values to nan
         block[block == 0] = np.nan

         # Find the minimum value per column
         highlight = (block == np.nanmin(block,axis=0,keepdims=True))

         # Convert nan back to 0
         block[np.isnan(block)] = 0

         # -----------------------------------------------
         # Version 1
         # -----------------------------------------------
         #for (idx,method) in enumerate(methods) :
         #   s = ("%d" % n) if (idx == 0) else ""
         #   s += "& %s &" % method[2]

         #   values = [0] * 9
         #   for index in range(20) :
         #      result = cache[idx][(n,index)]
         #      values = [v+r for (v,r) in zip(values,result[:9])]
         #   values = [round(v/20.) for v in values]

         #   s += "&".join((str(x) if (x!=0) else "--") for x in values[:3])
         #   for offset in [3,6] :
         #      s += "& %d & %d & %d & %d" % (values[offset],
         #                                    values[offset]-2*values[0],
         #                                    values[offset+1],
         #                                    values[offset+2])
         #   fp.write("%s\\\\\n" % s)

         # -----------------------------------------------
         # Version 2
         # -----------------------------------------------
         for (idx,method) in enumerate(methods) :
            s = ("%d" % n) if (idx == 0) else ""
            s += "& %s " % method[2]

            for i in range(block.shape[1]) :
               v = int(block[idx,i])
               v = (str(v) if (v!=0) else "--")
               if (highlight[idx,i]) :
                  v = "\\cellcolor{highlight}{\\color{darkblue}%s}" % v
               s += "&" + v
            fp.write("%s\\\\\n" % s)

         fp.write("\\hline\n")
      fp.write("\\end{tabular}\n")


# ======================================================================
# Plots
# ======================================================================

if (True) :
   cl.ensureDirExists('fig')

   settings = [(0,"cnot1",   True,  True),
               (1,"single1", False, True),
               (2,"depth1",  False, True),
               (6,"cnot2",   True,  True),
               (7,"single2", False, True),
               (8,"depth2",  False, True)]

   margin_left = ["-81.5","0"]
   margin_top  = ["-64.0","0"]

   labels = [""]+[method[2] for method in methods[:-1]]
   cmap = colors.LinearSegmentedColormap.from_list("white_to_blue", [(0.1, 0.1, 1), (1, 1, 1)], N=256)

   for (result_index, name, y_labels, x_labels) in settings :
      N = len(methods)-1; idx = 0; num_exp = (26-3)*20
      counts = np.zeros((N, num_exp), dtype=np.int16)
      for n in range(3,26) :
         for index in range(20) :
            for k in range(N) :
               counts[k,idx] = cache[k][(n,index)][result_index]
            idx += 1

      comparison = np.zeros((N,N),dtype=np.float)
      for i in range(N) :
         comparison[i,i] = 1
         for j in range(N) :
            if (i != j) :
               comparison[i,j] = np.sum(counts[i,:] < counts[j,:]) / float(counts.shape[1])

      # Plot the result
      fig, ax = plt.subplots()
      im = ax.imshow(comparison,cmap=cmap)

      for i in range(N):
         text = ax.text(i, i, "-", ha="center", va="center", color="k",size=16)
         for j in range(N):
            if (i != j) :
               s = int(100*comparison[i,j])
               c = "w" if (comparison[i,j] < 0.5) else "k"
               text = ax.text(j, i, s, ha="center", va="center", color=c, size=16)

      ax.xaxis.tick_top()
      ax.set_xticklabels(labels,fontsize=16)
      plt.setp(ax.get_xticklabels(), rotation=40, ha="left", rotation_mode="anchor")
      if (not x_labels) :
         plt.tick_params(axis='x',which='both',bottom=False,top=False)

      ax.set_yticklabels(labels,fontsize=16)
      if (not y_labels) :
         plt.tick_params(axis='y',which='both',left=False,right=False)

      fig.tight_layout()
      plt.savefig("fig/Figure_basic_%s-uncropped.pdf" % name, transparent=True)
      plt.close()
      os.system("pdfcrop --margins '%s %s 0 0' fig/Figure_basic_%s-uncropped.pdf fig/Figure_basic_%s.pdf" %
                (margin_left[y_labels], margin_top[x_labels], name, name))
