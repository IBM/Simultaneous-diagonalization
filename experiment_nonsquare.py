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

   # Circuit with optimization and randomization
   RC3 = clq.RecordCircuit(T.n+1); RC3.singleRz = True
   funOrderZ = (lambda x : cl.orderZ_randomized(x,100))
   clm.circuitDiagExponentiateGenerate(RC3, R, T, np.ones(T.m), funOrderZ=funOrderZ)
   C3 = RC3.circuit; #C3 = clq.qiskit.compiler.transpile(RC3.circuit, optimization_level=2)

   # Gather information
   c0 = C0.count_ops()
   c1 = C1.count_ops()
   c2 = C2.count_ops()
   c3 = C3.count_ops()
   return (c0.get('cx',0),sum([value for (key,value) in c0.items()])-c0.get('cx',0), C0.depth(),
           c1.get('cx',0),sum([value for (key,value) in c1.items()])-c1.get('cx',0), C1.depth(),
           c2.get('cx',0),sum([value for (key,value) in c2.items()])-c2.get('cx',0), C2.depth(),
           c3.get('cx',0),sum([value for (key,value) in c3.items()])-c3.get('cx',0), C3.depth())


def _apply_direct_info(R1,R2,R3) :
   # Circuits
   RC1 = clq.RecordCircuit(R1.nQubits); RC1.singleRz = True
   R1.sendTo(RC1)
   #C1 = RC1.circuit
   C1 = clq.qiskit.compiler.transpile(RC1.circuit, optimization_level=2)

   RC2 = clq.RecordCircuit(R2.nQubits); RC2.singleRz = True
   R2.sendTo(RC2)
   #C2 = RC2.circuit
   C2 = clq.qiskit.compiler.transpile(RC2.circuit, optimization_level=2)

   RC3 = clq.RecordCircuit(R3.nQubits); RC3.singleRz = True
   R3.sendTo(RC3)
   #C3 = RC3.circuit
   C3 = clq.qiskit.compiler.transpile(RC3.circuit, optimization_level=2)

   # Gather information
   c1 = C1.count_ops()
   c2 = C2.count_ops()
   c3 = C3.count_ops()
   return (0,0,0,
           c1.get('cx',0),sum([value for (key,value) in c1.items()])-c1.get('cx',0), C1.depth(),
           c2.get('cx',0),sum([value for (key,value) in c2.items()])-c2.get('cx',0), C2.depth(),
           c3.get('cx',0),sum([value for (key,value) in c3.items()])-c3.get('cx',0), C3.depth())


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
   T1 = cl.Tableau(M)
   R1 = cl.RecordOperations(T1.n+1)
   clm.circuitDirectExponentiation(R1, cl.matrix_to_pauli(M), np.ones(T1.m), optimized=True)

   index = cl.orderPaulis(M)
   Mbar = M[index,:]
   T2 = cl.Tableau(Mbar)
   R2 = cl.RecordOperations(T2.n+1)
   clm.circuitDirectExponentiation(R2, cl.matrix_to_pauli(Mbar), np.ones(T2.m), optimized=True)

   index = cl.orderPaulis_randomized(M,100)
   Mbar = M[index,:]
   T3 = cl.Tableau(Mbar)
   R3 = cl.RecordOperations(T3.n+1)
   clm.circuitDirectExponentiation(R3, cl.matrix_to_pauli(Mbar), np.ones(T3.m), optimized=True)

   return _apply_direct_info(R1,R2,R3) + (None,)



# ======================================================================
# Experiment setup
# ======================================================================

methods = [("cz",            apply_method_cz,            "cz"),
           ("csw_cz",        apply_method_csw_cz,        "csw-cz"),
           ("csw_cnot",      apply_method_csw_cnot,       "csw-cnot"),
           ("cnot",          apply_method_cnot,          "cnot"),
           ("cnot_log2",     apply_method_cnot_log2,     "cnot-log2"),
           ("cnot_best",     apply_method_cnot_best,     "cnot-best"),
           ("greedy1",       apply_method_greedy1,       "greedy-1"),
           ("greedy2",       apply_method_greedy2,       "greedy-2"),
           ("original",      apply_method_direct,        "direct")]



# ======================================================================
# Generate all required results
# ======================================================================
prefix = 'experiment_nonsquare_'
for method in methods :
   name = prefix + method[0]
   function = method[1]
   R = cl.ResultCache(name)
   n = 20
   for m in [2,3,5,10,40,50,100,200] :
      for index in range(20) :
         key = (n,m,index)
         if (key not in R) :
            print("Method %s, problem size %d-%d-%d" % (name, n, m, index))
            M = cl.create_basic_problem(n,index)
            C = cl.generate_full_rank_weights(m,n,seed=100*n+index+m)
            M = np.dot(C,M) % 2
            value = function(M)
            R[key] = value
   R.flush()

cache = [cl.ResultCache(prefix + method[0]) for method in methods]



# ======================================================================
# Table with results
# ======================================================================

if (True) :
   cl.ensureDirExists('tables')

   filename = "tables/table_experiment_nonsquare.tex"
   print("Writing %s . . ." % filename)
   with open(filename,"w") as fp :
      fp.write("\\begin{tabular}{|l|l|rrr|rrr|rrr|}\n")
      fp.write("\\hline\n")
      fp.write("$m$ & Algorithm & \\multicolumn{3}{c|}{\\cnot\\ count} "
               "& \\multicolumn{3}{c|}{Single qubit} "
               "& \\multicolumn{3}{c|}{Depth}\\\\\n")
      fp.write("&& base & opt & rnd"
               " & base & opt & rnd"
               " & base & opt & rnd\\\\\n")
      fp.write("\\hline\n")

      # --------------------------------------------------
      # Version 1
      # --------------------------------------------------
      #n = 20
      #for m in [3,10,50,200] :
      #   for idx in range(len(methods)) :
      #      method = methods[idx]
      #      s = ("%d" % m) if (idx == 0) else ""
      #      s += "& %s " % method[2]
      #
      #      values = [0] * 12
      #      for index in range(20) :
      #         result = cache[idx][(n,m,index)]
      #         values = [v+r for (v,r) in zip(values,result[:12])]
      #      values = [round(v/20.) for v in values]
      #
      #      for entry in [0,1,2] :
      #         s += "& %d & %d & %d" % (values[entry+3],values[entry+6],values[entry+9])
      #
      #      fp.write("%s\\\\\n" % s)
      #   fp.write("\\hline\n")

      # --------------------------------------------------
      # Version 2
      # --------------------------------------------------
      n = 20
      for m in [3,10,50,200] :

         block = np.zeros((len(methods),12),dtype=int)
         for idx in range(len(methods)) :
            values = [0] * 12
            for index in range(20) :
               result = cache[idx][(n,m,index)]
               values = [v+r for (v,r) in zip(values,result[:12])]
            block[idx,:] = [round(v/20.) for v in values]

         # Find the minimum value per column
         highlight = (block == np.nanmin(block,axis=0,keepdims=True))

         # Output the entries
         for idx in range(len(methods)) :
            method = methods[idx]
            s = ("%d" % m) if (idx == 0) else ""
            s += "& %s " % method[2]

            for entry in [0,1,2] :
               for offset in [3,6,9] :
                  v = str(block[idx,entry+offset])
                  if (highlight[idx,entry+offset]) :
                     v = "\\cellcolor{highlight}{\\color{darkblue}%s}" % v
                  s += "& "+v
            fp.write("%s\\\\\n" % s)
         fp.write("\\hline\n")


      fp.write("\\end{tabular}\n")
