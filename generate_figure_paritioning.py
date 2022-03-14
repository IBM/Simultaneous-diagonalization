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
import cl_qiskit as clq
import cl_methods as clm
import cl_chemistry as clc

import matplotlib.pyplot as plt
import numpy as np
import os


def _apply_diag_info(T,R, weights) :
   # Circuit for diagonalization
   RC0 = clq.RecordCircuit(T.n)
   R.sendTo(RC0)
   C0 = RC0.circuit; #C0 = clq.qiskit.compiler.transpile(RC0.circuit, optimization_level=2)

   # Circuit with optimization
   RC = clq.RecordCircuit(T.n+1); RC.singleRz = True
   clm.circuitDiagExponentiateGenerate(RC, R, T, weights, funOrderZ=cl.orderZ)
   C  = RC.circuit

   # Gather information
   c = C.count_ops()
   c0 = C0.count_ops()
   return (c.get('cx',0),sum([value for (key,value) in c.items()])-c.get('cx',0), C.depth(),
           c0.get('cx',0),sum([value for (key,value) in c0.items()])-c0.get('cx',0), C0.depth())

def _apply_method_cnot(T, weights, blocksizes) :
   R = cl.RecordOperations(T.n)
   blocksize = clm.circuitDiagonalize_CNot(T,R,blocksizes)
   return _apply_diag_info(T,R, weights)

def _apply_method_generic(M, weights, method) :
   T = cl.Tableau(M)
   R = cl.RecordOperations(T.n)
   clm.circuitDiagonalize(T,R,method)
   return _apply_diag_info(T,R,weights)


# User callable methods - simultaneous diagonalization
def apply_method_cnot(M, weights) :
   T = cl.Tableau(M)
   return _apply_method_cnot(T, weights, [])

def apply_method_cnot_log2(M, weights) :
   T = cl.Tableau(M)
   return _apply_method_cnot(T, weights, [int(math.ceil(math.log2(T.n)))])

def apply_method_cnot_best(M, weights) :
   T = cl.Tableau(M)
   return _apply_method_cnot(T, weights, range(1,T.n+1))

def apply_method_cz(M, weights) :
   return _apply_method_generic(M, weights, cl.zeroX_algorithm1_cz)

def apply_method_greedy0(M, weights) :
   return _apply_method_generic(M, weights, cl.zeroX_algorithm1_greedy0)

def apply_method_greedy1(M, weights) :
   return _apply_method_generic(M, weights, cl.zeroX_algorithm1_greedy1)

def apply_method_greedy2(M, weights) :
   return _apply_method_generic(M, weights, cl.zeroX_algorithm1_greedy2)


# ======================================================================
# Helper routines
# ======================================================================

def getFilename(instance) :
   return "partitioning_%s_%s_%s" % (instance.molecule,
                                     instance.basis_star,
                                     instance.encoding_abbrv.lower())


def getInstance(instances, encoding) :
   # Get the instance corresponding to the given encoding
   instance = [inst[1] for inst in instances if (inst[1].encoding_abbrv.lower() == encoding)]
   if (len(instance) > 1) :
      raise Exception("Encoding '%s' appears more than once" % str(encoding))
   return instance[0] if instance else None


def getResults(instance) :
   cache       = None
   hamiltonian = None
   colorings   = {}
   for strategy in ['sequential','independent_set','largest_first'] :
      # Get the coloring
      coloring = instance.coloring(strategy=strategy, cache=True, compute=False)
      colorings[strategy] = coloring
      if (coloring is None) :
         continue

      # Determine the cache
      if (cache is None) :
         cache = cl.ResultCache(getFilename(instance))

      # Apply all methods
      for (method_infix, method_fun, method_name) in methods :
         for (coloring_index, indices) in enumerate(coloring) :
            key = (strategy, method_infix, coloring_index)
            if (key in cache) :
               continue

            # Get the Hamiltonian
            if (hamiltonian is None) :
               hamiltonian = instance.Hamiltonian()

            # Get the local terms
            paulis  = [hamiltonian[idx][0] for idx in indices]
            weights = [hamiltonian[idx][1] for idx in indices]
            weights = np.asarray(weights)
            M       = cl.pauli_to_matrix(paulis)
            result  = method_fun(M, weights)

            # Add result to cache
            cache[key] = result

   # Flush the cache if needed
   if (cache is not None) :
      cache.flush()

   return (cache, colorings)


def getPlotCoordinates(instances, strategies, methods) :
   if (not isinstance(instances,(list,tuple))) :
     instances = (instances,)
   if (not isinstance(methods,(list,tuple))) :
     methods = (methods,)

   x = [[] for s in strategies]
   y = [[] for s in strategies]
   for instance in instances :
      (cache, colorings) = getResults(instance)
      if (cache is not None) :
         for (strategy_idx,strategy) in enumerate(strategies) :
            coloring = colorings[strategy]
            if (coloring is None) :
               continue

            for (idx,indices) in enumerate(coloring) :
               for method in methods :
                  result = cache[(strategy, method, idx)]
                  x[strategy_idx].append(len(indices))
                  y[strategy_idx].append(result[3])

   return (x,y)



# ======================================================================
# Generate all required results
# ======================================================================

# Experiment setup
methods = [("cz",            apply_method_cz,            "cz"),
           ("cnot",          apply_method_cnot,          "cnot"),
           ("greedy1",       apply_method_greedy1,       "greedy-1"),
           ("greedy2",       apply_method_greedy2,       "greedy-2")]

strategies = ['independent_set','largest_first','sequential']
legend     = [s.replace('_',' ').capitalize() for s in strategies]
markers    = ['.','x','+']

# Get list of molecules
molecules = clc.index_molecules()
molecules.sort()

# Pre-compute all results
if (False) :
   for (name, instances) in molecules :
      for instance in instances :
         getResults(instance[1])

# Display all results
if (False) :
   for (name, instances) in molecules :
      for instance in instances :
         instance = instance[1]
         for method in ['cz','cnot','greedy1','greedy2'] :
            (x,y) = getPlotCoordinates(instance, strategies, method)
            for (xx,yy,marker) in zip(x,y,markers) :
               plt.plot(xx,yy,marker)
            plt.title("%s-%s (%s), method = %s" %
                      (instance.molecule, instance.basis, instance.encoding, method))
            plt.show()



# Display select results
settings = [("Hamiltonians/H2O_6-31G_26qubits", "cz"),
            ("Hamiltonians/HCl_STO3g_20qubits", "greedy2")]

if (True):
   cl.ensureDirExists('fig')
   pltIndex = 0
   plt.rcParams.update({'font.size': 16})
   for (directory, method) in settings :
      molecule = clc.index_molecule(directory)
      for (name, instance) in molecule :
         (x,y) = getPlotCoordinates(instance, strategies, method)
         for (xx,yy,marker) in zip(x,y,markers) :
            plt.plot(xx,yy,marker,markersize=8)

         pltIndex += 1
         if (pltIndex == 3) :
            plt.legend(legend)

         plt.xlabel('Partition size')
         plt.ylabel("CNot count")
         plt.gcf().tight_layout()

         prefix = ("fig/Figure_partition_%s_%s_%s_%s" %
                   (instance.molecule, instance.basis_star, instance.encoding_abbrv, method))
         plt.savefig("%s-uncropped.pdf" % prefix, transparent=True)
         os.system("pdfcrop %s-uncropped.pdf %s.pdf" % (prefix,prefix))
         plt.close()
