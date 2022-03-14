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
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import FuncFormatter
from itertools import permutations

def plotZ(Z, exportFilename=None) :
   (m,n) = Z.shape
   cmap = colors.LinearSegmentedColormap.from_list("white_and_gray", [(1, 1, 1), (0.6, 0.6, 0.6)], N=2)
   fig, ax = plt.subplots()
   im = ax.imshow(Z.T,cmap=cmap)
   ax.set_yticklabels([])
   ax.set_xticklabels([])
   ax.set_yticks([])
   ax.set_xticks([])

   for i in range(1,m) :
      plt.plot([-0.5+i,-0.5+i],[-0.5,-0.5+n],color='k',linewidth=0.7)
   for i in range(1,T.n) :
      plt.plot([-0.5,-0.5+m],[-0.5+i,-0.5+i],color='k',linewidth=0.7)

   for i in range(n) :
      v = Z[:,i]
      c = np.sum(v[:-1] != v[1:]) + v[0] + v[-1]
      ax.text(m-0.25,i, str(c), fontsize=12, ha='left', va='center')

   if (exportFilename) :
      plt.gcf().tight_layout()
      plt.savefig(exportFilename + "-uncropped.pdf", transparent=True)
      plt.close()
      os.system("pdfcrop %s-uncropped.pdf %s.pdf" % (exportFilename, exportFilename))
   else :
      plt.show()


# Make sure the figure directory exists
cl.ensureDirExists('fig')

# Create the test problem
M = cl.create_basic_problem(7,0)
C = cl.generate_full_rank_weights(20,7,seed=1)
M = np.dot(C,M) % 2

# Apply diagonalization and get the final Z matrix
T = cl.Tableau(M)
R = cl.RecordOperations(T.n)
T.addRecorder(R)
cl.zeroX_algorithm1_cz(T)
T = cl.Tableau(M)
R.apply(T)
Z = T.getZ()

# Plot the results
plotZ(Z,'fig/Figure_9a')
print("Original: %d" % cl.countCNot(Z))

idx = cl.orderZ(Z)
plotZ(Z[idx,:],'fig/Figure_9b')
print("Sorted  : %d" % cl.countCNot(Z[idx,:]))


# Generate histogram of actual permutations
if (True) :
   base = list(range(7))
   count = []
   for idx2 in permutations(base) :
       idx1 = cl.orderZ(Z[:,idx2])
       count.append(cl.countCNot(Z[idx1,:][:,idx2]))

   def format_percentage(y, position):
       return str(100 * y)

   # Count is always even
   plt.hist(count,bins=list(range(min(count)-1,max(count)+2,2)),rwidth=0.9,density=True)
   plt.gca().set_xticklabels([str(x) for x in range(min(count),max(count)+1,2)],fontsize=16)
   plt.gca().set_xticks(list(range(min(count),max(count)+1,2)))
   plt.gca().yaxis.set_major_formatter(FuncFormatter(format_percentage))
   plt.xlabel('Number of CNOT gates',fontsize=16)
   plt.ylabel("Percentage",fontsize=16)

   for tick in plt.gca().yaxis.get_major_ticks():
      tick.label.set_fontsize(16)

   plt.gcf().tight_layout()

   ratio = 0.5
   xleft, xright = plt.gca().get_xlim()
   ybottom, ytop = plt.gca().get_ylim()
   plt.gca().set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)

   plt.savefig("fig/Figure_9c-uncropped.pdf", transparent=True)
   plt.close()
   os.system("pdfcrop fig/Figure_9c-uncropped.pdf fig/Figure_9c.pdf")
