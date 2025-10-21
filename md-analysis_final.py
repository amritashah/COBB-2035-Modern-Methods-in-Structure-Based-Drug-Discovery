#!/usr/bin/env python3

# feel free to clean up the imports
import MDAnalysis as mda
from MDAnalysis.coordinates.memory import MemoryReader
from MDAnalysis.analysis.base import AnalysisFromFunction
from MDAnalysis.analysis import align, rms
from MDAnalysis.analysis.dihedrals import *
from MDAnalysis.analysis import contacts

import matplotlib.pyplot as plt
import numpy as np
import sys, argparse


parser = argparse.ArgumentParser(description='Analyze a MD trajectory using MDAnalysis')
parser.add_argument("topology",help="Topology PDB file from OpenMM")
parser.add_argument("trajectory",help="Trajectory file")
args = parser.parse_args()

# KEEP all the hardcoded output file names as this is how your script will be graded.

# Preprocess trajectory
u = mda.Universe(args.topology, args.trajectory) # load topology & trajectory files into universe
prot = u.select_atoms("protein") # want to analyze only the protein, so removing the solvent
coordinates = AnalysisFromFunction(lambda ag: ag.positions.copy(), prot).run().results['timeseries']
u_prot = mda.Merge(prot)
u_prot.load_new(coordinates, format=MemoryReader) # sub-universe created

# RMSD Calculations
align.AlignTraj(u_prot, u_prot, select="backbone", ref_frame=0, in_memory=True).run()
rmsd_first = rms.RMSD(u_prot, # universe to align
                  u_prot, # reference atom group
                  select='backbone', # group to superimpose and calculate RMSD
                  ref_frame=0).run() # frame index of reference
rmsd_last = rms.RMSD(u_prot, # universe to align
                  u_prot, # reference atom group
                  select='backbone', # group to superimpose and calculate RMSD
                  ref_frame=-1).run() # frame index of reference
np.save('rmsd_first.npy',rmsd_first.rmsd[:, 2])
np.save('rmsd_last.npy',rmsd_last.rmsd[:, 2])

plt.figure()
plt.plot(rmsd_first.results.rmsd[:, 0], rmsd_first.results.rmsd[:, 2], label='First')
plt.plot(rmsd_last.results.rmsd[:, 0], rmsd_last.results.rmsd[:, 2], alpha=0.8, label='Last')
plt.xlabel('Frame')
plt.ylabel(r'RMSD ($\AA$)')
plt.legend()
plt.savefig("rmsd_plot.png", dpi=300)
plt.close()

# RMSF Calculations
u = mda.Universe(args.topology, args.trajectory) # load topology & trajectory files into universe
prot = u.select_atoms("protein") # want to analyze only the protein, so removing the solvent
coordinates = AnalysisFromFunction(lambda ag: ag.positions.copy(), prot).run().results['timeseries']
u_prot = mda.Merge(prot)
u_prot.load_new(coordinates, format=MemoryReader) # sub-universe created

average = align.AverageStructure(u_prot, u_prot, select='backbone', ref_frame=0).run()
ref = average.results.universe
align.AlignTraj(u_prot, ref, select='backbone', in_memory=True).run()

ca_atoms = u_prot.select_atoms('name CA')
ca_rmsf = rms.RMSF(ca_atoms).run()
np.save('ca_rmsf.npy',ca_rmsf.rmsf) #alpha carbon only rmsf

plt.figure()
plt.plot(ca_atoms.resids, ca_rmsf.results.rmsf)
plt.xlabel('Residue number')
plt.ylabel(r'RMSF $\AA$)')
plt.savefig("rmsf_plot.png", dpi=300)
plt.close()

# iterate through each residue of the protein and set the tempfactor attribute for every atom in the residue to the alpha-carbon RMSF value; this is necessary so every atom in the residue is coloured with the alpha-carbon RMSF

rmsf_all = rms.RMSF(u_prot.atoms).run()
rmsf_all_results = rmsf_all.results.rmsf
prot.atoms.tempfactors = rmsf_all_results

prot.atoms.write('rmsf.pdb') # must write pdb with tempfactors set for every atom

# Ramachandran 
res62 = u_prot.select_atoms('resid 62')
rama = Ramachandran(res62).run()
np.save('ramachandran.npy',rama.results['angles'])

plt.figure(figsize=(6, 6))
rama.plot(ref=True, marker='.', color='black', s=5, alpha=0.3)
plt.xlim(-180, 180)
plt.ylim(-180, 180)
plt.savefig('rama_resi61.png', dpi=300)
plt.close()

# Janin
res62 = u_prot.select_atoms('resid 62')
janin = Janin(res62).run()
np.save('janin.npy',janin.results['angles'])

plt.figure(figsize=(6, 6))
janin.plot(ref=True, marker='.', color='black', s=5, alpha=0.3)
plt.savefig('janin_62.png', dpi=300)



