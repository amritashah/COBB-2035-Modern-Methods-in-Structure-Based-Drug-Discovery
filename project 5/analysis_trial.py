#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore", message="The Bio.Application")

# feel free to clean up the imports
import MDAnalysis
from MDAnalysis.coordinates.memory import MemoryReader
from MDAnalysis.analysis.base import AnalysisFromFunction
from MDAnalysis.analysis import align, rms
import MDAnalysis.transformations as transformations
from MDAnalysis.analysis.distances import distance_array
from collections import Counter

warnings.filterwarnings("ignore",message="DCDReader")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys, argparse


parser = argparse.ArgumentParser(description='Assignment 5 analysis of an MD trajectory')
parser.add_argument("--topology",required=True,help="Topology PDB file from OpenMM of ligand/receptor system")
parser.add_argument("--trajectory",required=True,help="Trajectory file of ligand/receptor system")
parser.add_argument("--step",type=int,default=100,help="Frequency to downsample simulation.")
parser.add_argument("--cutoff",type=float,default=4.0,help="Distance threshold for contacts analysis")
parser.add_argument("--ligand_sel",required=True,help="Selection of ligand for RMSD analysis")
parser.add_argument("--selA",required=True,help="Selection A for contacts analysis")
parser.add_argument("--selB",required=True,help="Selection B for contacts analysis")
parser.add_argument("--output_prefix",required=True,help="Prefix for all output files")
parser.add_argument("--output_traj",required=False,help="Base filename for outputing the processed trajectory")

args = parser.parse_args()

# KEEP all the hardcoded output file names as this is how your script will be graded.

# Preprocess trajectory
U = MDAnalysis.Universe(args.topology,args.trajectory,in_memory=True,in_memory_step=args.step)
# only loading in every 100th frame to reduce memory usage
protein = U.select_atoms('protein') # only protein
notwater = U.select_atoms('not resname HOH WAT') # everything except water (so protein+ligands, ions etc.)
transforms = [transformations.unwrap(protein), # removes periodic boundary discontinuities for the protein (so it's whole and continuous)
              transformations.center_in_box(protein), # centers protein in periodic simulation box
              transformations.wrap(U.select_atoms('not protein'),compound='residues')]
                    # wraps all non-protein residues back into primary box around the protein
              
U.trajectory.add_transformations(*transforms) # every frame in U.trajectory will have these transformations

u = MDAnalysis.Merge(notwater).load_new(
         AnalysisFromFunction(lambda ag: ag.positions.copy(), notwater).run().results['timeseries'],
         format=MemoryReader)
# run a fxn over all frames of U, collecting atomic positions of notwater after transformations
# .run.results[timeseries] gives np array of shape (n_frames, n_atoms. 3)
# MDAnalysis creates new universe u with topology of non-water watoms and the transformed coordinates.
         
if args.output_traj:
    system = u.select_atoms('all')       
    system.write(f'{args.output_traj}.pdb')
    system.write(f'{args.output_traj}.dcd',frames=u.trajectory[1:])
# if output filename is specified, save PDB of processed system & dcd of all frames except first one

# Align trajectory using alpha carbons to first frame
align.AlignTraj(u, u, select="protein and name CA", ref_frame=0, in_memory=True).run()
ligand = u.select_atoms(args.ligand_sel)

# RMSD Calculations
u.trajectory[0]
ref_pos_first = ligand.positions.copy()   # ligand positions in first frame
u.trajectory[-1]
ref_pos_last = ligand.positions.copy()    # ligand positions in last frame
u.trajectory[0]  # reset trajectory to first frame

# --- RMSD vs first frame ---
first_rmsds = []
for ts in u.trajectory:
    diff = ligand.positions - ref_pos_first
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    first_rmsds.append(rmsd)
first_rmsds = np.array(first_rmsds)

# --- RMSD vs last frame ---
u.trajectory[0]  # reset to first frame
last_rmsds = []
for ts in u.trajectory:
    diff = ligand.positions - ref_pos_last
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    last_rmsds.append(rmsd)
last_rmsds = np.array(last_rmsds)


plt.figure(figsize=(6,4))
plt.plot(np.arange(len(first_rmsds)), first_rmsds, label='RMSD to First Frame')
plt.plot(np.arange(len(last_rmsds)), last_rmsds, alpha=0.8, label='RMSD to Last Frame')
plt.xlabel('Frame')
plt.ylabel('Ligand RMSD (Å)')
#plt.ylim(top=3.0)
plt.legend()
plt.savefig(f'{args.output_prefix}_rmsd.png',bbox_inches='tight',dpi=300)

#first_rmsds should have the rmsd of args.ligand_selection to the first frame
#and last_rmsds to the last frame in the alpha carbon aligned structure
#these files must be output with these file names for the autograder
np.save('first_rmsds.npy', first_rmsds)
np.save('last_rmsds.npy', last_rmsds)

# Contacts analysis
# use MDAnalysis.analysis.distances.distance_array to calculate all pairwise
# distances between selA and selB at each frame of the simulation and count
# the percent of time atoms are within args.cutoff

selA = u.select_atoms(args.selA)
selB = u.select_atoms(args.selB)

contact_counts = Counter()
total_frames = 0

for ts in u.trajectory:
    nearby = u.select_atoms("not name H* and byres around 10 (resid 61)")
    for res in nearby.residues:
        contact_counts[res.resid] += 1
    total_frames += 1

# Convert to contact frequency (fraction of frames)
contact_freq = {resid: count / total_frames for resid, count in contact_counts.items()}

# Sort by frequency
for resid, freq in sorted(contact_freq.items(), key=lambda x: x[1], reverse=True):
    resname = u.select_atoms(f"resid {resid}").residues.resnames[0]
    print(f"{resname}{resid}: {freq:.2f}")

resids = np.array(list(contact_freq.keys()))
freqs = np.array(list(contact_freq.values()))
data = np.column_stack((resids, freqs))
np.save(f"{args.output_prefix}res61_contacts.npy", data)


cutoff = args.cutoff

contacts = np.zeros((len(selA), len(selB)), dtype=float)

for frame in u.trajectory:
    dist = distance_array(selA.positions, selB.positions)
    # if selA has n atoms & selB has m atoms, dist is. a n x m array of pairwise dist. btwn A & B
    contacts += (dist < cutoff)

contacts /= len(u.trajectory)
# heat map needs ticklabels indicates residue number, name, and atom name
ylabels = [f'{atom.resname}{atom.resid}:{atom.name}' for atom in selA]
xlabels = [f'{atom.resname}{atom.resid}:{atom.name}' for atom in selB]

plt.figure(figsize=(len(xlabels)*.2+1,len(ylabels)*.2+1))
sns.heatmap(contacts,xticklabels=xlabels,yticklabels=ylabels, vmin=0,vmax=1, cmap="viridis",cbar_kws={'pad':0.01,'label':'Contact Frequency'})
plt.tight_layout()
plt.savefig(f'{args.output_prefix}_contacts.png',bbox_inches='tight',dpi=300)

# For the autograder - should be a len(selA) x len(selB) matrix of numbers
# between 0 and 1 representing contact frequencies.
np.save('contacts.npy',contacts)