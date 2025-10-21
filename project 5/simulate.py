#!/usr/bin/env python3

import openmm
import openmm.app as app
from openmm.unit import *
from pdbfixer.pdbfixer import PDBFixer
from openmmforcefields.generators import SMIRNOFFTemplateGenerator
from openff.toolkit import Molecule
from openmm import unit as openmm_unit

import sys, argparse
from io import StringIO
from typing import Iterable
import numpy as np

     
parser = argparse.ArgumentParser(description='Simulate a PDB using OpenMM')
parser.add_argument("--pdb",required=True,help="PDB file of the receptor, including any ion")
parser.add_argument("--ligand",required=True,help="Ligand file name")

parser.add_argument("--steps",type=int,default=125000000,help="Number of 2fs time steps")
parser.add_argument("--etime",type=int,default=100,help="Picoseconds spent in EACH equilibration phase.")
parser.add_argument("--system_pdb",type=str,default="system.pdb",help="PDB of system AFTER energy minimization")

parser.add_argument("--temperature",type=float,default="300",help="Temperature for simulation in Kelvin")

parser.add_argument("--etrajectory",type=str,default="etrajectory.dcd",help="Equilibration  dcd trajectory name")
parser.add_argument("--trajectory",type=str,default="trajectory.dcd",help="Production dcd trajectory name")
parser.add_argument("--einfo",type=argparse.FileType('wt'),default=sys.stdout,help="Equilibration simulation info file")
parser.add_argument("--info",type=argparse.FileType('wt'),default=sys.stdout,help="Production simulation info file")

args = parser.parse_args()
temperature = args.temperature*kelvin

#Load PDB and add any missing residues/atoms/hydrogens (at pH 7)
fixer = PDBFixer(args.pdb)
fixer.findMissingResidues()
fixer.findMissingAtoms()
fixer.addMissingAtoms()
fixer.addMissingHydrogens(7.0)

print("Fixed up PDB",fixer.missingResidues)

#Load ligand and parameterize
ligand = Molecule.from_file(args.ligand)

#create xml of parameterization
smirnoff = SMIRNOFFTemplateGenerator(molecules=ligand)
ligandxml = StringIO()
ligandxml.write(smirnoff.generate_residue_template(ligand))

#pdb format
ligandpdb = StringIO()
ligand.to_file(ligandpdb,file_format='pdb')

#load into openmm and fix
ligandpdb.seek(0) #set to beginning of "file"
ligandxml.seek(0)
lfixer = PDBFixer(pdbfile=ligandpdb)
forcefield  = app.ForceField(ligandxml)
lfixer.addMissingHydrogens(7.0,forcefield=forcefield)

#combine receptor and ligand
modeller = app.Modeller(fixer.topology, fixer.positions)
modeller.add(lfixer.topology,lfixer.positions)


#BEFORE adding the water box, perform a minimization of the structure
ligandxml.seek(0)
#amber14/tip3p.xml contains parameters for Mg
forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3p.xml',  'implicit/gbn2.xml',ligandxml)
integrator = openmm.VerletIntegrator(2*femtosecond)
system = forcefield.createSystem(modeller.topology)
minimizer = app.Simulation(modeller.topology, system, integrator)
minimizer.context.setPositions(modeller.positions)
minimizer.minimizeEnergy()
state = minimizer.context.getState(getPositions=True)
modeller.positions = state.getPositions()



#Using the minimized positions of the protein, add an octahedron water box
#with 1nm of padding (neutralize).
ligandxml.seek(0)
forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3p.xml', ligandxml)
modeller.addSolvent(forcefield, padding=1.0*nanometer,boxShape='octahedron')
topology = modeller.topology
positions = modeller.positions


# Setup the Simulation
# PME, 1nm cutoff, HBonds constrained
# LangevinMiddleIntegrator with friction=1/ps  and 2fs timestep
system = forcefield.createSystem(topology, nonbondedMethod=app.PME, 
                nonbondedCutoff=1*nanometer,  constraints=app.HBonds)

barostat = openmm.MonteCarloBarostat(1*atmospheres, temperature, 0)
system.addForce(barostat)
    
integrator = openmm.LangevinMiddleIntegrator(temperature, 1/picosecond, 2*femtoseconds)
simulation = app.Simulation(topology, system, integrator)
simulation.context.setPositions(positions)

# Energy minimize
print('Performing energy minimization...')
simulation.minimizeEnergy()

state = simulation.context.getState(getEnergy=True,getPositions=True)
print("Energy",state.getPotentialEnergy())

# Write out PDB of topology and positions of system before any simulation
app.PDBFile.writeFile(topology, state.getPositions(), open(args.system_pdb,'wt'))

if args.etime == 0:
  sys.exit(0)
  
# Equilibrate the system in two phases for 100ps.  
# We will report on the state of the system at a more fine-grained level during equilibration

stateReporter = app.StateDataReporter(args.einfo, reportInterval=50,step=True,temperature=True,volume=True,potentialEnergy=True,speed=True)
dcdReporter = app.DCDReporter(args.etrajectory, 500)
simulation.reporters.append(stateReporter)
simulation.reporters.append(dcdReporter)

# In the first equilibration step, we gently warm up the NVT system to
# 300K. Starting at 3K, simulate 1ps at a time, increasing the temperature
# by 3K every picosecond for a total of 100ps

  
print('First Equilibration...')

tempstep = temperature/100
tsteps = args.etime*5
T = tempstep
for i in range(100):
    integrator.setTemperature(T)
    simulation.step(tsteps)
    T += tempstep
    

# In the second equilibration step, enable the MonteCarloBarostat barostat 
#at 1atm pressure and a frequency of 25. 

print('Second Equilibration...')
barostat.setFrequency(25)
simulation.step(args.etime*500)


# Production simulation
# Replace equilibration reporters that report every 10ps.
simulation.reporters = []
simulation.currentStep = 0

stateReporter = app.StateDataReporter(args.info, reportInterval=5000,step=True,temperature=True,volume=True,potentialEnergy=True,speed=True)
dcdReporter = app.DCDReporter(args.trajectory, 5000)
simulation.reporters.append(stateReporter)
simulation.reporters.append(dcdReporter)

print('Simulating...')

simulation.step(args.steps)

