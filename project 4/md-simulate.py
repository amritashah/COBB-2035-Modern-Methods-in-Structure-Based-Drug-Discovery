#!/usr/bin/env python3

from openmm import *
from openmm.app import *
from openmm.unit import *
from pdbfixer.pdbfixer import PDBFixer
import sys, argparse


parser = argparse.ArgumentParser(description='Simulate a PDB using OpenMM')
parser.add_argument("--pdb",required=True,help="PDB file name")
parser.add_argument("--temperature",type=float,default=300,help="Temperature for simulation in Kelvin")
parser.add_argument("--steps",type=int,default=125000000,help="Number of 2fs time steps")
parser.add_argument("--etrajectory",type=str,default="etrajectory.dcd",help="Equilibration  dcd trajectory name")
parser.add_argument("--trajectory",type=str,default="trajectory.dcd",help="Production dcd trajectory name")
parser.add_argument("--einfo",type=argparse.FileType('wt'),default=sys.stdout,help="Equilibration simulation info file")
parser.add_argument("--info",type=argparse.FileType('wt'),default=sys.stdout,help="Production simulation info file")
parser.add_argument("--system_pdb",type=str,default="system.pdb",help="PDB of system, can be used as topology")

args = parser.parse_args()

#Load PDB and add any missing residues/atoms/hydrogens (at pH 7) (pdbfixer)
fixer = PDBFixer(filename=args.pdb)
fixer.findMissingResidues()
fixer.findNonstandardResidues()
fixer.replaceNonstandardResidues()
fixer.removeHeterogens(False)
fixer.findMissingAtoms()
fixer.addMissingAtoms()
fixer.addMissingHydrogens(7.0)

#BEFORE adding the water box, perform a minimization of the structure
#This is necessary because, as you will recall from class, OpenMM
#doesn't do a great job adding residues and if we wait until after adding
#water to minimize, the water will "get in the way" and prevent the minimizer
#from resolving the clashes in the modelled residues.
#Use the Amber14 forcefield with the 'implicit/gbn2.xml' water model
ff = ForceField('amber14-all.xml', 'implicit/gbn2.xml')
system_min = ff.createSystem(fixer.topology, nonbondedMethod=NoCutoff, constraints=HBonds)

integrator_min = LangevinIntegrator(args.temperature*kelvin, 1/picosecond, 2*femtosecond)
simulation_min = Simulation(fixer.topology, system_min, integrator_min)
simulation_min.context.setPositions(fixer.positions)

simulation_min.minimizeEnergy()
state_min = simulation_min.context.getState(getPositions=True)
positions_min = state_min.getPositions()

#Using the minimized positions of the protein, add an octahedron water box
#with 1nm of padding (neutralize).
modeller = Modeller(fixer.topology, positions_min)
ff_ex = ForceField('amber14-all.xml','amber14/tip3p.xml')
modeller.addSolvent(ff_ex, model='tip3p', boxShape='octahedron', padding=1.0*nanometer, neutralize=True)

# Write out PDB of topology and positions of system to args.system_pdb
with open(args.system_pdb, 'w') as f:
    PDBFile.writeFile(modeller.topology, modeller.positions, f)

# Setup the Simulation
# Note you need to add the barostat (1atm) before creating the simulation object,
# even though it should be disabled initially (interval = 0) 
# PME, 1nm cutoff, HBonds constrained
# LangevinMiddleIntegrator with friction=1/ps  and 2fs timestep
sys_pdb = PDBFile(args.system_pdb)
system = ff_ex.createSystem(modeller.topology, nonbondedMethod=PME, nonbondedCutoff=1.0*nanometer, constraints=HBonds)
barostat = MonteCarloBarostat(1*atmosphere, args.temperature*kelvin, 0)
system.addForce(barostat)
integrator = LangevinMiddleIntegrator(args.temperature*kelvin, 1/picosecond, 2*femtosecond) 

simulation = Simulation(modeller.topology, system, integrator)
simulation.context.setPositions(modeller.positions)

# Energy minimize
simulation.minimizeEnergy()

# Equilibrate the system in two phases for a total 70ps.  
# Ideally we would equilibrate longer (e.g. 100ps each phase), but the autograder will time out.
# We will report on the state of the system at a more fine-grained level during equilibration
# Don't change this output as it will be auto-graded
stateReporter = StateDataReporter(args.einfo, reportInterval=50,step=True,temperature=True,volume=True,potentialEnergy=True,speed=True)
dcdReporter = DCDReporter(args.etrajectory, 500)
simulation.reporters.append(stateReporter)
simulation.reporters.append(dcdReporter)

# In the first equilibration step, we gently warm up the NVT system to
# the desired temperature T. Starting at T/100, simulate 0.5ps at a time, increasing the temperature
# by T/100 every half picosecond for a total of 50ps
T_target = args.temperature*kelvin
T_start = T_target/100
dt = 0.5*picosecond
steps_per_dt = int(dt/(2*femtosecond))
for i in range(100):
    integrator.setTemperature(T_start+(i*T_start))
    simulation.step(steps_per_dt)

# In the second equilibration step, enable the MonteCarloBarostat barostat 
#at 1atm pressure and a frequency of 25. Simulate for 20ps
barostat.setFrequency(25)
simstep = 20*picosecond
steps_in_sim = int(simstep/(2*femtosecond))
simulation.step(steps_in_sim)

# Replace equilibration reporters with reporters that report every 10ps.
simulation.reporters = [] # append new ones
simulation.currentStep = 0
report_interval = int((10*picosecond)/(2*femtosecond))
simulation.reporters.append(StateDataReporter(args.info, reportInterval=report_interval,step=True,temperature=True,volume=True,potentialEnergy=True,speed=True))
simulation.reporters.append(DCDReporter(args.trajectory, report_interval))


# Production NTP simulation for args.steps
simulation.step(args.steps) 

