#!/usr/bin/env python3

import openmm
from openmm.app import *
from openmm import *
from openmm.unit import *
import numpy as np
import matplotlib
matplotlib.use('Agg') # don't open a display
import matplotlib.pyplot as plt
import scipy
import argparse
import scipy.signal

def dihedral(p1, p2, p3, p4):
    '''Return dihedral angle in radians between provided points.
    This is the same calculation used by OpenMM for force calculations. '''
    v12 = np.array(p1-p2)
    v32 = np.array(p3-p2)
    v34 = np.array(p3-p4)
    
    #compute cross products 
    cp0 = np.cross(v12,v32)
    cp1 = np.cross(v32,v34)
    
    #get angle between cross products
    dot = np.dot(cp0,cp1)
    if dot != 0:
        norm1 = np.dot(cp0,cp0)
        norm2 = np.dot(cp1,cp1)
        dot /= np.sqrt(norm1*norm2)
        
    if dot > 1.0:
        dot = 1.0
    elif dot < -1.0:
        dot = -1.0

    if dot > 0.99 or dot < -0.99:
        #close to acos singularity, so use asin isntead
        cross = np.cross(cp0,cp1)
        scale = np.dot(cp0,cp0)*np.dot(cp1,cp1)
        angle = np.arcsin(np.sqrt(np.dot(cross,cross)/scale))
        if dot < 0.0:
            angle = np.pi - angle
    else:
        angle = np.arccos(dot)
    
    #figure out sign
    sdot = np.dot(v12,cp1)
    angle *= np.sign(sdot)
    return angle
    
    
def make_rotation_matrix(p1, p2, angle):
    '''Make a rotation matrix for rotating angle radians about the p2-p1 axis'''
    # https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    vec = np.array((p2-p1))
    x,y,z = vec/np.linalg.norm(vec)
    cos = np.cos(angle)
    sin = np.sin(angle)
    R = np.array([[cos+x*x*(1-cos), x*y*(1-cos)-z*sin, x*z*(1-cos)+y*sin],
                  [y*x*(1-cos)+z*sin, cos+y*y*(1-cos), y*z*(1-cos)-x*sin],
                  [z*x*(1-cos)-y*sin, z*y*(1-cos)+x*sin, cos+z*z*(1-cos)]])
    return R
    
    
def moving_atoms(pdb,a=1,b=2):
    '''Identify the atoms on the b side of a dihedral with the central
    atoms a and b.  This is not the most efficient algorithm, but 
    for small molecules it really does not matter.
    A boolean mask of these atoms is returned.'''
    moving = np.zeros(pdb.topology.getNumAtoms())
    moving[b] = 1
    moving[a] = -1
    changed = True
    while changed:
        changed = False
        for b in pdb.topology.bonds():
            if (moving[b.atom1.index] + moving[b.atom2.index]) == 1:
                moving[b.atom1.index] = moving[b.atom2.index] = 1
                changed = True
    moving[1] = 0    
    return moving.astype(bool)      


parser = argparse.ArgumentParser(description='CompStruct Assignment 3')
parser.add_argument('pdb',help='input PDB file')
parser.add_argument('--aindex',help='index of first dihedral atom',type=int,default=1)
parser.add_argument('--bindex',help='index of second dihedral atom',type=int,default=2)
parser.add_argument('--integrator',help='integrator to use',choices=['verlet','langevin'],default='langevin')
parser.add_argument('--temp',help='temperature to simulate at',type=int,default=300)
parser.add_argument('--steps',help='number of simulation steps (1fs)',type=int,default=1000000)
parser.add_argument('--output',help='output filename for graph',default='out.png')
args = parser.parse_args()

# Making the Reporter
class AngleReporter(object):
    def __init__(self, reportInterval):
        self._reportInterval = reportInterval
        self.dihedrals = []

    def describeNextReport(self, simulation):
        # tells OpenMM we need positions at each reporting step
        return (self._reportInterval, True, False, False, False)
    
    def report(self, simulation, state):
        pos = state.getPositions().value_in_unit(nanometer)
        d = dihedral(*pos[:4])
        if d < 0: d += 2*np.pi
        self.dihedrals.append(d)

# setup openmm system using amber 14 forcefield which defines the potential energy function
pdb = PDBFile(args.pdb)
ff = ForceField('amber14-all.xml')
system = ff.createSystem(pdb.topology,ignoreExternalBonds=True)
# set integrator
if args.integrator == 'verlet':
    integrator = VerletIntegrator(1*femtosecond)
else:
    integrator = LangevinIntegrator(args.temp*kelvin, 1/picosecond, 1*femtosecond)
    integrator.setRandomNumberSeed(42)

# simulation code below
simulation = Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(pdb.positions)
simulation.reporters.append(AngleReporter(reportInterval=1))
simulation.step(args.steps)

# store atom positions
# note that more code is necessary for this to be a general solution
# as opposed to only working with our carefully prepared inputs
origpos = np.array(pdb.getPositions()._value)
newpos = origpos.copy()

capos = origpos[args.aindex]
cpos = origpos[args.bindex]

# a boolean mask of the atoms that should be rotated around the dihedral
mask = moving_atoms(pdb, args.aindex, args.bindex)

# we will output a pdb of the result of our rotations around the dihedral
out = open('rot.pdb','wt')
energies = []
angles = []
for d in range(0,360,1):
    # rotate atoms
    R = make_rotation_matrix(capos,cpos,np.deg2rad(d))    
    newpos[mask] = np.matmul(R,origpos[mask].T).T
     
    # setPositions to newpos in simulation (TODO)
    simulation.context.setPositions(newpos)

    
    # get simulation.context state, fetching energy (TODO)
    energy1 = simulation.context.getState(getPositions=False, getEnergy=True)
   
    # record the energy (TODO)
    energies.append(energy1.getPotentialEnergy()) #*(joule/mole))

    # record the dihedral
    d = dihedral(*newpos[:4])
    if d < 0: d += 2*np.pi
    angles.append(np.rad2deg(d))
    
    #write out modified structure
    PDBFile.writeModel(pdb.topology, newpos*nanometers,out,modelIndex=d)

out.close()

T = args.temp * kelvin
    
# compute probabilities of each state (TODO)
R = MOLAR_GAS_CONSTANT_R # 8.314 J/K*mol
boltz_factors = np.array([np.exp(-en / (R * T)) for en in energies])
probs = (boltz_factors / boltz_factors.sum()).tolist()

dihedrals = simulation.reporters[-1].dihedrals

#convert to degrees    
dihedrals_deg = (np.rad2deg(dihedrals)+360)%360
cnts,bins = np.histogram(dihedrals_deg,range(361))

#this is what the autograder will look at (with a small time step)
print(cnts)    

#make plot
plt.hist(dihedrals_deg,bins=range(361),density=True);    
plt.plot(angles,probs,label=r'$\frac{1}{\hat{Z}}e^{\frac{-U}{k_BT}}$')
plt.xlim(0,360)
plt.legend(fontsize=16)
plt.xlabel('Dihedral Angle (Degrees)')
plt.ylabel('Frequency/Probability')
plt.title(os.path.split(args.pdb)[-1][:-4].upper())
plt.savefig(args.output,bbox_inches='tight')