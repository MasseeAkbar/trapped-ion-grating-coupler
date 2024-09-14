##"""
#Copyright (c) 2020 Ansys Inc. """

######## IMPORTS ########
# General purpose imports
import os,sys
import numpy as np
import scipy as sp
import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
directory=r'%s' %str(Path('__file__').parent.absolute())

#default path for current release 
sys.path.append("C:\\Program Files\\Lumerical\\v232\\api\\python\\") 
sys.path.append("C:\\Program Files\\Lumerical\\v232\\python\\") 
sys.path.append(directory) #Current directory

from lumjson import LumEncoder, LumDecoder

# Optimization specific imports
from lumopt.utilities.load_lumerical_scripts import load_from_lsf
from lumopt.utilities.wavelengths import Wavelengths
from lumopt.geometries.parameterized_geometry import ParameterizedGeometry
from lumopt.geometries.polygon import FunctionDefinedPolygon
from lumopt.figures_of_merit.modematch import ModeMatch
from lumopt.optimizers.generic_optimizers import ScipyOptimizers
from lumopt.optimization import Optimization
from lumopt.optimization import SuperOptimization
from lumopt.utilities.materials import Material
import lumapi
import matplotlib

import matplotlib.pyplot as plt

cur_path = str(directory)

# Simulation parameters (from original code)
lambda_c = 1.092e-6
bandwidth_in_nm = 0
height = 220e-9
etch_depth = 220e-9
y0 = 0
x_begin = -5.1e-6
x_end = 22e-6
n_grates = 25
min_feature_size = 0.1

indexSi = 3.47668
indexSiO2 = 1.44401

params_file_apod = os.path.join(cur_path, "pid_optim_1.json")
result_file = os.path.join(cur_path, "pid_optim_final.json")
base_sim_2d = os.path.join(cur_path, "pid_grating_coupler_2D_TE_base.fsp")
base_script_2d = os.path.join(cur_path, "pid_grating_coupler_2D_TE_base.lsf")
sim_2d_final = os.path.join(cur_path, "pid_grating_coupler_2D_final.fsp")


def grating_params_pos(params, output_waveguide_length=0.48e-6):
    """Generate vertices for the grating coupler geometry."""
    y3 = y0 + height
    y1 = y3 - etch_depth
    x0 = params[0] * 1e-6  # First parameter: starting position

    verts = np.array([[x_begin, y0], [x_begin, y3], [x0, y3], [x0, y1]])

    for i in range(n_grates - 1):
        x1 = x0 + params[i * 2 + 1] * 1e-6  # Width of the etch
        x2 = x1 + params[i * 2 + 2] * 1e-6  # Width up
        verts = np.concatenate((verts, np.array([[x1, y1], [x1, y3], [x2, y3], [x2, y1]])), axis=0)
        x0 = x2

    x1 = x0 + params[(n_grates - 1) * 2 + 1] * 1e-6  # Final etch width
    verts = np.concatenate((verts, np.array([[x1, y1], [x1, y3], [x_end, y3], [x_end, y0]])), axis=0)

    return verts

def etched_grating(params, fdtd, update_only=False):
    """Update grating geometry in FDTD simulation."""
    
    verts = grating_params_pos(params)
    if not update_only:
        fdtd.addpoly()
        fdtd.set("name", "grating_poly")
        fdtd.setnamed("grating_poly", "x", 0)
        fdtd.setnamed("grating_poly", "y", y0)
        fdtd.set("z", 0)
        fdtd.set("z span", 0.48e-6)
        fdtd.setnamed("grating_poly", "index", indexSi)
    fdtd.setnamed("grating_poly", "vertices", verts)

def runGratingOptimization(bandwidth_in_nm, etch_depth, n_grates, initial_params, min_feature_size, working_dir):
    """Run the FDTD optimization for the grating coupler."""
    # Initialize grating parameters if not provided
    if initial_params is None:
        params = np.zeros(2 * n_grates)
        for i in range(n_grates):
            params[i * 2] = 0.2  # Width up
            params[i * 2 + 1] = 0.4 * ((i + 1) / (n_grates + 1))  # Width of the deep etch
            params[0] = 0  # Starting position of the grating
    else:
        params = initial_params

    bounds = [(min_feature_size, 1)] * (2 * n_grates)
    bounds[0] = (-3, 3)  # Special bounds for starting position

    # Geometry definition
    geometry = ParameterizedGeometry(
        func=etched_grating,
        initial_params=params,
        bounds=bounds,
        dx=1e-5
    )

    # Figure of Merit definition
    fom = ModeMatch(
        monitor_name='fom',
        mode_number=1,
        direction='Forward',
        target_T_fwd=lambda wl: np.ones(wl.size),
        norm_p=1
    )

    # Optimization algorithm setup
    optimizer = ScipyOptimizers(
        max_iter=1,
        method='L-BFGS-B',
        scaling_factor=1,
        pgtol=1e-6,
        ftol=0
    )

    # Load the base simulation script
    base_script = load_from_lsf(os.path.join(working_dir, base_script_2d))

    # Define wavelengths
    lambda_start = lambda_c * 1e9 - bandwidth_in_nm / 2
    lambda_end = lambda_c * 1e9 + bandwidth_in_nm / 2
    wavelengths = Wavelengths(
        start=lambda_start * 1e-9,
        stop=lambda_end * 1e-9,
        points=int(bandwidth_in_nm / 10) + 1
    )

    # Optimization object
    opt = Optimization(
        base_script=base_script,
        wavelengths=wavelengths,
        fom=fom,
        geometry=geometry,
        optimizer=optimizer,
        use_var_fdtd=False,
        hide_fdtd_cad=False,
        use_deps=True,
        plot_history=False,
        store_all_simulations=True,
        save_global_index=False
    )
    
    # Run the optimization
    result = opt.run(working_dir=working_dir)

    return result[0], result[1]  # Return the FoM and gradients

def FDTD_Simulation(grating_coupler_coords):
    """The main function that takes in grating coupler coordinates and returns the FoM and gradients."""
    
    working_dir = cur_path
    
    # Open the parameters file to get the initial parameters
    with open(params_file_apod) as fh:
        initial_params = json.load(fh, cls=LumDecoder)["initial_params"]

    # Run the optimization with the given grating coupler coordinates
    FoM, gradients = runGratingOptimization(
        bandwidth_in_nm=bandwidth_in_nm,
        etch_depth=etch_depth,
        n_grates=n_grates,
        initial_params=grating_coupler_coords,  # These are your NN predicted coordinates
        min_feature_size=min_feature_size,
        working_dir=working_dir
    )

    # Use the gradients to update parameters (not the coordinates themselves)
    learning_rate = 0.01  # Set a learning rate
    opt_params_2D = grating_coupler_coords - learning_rate * gradients  # Proper gradient update step

    # Save results in the specified format
    with open(os.path.join(cur_path, result_file), "w") as fh:
        json.dump({"initial_params": opt_params_2D}, fh, cls=LumEncoder, indent=4)

    # np.savetxt(os.path.join(cur_path, "apod_2D_params.txt"), opt_params_2D)
    #np.savetxt(os.path.join(cur_path, 'fom.csv'), [FoM], delimiter=',')
    #np.savetxt(os.path.join(cur_path, 'gradient.csv'), gradients, delimiter=',')

    # Run the 2D simulation with optimized structure and save the final result
    with lumapi.FDTD(filename=os.path.join(cur_path, base_sim_2d), hide=True) as fdtd:
        vtx = grating_params_pos(opt_params_2D)
        fdtd.addpoly()
        fdtd.set("vertices", vtx)
        fdtd.set("x", 0)
        fdtd.set("y", 0)
        fdtd.set("z", 0)
        fdtd.set("z span", 0.48e-6)
        fdtd.set("index", indexSi)
        fdtd.save(os.path.join(cur_path, sim_2d_final))

    return FoM, gradients  # Return the actual gradients