import numpy as np
from numba import njit
from programmable_cubes_UDP import ProgrammableCubes
from programmable_cubes_UDP import programmable_cubes_UDP
from solution.src.misc import *
from solution.src.pathfinding import *
from solution.src.pairing import *



def debug_plot(udp:programmable_cubes_UDP):
    ti = udp.initial_cube_types
    ci = udp.final_cube_positions
    ct = udp.target_cube_positions
    tt = udp.target_cube_types
    types = np.arange(np.max(ti)+1)
    wti = have_wrong_type(ci,ti,ct,tt)
    # wti - wrong type ids, from cube_position
    # These are cubes away from the structure and the hollow points in the structure
    wpi, epi = get_wrong_cube_ids(ci,ct)
    if len(wti) == 0:
        wti = np.array([], dtype=int)
    if len(epi) == 0:
        epi = np.array([], dtype=int)
    if len(wpi) == 0:
        wpi = np.array([], dtype=int)
    # wpi - wrong place ids, from cube_position
    # epi - empty place ids, from target configuration ct

    wrong_ids_initial = np.concatenate([wti,wpi])
    coordinates_initial = np.concatenate([ct[wti],ct[wpi]])
    debug_types_initial = -np.ones(shape=(coordinates_initial.shape[0]),dtype=int)
    wrong_ids_target = np.concatenate([wti,epi])
    coordinates_target = np.concatenate([ct[epi],ct[wti]])
    debug_types_target = -2*np.ones(shape=(coordinates_target.shape[0]),dtype=int)
    all_coords =np.concatenate([coordinates_initial,coordinates_target])
    all_types = np.concatenate([debug_types_initial,debug_types_target])
    print(all_coords.shape)
    print(all_types.shape)
    np.save(f"out_jwst_wrong",all_coords)

    udp.debug_plot("none",cube_type_to_plot=types,custom_config=all_coords,custom_cube_types=all_types)


def save_achieved_config(udp:programmable_cubes_UDP,chromosome:np.ndarray):
    udp.fitness(end_chromosome(chromosome))
    np.save(f"out_config",udp.final_cube_positions)

def save_mistakes(udp:programmable_cubes_UDP):
    ti = udp.initial_cube_types
    ci = udp.final_cube_positions
    ct = udp.target_cube_positions
    tt = udp.target_cube_types
    wti = have_wrong_type(ci,ti,ct,tt)
    wpi, epi = get_wrong_cube_ids(ci,ct)
    if len(wti) == 0:
        wti = np.array([], dtype=int)
    if len(epi) == 0:
        epi = np.array([], dtype=int)
    if len(wpi) == 0:
        wpi = np.array([], dtype=int)

    coordinates_initial = np.concatenate([ct[wti],ci[wpi]])
    coordinates_target = np.concatenate([ct[epi],ct[wti]])
    all_coords =np.concatenate([coordinates_initial,coordinates_target])
    
    np.save(f"out_wrong",all_coords)