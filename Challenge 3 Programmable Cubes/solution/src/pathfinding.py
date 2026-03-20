import numpy as np
from numba import njit
from heapq import heappush, heappop
from programmable_cubes_UDP import ProgrammableCubes
from programmable_cubes_UDP import programmable_cubes_UDP
from solution.src.misc import *


def reconstruct_path(parents,end):
    path = [end]
    try:
        p = parents[tuple(end)][0]
    except:
        return np.array(path)
    while p:
        path.append(np.array(p))
        try:
            p = parents[tuple(p)][0]
        except:
            break
    path.reverse()
    return np.array(path)

def reconstruct_rotations(parents,end):
    chrom = []
    try:
        chrom.append(parents[tuple(end)][1])
        p = parents[tuple(end)][0]
    except:
        return np.array(chrom)
    while p:
        try:
            chrom.append(parents[tuple(p)][1])
            p = parents[tuple(p)][0]
        except:
            break
    chrom.reverse()
    return np.array(chrom)
def rotations_to_chromosome(id:int,rotations:list):
    chrom = []
    for i in np.arange(len(rotations)):
        chrom.append(id)
        chrom.append(rotations[i])
    return chrom

def astar_cubes(cubes : ProgrammableCubes, id : int, end : np.ndarray, budget : int=np.inf, heur : callable=manhattan):
    """ 
    cubes - working cube class
    id = id of the active cube
    end - end, target position
    heur - distance for the heuristic
    budget - method continues when g value higher than this is reached

    f ... the heuristic + #rots to get there
    g ... #rots to get there
    parents ... stores [parent_position,rotation] - to be extracted as set of rotations afterwards
    """
    start = np.array(cubes.cube_position[id]) # ! make copy
    # Initialize arrays
    positions_to_search = []
    f = {}
    g = {}
    parents = {}

    # Initialize start
    heappush(positions_to_search,(heur(start,end),tuple(start)))
    f[tuple(start)] = 0
    g[tuple(start)] = heur(start,end)

    while len(positions_to_search) > 0:
        value, current_pos = heappop(positions_to_search)
        # goal check
        if np.sum(np.abs(current_pos-end)) == 0:
            cubes.apply_update_at_position(id,start) # return to original state
            chrom = rotations_to_chromosome(id,reconstruct_rotations(parents,end))
            path = reconstruct_path(parents,end)
            return chrom,path,True
        

        # Escape when too many iterations
        if budget < g[current_pos]:
            continue

        # Move cube to currently searched position
        cubes.apply_update_at_position(id,current_pos)
        # Try all rotations, check distances, add to positions_to_search
        for rot in ROTS:
            if not cubes.apply_single_update_step(id,rot):
                continue
            new_pos = cubes.cube_position[id]

            # Get the number of steps to get to c and the previous values if exists
            tmp_g = g[current_pos] + 1
            previous_g = np.inf
            try:
                previous_g = g[tuple(new_pos)]
            except:
                pass
            #
            if previous_g == np.inf or tmp_g < previous_g:
                parents[tuple(new_pos)] = [current_pos,rot]
                g[tuple(new_pos)] = tmp_g
                f[tuple(new_pos)] = heur(new_pos,end)+tmp_g
                heappush(positions_to_search,(f,tuple(new_pos)))
            cubes.apply_single_update_step(id,inv_rot(rot),verbose=True)            
    cubes.apply_update_at_position(id,start) # return to original state
    return [],[],False

def dijkstra_cubes_multiple(cubes : ProgrammableCubes, id : int, ends : np.ndarray, budget : int=np.inf):
    """ 
    Look for closest path to one of the ends. Does not use any heuristic

    cubes - working cube class
    id = id of the active cube
    ends - end, target position
    budget - method continues when g value higher than this is reached

    g ... #rots to get there
    parents ... stores [parent_position,rotation] - to be extracted as set of rotations afterwards

    returns chromosome, path, success // results can be accessed using path[path.shape[0] - 1]
    """
    start = np.array(cubes.cube_position[id]) # ! make copy
    # Initialize arrays
    positions_to_search = []
    g = {}
    parents = {}

    # Initialize start
    g[tuple(start)] = 0
    heappush(positions_to_search,(g[tuple(start)],tuple(start)))

    while len(positions_to_search) > 0:
        value, current_pos = heappop(positions_to_search)
        
        # goal check
        for end in ends:
            if np.sum(np.abs(current_pos-end)) == 0:
                cubes.apply_update_at_position(id,start) # return to original state
                return rotations_to_chromosome(id,reconstruct_rotations(parents,end)),reconstruct_path(parents,end),True
        

        # Escape when too many iterations
        if budget < g[current_pos]:
            continue

        # Move cube to currently searched position
        cubes.apply_update_at_position(id,current_pos)
        # Try all rotations, check distances, add to positions_to_search
        for rot in ROTS:
            if not cubes.apply_single_update_step(id,rot):
                continue
            new_pos = cubes.cube_position[id]

            # Get the number of steps to get to c and the previous values if exists
            tmp_g = g[current_pos] + 1
            previous_g = np.inf
            try:
                previous_g = g[tuple(new_pos)]
            except:
                pass
            #
            if previous_g == np.inf or tmp_g < previous_g:
                parents[tuple(new_pos)] = [current_pos,rot]
                g[tuple(new_pos)] = tmp_g
                heappush(positions_to_search,(g[tuple(new_pos)],tuple(new_pos)))
            cubes.apply_single_update_step(id,inv_rot(rot),verbose=True)            
    cubes.apply_update_at_position(id,start) # return to original state
    return [],[],False

def axis_search(cubes : ProgrammableCubes, id:int, c:np.ndarray, budget:int = np.inf,heur = manhattan):
    """
    finds path for cube with index "id" to the coordinate "c" with valid moves on "cubes"

    returns chromosomes as chrom, coordinates in path, success
    """
    start = np.array(cubes.cube_position[id])
    # Initialize variables
    chrom = []
    path = []
    c1 = cubes.cube_position[id]
    c2 = c
    # Speedups
    if not is_connected(c2,cubes.cube_position):
        return [],[],False
    
    # Main loop
    while np.sum(c1 == c2) != 3 and len(chrom) < budget:
        # rots clockwise,counter-clockwise around x,z,y
        # 0,1 - to change y,z
        # 2,3 to change x,y
        # 4,5 to change x,z
        # idea 1
        # if x is not equal
        # rotate 2,3,4,5 and find where closer
        #dir = 0 # x
        changed_smth = False
        for dir in np.arange(0,3):
            diff = np.abs(c1 - c2)
            dist = heur(c1,c2)
            if diff[dir] != 0: # change only coordinates which dont match
                for rot in ROTS_XYZ[dir]:
                    # Apply rotation
                    res = cubes.apply_single_update_step(id,rot)
                    # Compute new difference
                    c1 = cubes.cube_position[id]
                    new_dist = heur(c1,c2)
                    if res == 1 and dist > new_dist: # legal and better
                        path.append(c1.copy())
                        chrom.append(id)
                        chrom.append(rot)
                        dist = new_dist
                        changed_smth = True
                        continue
                    elif res == 1: # legal but not better, revert
                        cubes.apply_single_update_step(id,inv_rot(rot))
                        c1 = cubes.cube_position[id]
                        continue
        if changed_smth == False: # break if all moves are tried
            break
    success = heur(c1,c2) == 0
    cubes.apply_update_at_position(id,start) # return to original state
    #success = 1
    return chrom,path,success

def axis_search_always_move(cubes : ProgrammableCubes, id:int, c:np.ndarray, budget:int = np.inf,heur = manhattan):
    """
    finds path for cube with index "id" to the coordinate "c" with valid moves on "cubes"

    returns chromosomes as chrom, coordinates in path, success
    """
    start = np.array(cubes.cube_position[id])
    # Initialize variables
    chrom = []
    path = []
    c1 = cubes.cube_position[id]
    c2 = c
    # Speedups
    if not is_connected(c2,cubes.cube_position):
        return [],[],False
    
    # Main loop
    while np.sum(c1 == c2) != 3 and len(chrom) < budget:
        # rots clockwise,counter-clockwise around x,z,y
        # 0,1 - to change y,z
        # 2,3 to change x,y
        # 4,5 to change x,z
        # idea 1
        # if x is not equal
        # rotate 2,3,4,5 and find where closer
        #dir = 0 # x
        changed_smth = False
        for dir in np.arange(0,3):
            diff = np.abs(c1 - c2)
            dist = heur(c1,c2)
            if diff[dir] != 0: # change only coordinates which dont match
                for rot in ROTS_XYZ[dir]:
                    # Apply rotation
                    res = cubes.apply_single_update_step(id,rot)
                    # Compute new difference
                    c1 = cubes.cube_position[id]
                    new_dist = heur(c1,c2)
                    if res == 1 and dist > new_dist: # legal and better
                        path.append(c1.copy())
                        chrom.append(id)
                        chrom.append(rot)
                        dist = new_dist
                        changed_smth = True
                        continue
                    elif res == 1: # legal but not better, revert
                        cubes.apply_single_update_step(id,inv_rot(rot))
                        c1 = cubes.cube_position[id]
                        continue
        if changed_smth == False: # break if all moves are tried
            break
    #success = heur(c1,c2) == 0
    #cubes.apply_update_at_position(id,start) # return to original state
    success = 1 # always moves
    return chrom,path,success