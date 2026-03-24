import numpy as np
from numba import njit
from heapq import heappush, heappop
from programmable_cubes_UDP import ProgrammableCubes
from programmable_cubes_UDP import programmable_cubes_UDP
from solution.src.misc import *
import random


def reconstruct(parents:dict,end:np.ndarray,val_id=1):
    """ 
    val_id : 0 for path, 1 for rotations
    """
    chrom = []
    end_key = tuple(end)
    if end_key not in parents:
        return np.array(chrom)
    p = parents[end_key][0]
    chrom.append(parents[end_key][val_id])
    while p in parents:
        chrom.append(parents[p][val_id])
        p = parents[p][0]
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
    start = np.array(cubes.cube_position[id],dtype=int) # ! make copy
    end_key = tuple(end)
    # Initialize arrays
    positions_to_search = []
    f = {}
    g = {}
    parents = {}

    # Initialize start
    start_key = tuple(start)
    g[start_key] = 0
    f[start_key] = heur(start,end)
    heappush(positions_to_search,(g[start_key],start_key))

    while len(positions_to_search) > 0:
        value, current_pos_key = heappop(positions_to_search)
        # goal check
        if current_pos_key==end_key:
            cubes.apply_update_at_position(id,start) # return to original state
            chrom = rotations_to_chromosome(id,reconstruct(parents,end,1))
            #path = reconstruct(parents,end)
            return chrom,True
        

        # Escape when too many iterations
        if budget < g[current_pos_key]:
            continue

        # Move cube to currently searched position
        cubes.apply_update_at_position(id,current_pos_key)
        # Try all rotations, check distances, add to positions_to_search
        for rot in ROTS:
            if not cubes.apply_single_update_step(id,rot):
                continue
            new_pos = cubes.cube_position[id]
            new_pos_key = tuple(new_pos)

            # Get the number of steps to get to c and the previous values if exists
            tmp_g = g[current_pos_key] + 1
            previous_g = np.inf
            try:
                previous_g = g[new_pos_key]
            except:
                pass
            #
            if previous_g == np.inf or tmp_g < previous_g:
                parents[new_pos_key] = [current_pos_key,rot]
                g[new_pos_key] = tmp_g
                f[new_pos_key] = heur(new_pos,end)+tmp_g
                heappush(positions_to_search,(f[new_pos_key],new_pos_key))
            cubes.apply_single_update_step(id,inv_rot(rot))            
    cubes.apply_update_at_position(id,start) # return to original state
    return np.array([],dtype=int),False

def astar_cubes_distance(cubes : ProgrammableCubes, id : int, end : np.ndarray, budget : int=np.inf, heur : callable=manhattan, dist = 2):
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
    start = np.array(cubes.cube_position[id],dtype=int) # ! make copy
    end_key = tuple(end)
    # Initialize arrays
    positions_to_search = []
    f = {}
    g = {}
    parents = {}

    # Initialize start
    start_key = tuple(start)
    g[start_key] = 0
    f[start_key] = heur(start,end)
    heappush(positions_to_search,(g[start_key],start_key))

    while len(positions_to_search) > 0:
        value, current_pos_key = heappop(positions_to_search)
        # goal check
        if manhattan(np.array(current_pos_key),end) < dist:
            cubes.apply_update_at_position(id,start) # return to original state
            chrom = rotations_to_chromosome(id,reconstruct(parents,current_pos_key,1))
            #path = reconstruct(parents,end)
            return chrom,True
        

        # Escape when too many iterations
        if budget < g[current_pos_key]:
            continue

        # Move cube to currently searched position
        cubes.apply_update_at_position(id,current_pos_key)
        # Try all rotations, check distances, add to positions_to_search
        for rot in ROTS:
            if not cubes.apply_single_update_step(id,rot):
                continue
            new_pos = cubes.cube_position[id]
            new_pos_key = tuple(new_pos)

            # Get the number of steps to get to c and the previous values if exists
            tmp_g = g[current_pos_key] + 1
            previous_g = np.inf
            try:
                previous_g = g[new_pos_key]
            except:
                pass
            #
            if previous_g == np.inf or tmp_g < previous_g:
                parents[new_pos_key] = [current_pos_key,rot]
                g[new_pos_key] = tmp_g
                f[new_pos_key] = heur(new_pos,end)+tmp_g
                heappush(positions_to_search,(f[new_pos_key],new_pos_key))
            cubes.apply_single_update_step(id,inv_rot(rot),verbose=True)            
    cubes.apply_update_at_position(id,start) # return to original state
    return np.array([],dtype=int),False

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
    start_key = tuple(start)
    # Initialize arrays
    positions_to_search = []
    g = {}
    parents = {}

    # Initialize start
    g[start_key] = 0
    heappush(positions_to_search,(g[start_key],start_key))

    while len(positions_to_search) > 0:
        value, current_pos_key = heappop(positions_to_search)
        
        # goal check
        for end in ends:
            if np.sum(np.abs(current_pos_key-end)) == 0:
                cubes.apply_update_at_position(id,start) # return to original state
                return rotations_to_chromosome(id,reconstruct(parents,end)),end,True
        

        # Escape when too many iterations
        if budget < g[current_pos_key]:
            continue

        # Move cube to currently searched position
        cubes.apply_update_at_position(id,current_pos_key)
        # Try all rotations, check distances, add to positions_to_search
        for rot in ROTS:
            if not cubes.apply_single_update_step(id,rot):
                continue
            new_pos = cubes.cube_position[id]
            new_pos_key = tuple(new_pos)

            # Get the number of steps to get to c and the previous values if exists
            tmp_g = g[current_pos_key] + 1
            previous_g = np.inf
            try:
                previous_g = g[new_pos_key]
            except:
                pass
            #
            if previous_g == np.inf or tmp_g < previous_g:
                parents[new_pos_key] = [current_pos_key,rot]
                g[new_pos_key] = tmp_g
                heappush(positions_to_search,(g[new_pos_key],new_pos_key))
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
        return [],False
    
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
    return chrom,success

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
        return [],False
    
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
                        #path.append(c1.copy())
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
    return chrom,success


def astar_move_neighbourhoods(cubes : ProgrammableCubes,id,c,budget):
    """
    experimental, does not work
    """
    # check if not stuck

    # try smth
    start = np.array(cubes.cube_position[id])
    surroundings_start = np.array(cubes.cube_surroundings[id])
    cubes.apply_update_at_position(id,c)
    surroundings_end = np.array(cubes.cube_surroundings[id])
    cubes.apply_update_at_position(id,start)

    print(f"Moving cube {id} from {start} to {c}")

    # Shuffle and get first few
    random.shuffle(surroundings_start)
    random.shuffle(surroundings_end)
    #print(surroundings_start)
    #print(surroundings_end)

    for i in surroundings_start:
        move_start = force_random_move(i,cubes)

        for j in surroundings_end:
            move_end = force_random_move(j,cubes)

            tmp_chrom, success = astar_cubes(cubes,id,c)
            if success:
            #print(success)
                # revert pending moves
                cubes.apply_single_update_step(j,inv_rot(move_end))
                cubes.apply_single_update_step(i,inv_rot(move_start))
                # return all moves combined
                return np.concatenate([
                    np.array([i,move_start]),
                    np.array([j,move_end]),
                    tmp_chrom,
                    np.array([j,inv_rot(move_end)]),
                    np.array([i,inv_rot(move_start)])
                    ]), True

            if move_end != -1:
                cubes.apply_single_update_step(j,inv_rot(move_end))
        if move_start != -1:
            cubes.apply_single_update_step(i,inv_rot(move_start))

    return [],False

def astar_move_neighbourhoods_all(cubes : ProgrammableCubes,id,c,budget):
    """
    experimental, does not work
    """
    # check if not stuck

    # try smth
    start = np.array(cubes.cube_position[id])
    surroundings_start = np.array(cubes.cube_surroundings[id])
    cubes.apply_update_at_position(id,c)
    surroundings_end = np.array(cubes.cube_surroundings[id])
    cubes.apply_update_at_position(id,start)

    #print(f"Moving cube {id} from {start} to {c}")

    # Shuffle and get first few
    random.shuffle(surroundings_start)
    random.shuffle(surroundings_end)
    surroundings = np.concatenate([surroundings_start,surroundings_end])
    #print(surroundings_start)
    #print(surroundings_end)
    
    moves = []
    for i in surroundings:
        move = force_random_move(i,cubes)
        if move != -1:
            moves.append([i,move])
    


    tmp_chrom, success = astar_cubes(cubes,id,c)
    if success:
        for move_start in moves:
            if move_start != -1:
                cubes.apply_single_update_step(move_start[0],inv_rot(move_start[1]))
       
        # return all moves combined
        
        moves_chromosome = np.array(moves)
        moves_chromosome_back = invert_chromosome_same_problem(moves_chromosome)
        formatted_chrom = np.concatenate([
            moves_chromosome.flatten(),
            tmp_chrom,
            moves_chromosome_back.flatten()
            ])
        return formatted_chrom, True

    moves.reverse()
    for move_start in moves:
        if move_start[1] != -1:
            cubes.apply_single_update_step(move_start[0],inv_rot(move_start[1]))


    return np.array([]),False

def empty_pathfinding(cubes : ProgrammableCubes,id,c,budget):
    """
    Pathfinding that does not even try to find chromosome but applied some moves and reverts them.
    Used for debugging.
    """
    surroundings_start = np.array(cubes.cube_surroundings[id])
    moves_start = []
    for i in surroundings_start:
        move = force_random_move(i,cubes)
        if move != -1:
            moves_start.append([i,move])
    moves_start.reverse()
    for move_start in moves_start:
        if move_start[1] != -1:
            #print(move_start)
            cubes.apply_single_update_step(int(move_start[0]),inv_rot(move_start[1]))
    return np.array([]),False

def brute_force_move_to(cubes: ProgrammableCubes, id:int, c:np.ndarray, budget:int=np.inf, max_attempts:int=100, max_moves:int=5):
    """
    Forcing astar by applying random sample of rotations and trying astar
    """
    start = np.array(cubes.cube_position[id], dtype=int)

    # step 1: try astar
    chrom, ok = astar_cubes(cubes, id, c, budget)
    if ok:
        return chrom, True

    # step 2: apply random movements until astar possible or max attempts
    num_cubes = len(cubes.cube_position)
    candidates = [i for i in range(num_cubes) if i != id]
    random.shuffle(candidates)
    #candidates = list(cubes.cube_surroundings[id])

    for attempt in range(max_attempts):
        if len(candidates) == 0:
            break
        picks = random.sample(candidates, min(max_moves, len(candidates)))
        applied = []

        for mover in picks:
            rot = force_random_move(mover, cubes)
            if rot != -1:
                applied.append((mover, int(rot)))

        if len(applied) == 0:
            continue

        # try A* after these relocations
        tmp_chrom, ok = astar_cubes(cubes, id, c, budget)
        if ok:
            # print(f"Moved {id} to {c}")
            # build forward/backward chromosomes
            forward = np.array([x for pair in applied for x in pair], dtype=int)
            back_pairs = [(m, inv_rot(r)) for (m, r) in reversed(applied)]
            backward = np.array([x for pair in back_pairs for x in pair], dtype=int)

            # revert physical moves now
            for mover, rot in reversed(applied):
                cubes.apply_single_update_step(mover, inv_rot(rot))

            full = np.concatenate([forward, tmp_chrom, backward])
            cubes.apply_update_at_position(id, start)
            return full, True

        # revert applied moves and restore cube id position
        for mover, rot in reversed(applied):
            cubes.apply_single_update_step(mover, inv_rot(rot))
        cubes.apply_update_at_position(id, start)

    cubes.apply_update_at_position(id, start)
    return np.array([], dtype=int), False


def brute_force_move_to_neighbourhoods(cubes: ProgrammableCubes, id:int, c:np.ndarray, budget:int=np.inf, max_attempts:int=100, max_moves:int=5):
    """
    Forcing astar by applying random sample of rotations and trying astar
    """
    start = np.array(cubes.cube_position[id], dtype=int)

    # step 1: try astar
    chrom, ok = astar_cubes(cubes, id, c, budget)
    if ok:
        return chrom, True

    # step 2: apply random movements until astar possible or max attempts

    start = np.array(cubes.cube_position[id])
    surroundings_start = np.array(cubes.cube_surroundings[id])
    cubes.apply_update_at_position(id,c)
    surroundings_end = np.array(cubes.cube_surroundings[id])
    cubes.apply_update_at_position(id,start)
    random.shuffle(surroundings_start)
    random.shuffle(surroundings_end)
    surroundings = np.concatenate([surroundings_start,surroundings_end])
    candidates = list(surroundings)

    for attempt in range(max_attempts):
        if len(candidates) == 0:
            break
        picks = random.sample(candidates, min(max_moves, len(candidates)))
        applied = []

        for mover in picks:
            rot = force_random_move(mover, cubes)
            if rot != -1:
                applied.append((mover, int(rot)))

        if len(applied) == 0:
            continue

        # try A* after these relocations
        tmp_chrom, ok = astar_cubes(cubes, id, c, budget)
        if ok:
            # print(f"Moved {id} to {c}")
            # build forward/backward chromosomes
            forward = np.array([x for pair in applied for x in pair], dtype=int)
            back_pairs = [(m, inv_rot(r)) for (m, r) in reversed(applied)]
            backward = np.array([x for pair in back_pairs for x in pair], dtype=int)# maybe dont do the backward chromosome here

            # revert physical moves now
            for mover, rot in reversed(applied):
                cubes.apply_single_update_step(mover, inv_rot(rot))

            full = np.concatenate([forward, tmp_chrom,backward])
            cubes.apply_update_at_position(id, start)
            return full, True

        # revert applied moves and restore cube id position
        for mover, rot in reversed(applied):
            cubes.apply_single_update_step(mover, inv_rot(rot))
        cubes.apply_update_at_position(id, start)

    cubes.apply_update_at_position(id, start)
    return np.array([], dtype=int), False


def force_random_move_pathfinding(cubes : ProgrammableCubes, id:int, c:np.ndarray, budget:int = np.inf,heur = manhattan):
    """
    moves cube 1 time randomly if possible and returns pathfinding format
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
    
    rot = force_random_move(id, cubes)
    if rot != -1:
        return [id,rot], True
    return chrom,False




