from numba import njit
import seaborn as sns
from programmable_cubes_UDP import ProgrammableCubes
from programmable_cubes_UDP import programmable_cubes_UDP
import numpy as np
import random
from heapq import heappush, heappop



### Working with cubes

ROTS = [0,1,2,3,4,5]
ROT_X = [2,3,4,5]
ROT_Y = [0,1,2,3]
ROT_Z = [0,1,4,5]
ROTS_XYZ = [ROT_X,ROT_Y,ROT_Z]

@njit(cache=True)
def inv_rot(rot : int) -> int:
    match rot:
        case 0:
            return 1
        case 1:
            return 0
        case 2:
            return 3
        case 3:
            return 2
        case 4:
            return 5
        case 5:
            return 4

def get_valid_rots(id:int,cubes:ProgrammableCubes) -> list:
    valid_rots = []
    for i in ROTS:
        res = cubes.apply_single_update_step(id,i)
        if res == 1:
            cubes.apply_single_update_step(id,inv_rot(i))
            valid_rots.append(i)
    return valid_rots
def force_random_move(id:int,cubes:ProgrammableCubes,rand : int = 0) -> int:
    for i in ROTS:
        res = cubes.apply_single_update_step(id,(i+rand)%6)
        if res == 1:
            return i
    return -1

DIRS = [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]
def is_connected(c:np.ndarray,positions:np.ndarray):
    """ Checks if there is at least one cube in some direction """
    # Target one has to be empty
    idx = np.where((positions == [c[0],c[1],c[2]]).all(axis=1))[0]
    if idx.size > 0:
        return False
    # One of the adjacent ones has to be filled
    for d in DIRS:
        idx = np.where((positions == [c[0]+d[0],c[1]+d[1],c[2]+d[2]]).all(axis=1))[0]
        if idx.size > 0:
            return True
    return False

## Detecting wrong cubes

def contains_coord(arr:np.ndarray, coord:np.ndarray):
    for id,c in enumerate(arr):
        if np.sum(c==coord) == len(coord):
            return id
    return -1

def have_wrong_type(pos1,type1,pos2,type2):
    wrong_type_ids = []
    for i in np.arange(len(pos1)):
        id = contains_coord(pos2,pos1[i])
        if id != -1 and type1[i] != type2[id]:
            wrong_type_ids.append(i)
    return wrong_type_ids

def get_wrong_cube_ids(pos1:np.ndarray,pos2:np.ndarray):
    """
    does not include color
    """
    arr, cnt = np.unique(np.concat([pos1,pos2]), axis=0, return_counts=True)
    wrong = arr[cnt==1]
    pos1_ids = []
    for coord in wrong:
        id = contains_coord(pos1,coord) 
        if id != -1:
            pos1_ids.append(id)
    pos1_ids = np.array(pos1_ids)

    pos2_ids = []
    for coord in wrong:
        id = contains_coord(pos2,coord) 
        if id != -1:
            pos2_ids.append(id)
    pos2_ids = np.array(pos2_ids)

    return pos1_ids,pos2_ids

## Pairing cubes

def pair_colours(wi:np.ndarray,wt:np.ndarray,ci:np.ndarray,ti:np.ndarray,ct:np.ndarray,tt:np.ndarray):
    """ 
    pairs based on colours, does not consider distance
    """
    pairs = np.zeros(shape=(len(wi),4),dtype=np.int64)
    if not wi.size > 0:
        return pairs
    shuffled_wt = wt.copy()
    random.shuffle(shuffled_wt)
    pairs[:,0] = wi
    pairs[:,1] = shuffled_wt
    pairs[:,2] = ti[wi]
    pairs[:,3] = tt[shuffled_wt]
    for i in np.arange(len(wi)):
        if pairs[i,2] != pairs[i,3]:
            for j in np.arange(i+1,len(wi)):
                if pairs[i,2] == pairs[j,3]:
                    tmp_id = pairs[i,1]
                    tmp_type = pairs[i,3]
                    pairs[i,1] = pairs[j,1]
                    pairs[i,3] = pairs[j,3]
                    pairs[j,1] = tmp_id
                    pairs[j,3] = tmp_type
                    break
    return pairs


## Astar path finding on cubes
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
@njit(cache=True)
def manhattan(a:np.ndarray,b:np.ndarray):
    return np.sum(np.abs(a-b))
def dijkstra(a,b):
    return 0

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



## Final function for finding chromosomes

def find_chromosome(udp:programmable_cubes_UDP,pairing : callable = pair_colours, pathfinding=axis_search,random_shuffle=False):
    # Initialize cubes
    cubes = ProgrammableCubes(udp.final_cube_positions)
    ti = udp.initial_cube_types
    ci = udp.final_cube_positions
    ct = udp.target_cube_positions
    tt = udp.target_cube_types
    types = np.arange(np.max(ti)+1)

    # Initialize outputs
    chrom = []
    #path = []
    iter = 0
    ESCAPE = 5
    placed_stat = []
    successes = 0
    while len(chrom) < udp.get_nix() and iter < ESCAPE:
        # Get cubes in position with wrong type, move them
        wrong_type_ids = have_wrong_type(cubes.cube_position,ti,ct,tt)
        for id in wrong_type_ids:
            move = force_random_move(id,cubes)
            if move != -1:
                chrom.extend([id,move])

        ## TODO: move wrongly placed ones first to good position, they get built around and are stuck


        # Get places which are wrong
        wi, wt = get_wrong_cube_ids(cubes.cube_position,ct)
        assert(len(wi) == len(wt)) # otherwise unsolvable
        # Pair them
        src_dest_all = pairing(wi,wt,cubes.cube_position,ti,ct,tt)

        # Find paths between them
        for src_dest in src_dest_all:
            if src_dest[2] != src_dest[3]: # the place has to be correct, otherwise counterproductive
                continue
            id = src_dest[0]
            end = ct[src_dest[1]]
            if not is_connected(end,cubes.cube_position):
                continue
            tmp_chrom,tmp_path,success = pathfinding(cubes,id,end,50)
            if success:
                chrom.extend(tmp_chrom)
                # Apply to cubes
                tmp_chrom = np.concatenate([tmp_chrom,[-1]])
                cubes.apply_chromosome(tmp_chrom,False)
                #path.extend(tmp_path)
            successes += (int)(success)
        placed_stat.append(successes)

        # Experiment: let all wrong cubes move 1 time randomly
        # results from few runs: overall improves fitness, maybe increases num of steps at start but helps in the end
        # wi, wt = get_wrong_cube_ids(cubes.cube_position,ct)
        # for id in wi:
        #     move = force_random_move(id,cubes)
        #     if move != -1:
        #         chrom.extend([id,move])

        iter += 1
        mistakes_cnt = len(wi)+len(wrong_type_ids)
        if mistakes_cnt == 0:
            break
        print(f"mistakes:{len(wi)}+{len(wrong_type_ids)} budget status: {len(chrom)/udp.get_nix()*100} % used, escape:{iter/ESCAPE*100}% successes: {successes}")
    return np.array(chrom,dtype=int)


def format_chromosome(udp:programmable_cubes_UDP,chromosome):
    """ From tutorial 1"""
    udp.fitness(np.array([-1]))
    cubes = ProgrammableCubes(udp.final_cube_positions)

    filtered_chromosome = []

    for i in range(int(len(chromosome)/2)):
        cube_id = chromosome[i*2]
        move = chromosome[i*2+1]
        done = cubes.apply_single_update_step(cube_id, move)
        # done is 1 if the move is legal and 0 otherwise
        if done == 1:
            filtered_chromosome += [cube_id, move]
    # The part to be evaluated by the fitness function ends here,
    # thus we add -1
    filtered_chromosome += [-1]
    
    # Fill up the remaining chromosome with 0s (will not be evaluated)
    if len(filtered_chromosome) < udp.get_nix():
        for i in range(udp.get_nix() - 1 - len(filtered_chromosome)):
            filtered_chromosome += [0]
        # Mandatory entry of -1 at the very end
        filtered_chromosome += [-1]
        
    return np.array(filtered_chromosome,dtype=int)

def end_chromosome(chromosome):
    return np.concatenate([chromosome,np.array([-1])])

def invert_chromosome(udp:programmable_cubes_UDP,chromosome):
    """
    expect even length 1D array
    """
    inv_chromosome = []
    # extract moves and ids
    moves = chromosome[::-2]
    ids = chromosome[(len(chromosome)-2)::-2]
    # invert rotations
    moves = [inv_rot(moves[i]) for i in range(len(moves))]
    # convert ids
    # i is from target, id is from achieved, id -> i
    # udp is the inverse problem
    achieved_config = udp.final_cube_positions
    avail_ids_achieved_config = np.ones(shape=len(achieved_config),dtype=bool)
    target_config = udp.target_cube_positions
    avail_ids_target_config = np.ones(shape=len(achieved_config),dtype=bool)
    id_conversion = {}
    for i in np.arange(len(target_config)):
        coord = target_config[i]
        id = contains_coord(achieved_config,coord)
        if id != -1:
            id_conversion[id] = i
            avail_ids_achieved_config[i] = False
            avail_ids_target_config[id] = False
    # randomly match the other cubes
    for i in np.arange(len(target_config)):
        for j in np.arange(i,len(target_config)):
            if avail_ids_achieved_config[i] and avail_ids_target_config[j]:
                id_conversion[i] = j
                avail_ids_achieved_config[i] = False
                avail_ids_target_config[j] = False
                break
    #print(id_conversion)
    #print(ids)
    ids = [id_conversion[id] for id in ids]
    #print(ids)
    #print(moves)
    # 
    inv_chromosome = [[ids[i],moves[i]] for i in range(len(moves))]
    inv_chromosome = np.array(inv_chromosome).flatten()
    return inv_chromosome

