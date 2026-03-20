import numpy as np
from numba import njit
from programmable_cubes_UDP import ProgrammableCubes
from programmable_cubes_UDP import programmable_cubes_UDP

@njit(cache=True)
def manhattan(a:np.ndarray,b:np.ndarray):
    return np.sum(np.abs(a-b))
@njit(cache=True)
def dijkstra(a:np.ndarray,b:np.ndarray):
    return 0

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

DIRS = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])
def is_connected(c:np.ndarray,positions:np.ndarray):
    """
    Checks if there is at least one cube in some direction
    """
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

def analyze_first_and_second_mistakes(udp):
    cubes = ProgrammableCubes(udp.final_cube_positions)
    ti = udp.initial_cube_types
    ci = udp.final_cube_positions
    ct = udp.target_cube_positions
    tt = udp.target_cube_types
    types = np.arange(np.max(ti)+1)
    wti = have_wrong_type(cubes.cube_position,ti,ct,tt)
    wpi, epi = get_wrong_cube_ids(cubes.cube_position,ct)
    if len(wti) == 0:
        wti = np.array([], dtype=int)
    if len(epi) == 0:
        epi = np.array([], dtype=int)
    if len(wpi) == 0:
        wpi = np.array([], dtype=int)
    return f"mistakes:{len(wpi)}+{len(wti)}"

def is_stuck(cubes,id):
    return get_valid_rots(id,cubes) == []
        

def get_stuck_wrong_cubes(cubes:ProgrammableCubes,wrong_ids:np.ndarray):
    stuck_ids = []
    for id in wrong_ids:
        if is_stuck(cubes,id):
            stuck_ids.append(id)
    return np.array(stuck_ids)

def get_freeroaming_cubes(cubes:ProgrammableCubes,ids:np.ndarray):
    """
    
    """
    freeroaming_ids = []
    for id in ids:
        if len(cubes.cube_neighbours[id]) == 1:
            freeroaming_ids.append(id)
    return np.array(freeroaming_ids)


def get_wrong_ids_and_coords(udp:programmable_cubes_UDP):
    """
    TODO:
    it is like wrong cube ids and have wrong type combined
    """
    ti = udp.initial_cube_types
    ci = udp.final_cube_positions
    ct = udp.target_cube_positions
    tt = udp.target_cube_types
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
    #debug_types_initial = -np.ones(shape=(coordinates_initial.shape[0]),dtype=int)
    wrong_ids_target = np.concatenate([wti,epi])
    coordinates_target = np.concatenate([ct[epi],ct[wti]])
    #debug_types_target = -2*np.ones(shape=(coordinates_target.shape[0]),dtype=int)
    #all_coords =np.concatenate([coordinates_initial,coordinates_target])
    #all_types = np.concatenate([debug_types_initial,debug_types_target])
    
    return wrong_ids_initial,wrong_ids_target

def have_wrong_type_ids_from_udp(udp:programmable_cubes_UDP):
    """
    
    """
    ti = udp.initial_cube_types
    ci = udp.final_cube_positions
    ct = udp.target_cube_positions
    tt = udp.target_cube_types
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
    wrong_ids_target = np.concatenate([wti,epi])
    coordinates_target = np.concatenate([ct[epi],ct[wti]])
    
    return wti,wpi

## Working with chromosome

def end_chromosome(chromosome):
    return np.concatenate([chromosome,np.array([-1])])

def invert_chromosome(udp:programmable_cubes_UDP,chromosome,same_ids = False):
    """
    expect even length 1D array
    """
    assert(chromosome.size % 2 == 0)
    inv_chromosome = []
    # extract moves and ids
    moves = np.array(chromosome[::-2],dtype=int)
    ids = np.array(chromosome[(len(chromosome)-2)::-2],dtype=int)
    # invert rotations
    moves = [inv_rot(moves[i]) for i in range(len(moves))]
    # convert ids
    # i is from target, id is from achieved, id -> i
    # udp is the inverse problem
    udp.fitness(end_chromosome(chromosome))

    achieved_config = udp.final_cube_positions
    avail_ids_achieved_config = np.ones(shape=len(achieved_config),dtype=bool)
    target_config = udp.target_cube_positions
    avail_ids_target_config = np.ones(shape=len(achieved_config),dtype=bool)

    id_conversion = {}
    for i in np.arange(len(target_config)):
        coord = target_config[i]
        id = contains_coord(achieved_config,coord)
        if id != -1:
            id_conversion[int(id)] = int(i)
            avail_ids_achieved_config[id] = False
            avail_ids_target_config[i] = False
    # randomly match the other cubes
    for i in np.arange(len(target_config)):
        if not avail_ids_achieved_config[i]:
            continue
        for j in np.arange(len(target_config)):
            if avail_ids_target_config[j]:
                id_conversion[i] = j
                avail_ids_achieved_config[i] = False
                avail_ids_target_config[j] = False
                break

    mapped_ids = [id_conversion.get(int(_id),-1) for _id in ids]
    if same_ids:
        mapped_ids = ids
    #print(ids)
    #print(moves)
    if any(m == -1 for m in mapped_ids):
        raise RuntimeError(f"Unmapped ids after conversion: {mapped_ids}")

    # 
    inv_pairs = [[mapped_ids[i],moves[i]] for i in range(len(moves))]
    inv_chromosome = np.array(inv_pairs).flatten()
    return inv_chromosome

def invert_chromosome_same_problem(chromosome):
    """
    Does not change the ids but can be used to return to original state of a given problem
    """
    ch = np.array(chromosome, dtype=int).flatten()
    pairs = ch.reshape(-1, 2)[::-1]                  # reversed rows [id,move]
    ids = pairs[:,0].astype(int).tolist()
    moves = [inv_rot(int(m)) for m in pairs[:,1].astype(int).tolist()]

    inv_pairs = np.column_stack([np.array(ids, dtype=int), np.array(moves, dtype=int)])
    return inv_pairs.flatten()