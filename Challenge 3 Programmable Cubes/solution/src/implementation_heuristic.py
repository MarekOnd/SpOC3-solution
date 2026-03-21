import numpy as np
from numba import njit
from programmable_cubes_UDP import ProgrammableCubes
from programmable_cubes_UDP import programmable_cubes_UDP
from solution.src.misc import *
from solution.src.pathfinding import *
from solution.src.pairing import *

## Final function for finding chromosomes

def find_chromosome(udp:programmable_cubes_UDP,pairing : callable = pair_colours, pathfinding=astar_cubes,budget=np.inf,random_shuffle=False,verbose=False):
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
        wrong_type_ids1,wrong_type_ids2 = have_wrong_type(cubes.cube_position,ti,ct,tt)
        # for id in wrong_type_ids:
        #     move = force_random_move(id,cubes)
        #     if move != -1:
        #         chrom.extend([id,move])

        ## TODO: move wrongly placed ones first to good position, they get built around and are stuck


        # Get places which are wrong
        wi, wt = get_wrong_cube_ids(cubes.cube_position,ct)
        wi = np.concatenate([wrong_type_ids1,wi])
        wt = np.concatenate([wt,wrong_type_ids2])
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
            tmp_chrom,success = pathfinding(cubes,id,end,budget=budget)
            if success:
                chrom.extend(tmp_chrom)
                # Apply to cubes
                tmp_chrom = np.concatenate([tmp_chrom,[-1]])
                cubes.apply_chromosome(tmp_chrom,False)
            successes += (int)(success)
        placed_stat.append(successes)

        # Experiment: let all wrong cubes move 1 time randomly
        # results from few runs: overall improves fitness, maybe increases num of steps at start but helps in the end
        if random_shuffle:
            print("shuffling randomly")
            wi, wt = get_wrong_cube_ids(cubes.cube_position,ct)
            for id in wi:
                move = force_random_move(id,cubes)
                if move != -1:
                     chrom.extend([id,move])

        iter += 1
        mistakes_cnt = len(wi)
        if mistakes_cnt == 0:
            break
        if verbose:
            print(f"mistakes:{len(wi)}[{len(wrong_type_ids1)}] budget status: {len(chrom)/udp.get_nix()*100} % used, escape:{iter/ESCAPE*100}% successes: {successes}")
    return np.array(chrom,dtype=int)

def filter_impossible_moves(udp:programmable_cubes_UDP,chromosome):
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
    return np.array(filtered_chromosome,dtype=int)
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


def find_chromosome_heuristic(udp:programmable_cubes_UDP, budget = np.inf, random_shuffle = False, dont_care_about_color=False,verbose=False):
    """ 
    Idea:
    mark indices at wrong places and with wrong types only once
    try to move the closest
    extend to further apart
    
    """
    # Initialize cubes
    cubes = ProgrammableCubes(udp.final_cube_positions)
    ti = udp.initial_cube_types
    ci = udp.final_cube_positions
    ct = udp.target_cube_positions
    tt = udp.target_cube_types
    types = np.arange(np.max(ti)+1)

    # Initialize outputs
    chrom = []
    successes = 0

    # These are part of the structure but with wrong type => need to be moved to allow others move in
    wti, wtt = have_wrong_type(cubes.cube_position,ti,ct,tt)
    # wti - wrong type ids, from cube_position
    # These are cubes away from the structure and the hollow points in the structure
    wpi, epi = get_wrong_cube_ids(cubes.cube_position,ct)
    # wpi - wrong place ids, from cube_position
    # epi - empty place ids, from target configuration ct

    if len(wti) == 0:
        wti = np.array([], dtype=int)
    if len(epi) == 0:
        epi = np.array([], dtype=int)
    if len(wpi) == 0:
        wpi = np.array([], dtype=int)
    if len(wtt) == 0:
        wtt = np.array([], dtype=int)
    np.random.shuffle(wti)
    np.random.shuffle(wpi)
    np.random.shuffle(epi)
    

    empty_coords = ct[epi] # which coordinates are empty but shouldnt be
    empty_coords_types = tt[epi] # which types should be at those coordinates


    if random_shuffle:
        for id in range(len(ti)):
            move = force_random_move(id,cubes,id)
            if move != -1:
                chrom.extend([id,move])

    

    # Step 1: move wti to epi
    #print(wti,wpi)
    for id in wti:
        empty_coords_with_correct_type = empty_coords[empty_coords_types==ti[id]]
        if dont_care_about_color:
            empty_coords_with_correct_type = np.delete(empty_coords,contains_coord(empty_coords,cubes.cube_position[id]),axis=0)
        start = np.array(cubes.cube_position[id])
        tmp_chrom, end, success = dijkstra_cubes_multiple(cubes,id,empty_coords_with_correct_type,budget)
        if success:
            chrom.extend(tmp_chrom)
            cubes.apply_chromosome(np.concatenate([tmp_chrom,[-1]]),True)
            end_id = contains_coord(empty_coords,end)
            # Remove the coordinate
            empty_coords = np.delete(empty_coords,end_id,0)
            empty_coords_types = np.delete(empty_coords_types,end_id,0)
            # Add the coordinate because it is now empty at the start
            #print(empty_coords.shape,start.shape)
            
            start_id_in_target = contains_coord(ct,start)
            empty_coords = np.append(empty_coords,[start],axis=0)
            empty_coords_types = np.append(empty_coords_types,tt[start_id_in_target])

            #print(f"{id} to {end}")
            successes += 1


    # Step 2: move wpi to epi (TODO or wti)
    for id in wpi:
        empty_coords_with_correct_type = empty_coords[empty_coords_types==ti[id]]
        #if dont_care_about_color:
        #    empty_coords_with_correct_type = empty_coords

        start = np.array(cubes.cube_position[id])
        tmp_chrom, end, success = dijkstra_cubes_multiple(cubes,id,empty_coords_with_correct_type,budget)
        if success:
            chrom.extend(tmp_chrom)
            cubes.apply_chromosome(np.concatenate([tmp_chrom,[-1]]),True)
            end_id = contains_coord(empty_coords,end)
            empty_coords = np.delete(empty_coords,end_id,0)
            empty_coords_types = np.delete(empty_coords_types,end_id,0)
            #print(f"{id} to {end}")
            successes += 1
    if verbose:
        print(f"mistakes:{len(wpi)}+{len(wti)} budget status: {len(chrom)/udp.get_nix()*100} % used, successes: {successes}")
    
    return np.array(chrom,dtype=int)






### Removal of stuck cubes

def get_relative_bridge_coords(dir):
    basis = np.array([[1,0,0],[0,1,0],[0,0,1],[-1,0,0],[0,-1,0],[0,0,-1]]) # ordered to work with random plane direction
    relative_vectors = np.zeros(shape=(5,3)) # + shaped bridge vectors are saved here
    relative_vectors[0] = basis[dir]
    relative_vectors[1] = basis[dir] + basis[(dir+1)%3]
    relative_vectors[2] = basis[dir] - basis[(dir+1)%3]
    relative_vectors[3] = basis[dir] + basis[(dir+2)%3]
    relative_vectors[4] = basis[dir] - basis[(dir+2)%3]
    return relative_vectors

def get_bridge_coords(cubes:ProgrammableCubes,stuck_id):
    stuck_position = cubes.cube_position[stuck_id]

    neigbour_positions = cubes.cube_position[cubes.cube_neighbours[stuck_id]]
    relative_neigbour_positions = neigbour_positions-stuck_position
    ## Find out which direction to make the bridge in
    # figuring out L shapes now
    # direction in which they dont change at all
    common_zero = np.sum(relative_neigbour_positions == np.array([0,0,0]),axis=0) == len(relative_neigbour_positions)
    direction_of_bridge = 0
    #direction_of_bridge = np.random.randint(1,6)
    try:
        direction_of_bridge = np.where(common_zero)[0][0]
    except:
        pass
    direction_of_bridge = direction_of_bridge*np.random.randint(1,3) # choose direction of bridge wrt the plane
    
    bridge_coords = get_relative_bridge_coords(direction_of_bridge) + stuck_position
    return bridge_coords

def create_bridge(cubes:ProgrammableCubes,stuck_id,bridge_cubes_ids):
    """ 

    !applies the transformation
    """
    chrom = np.array([])
    bridge_coords = get_bridge_coords(cubes,stuck_id)

    #freeroaming = get_freeroaming_cubes(cubes,np.arange(len(cubes.cube_position)))
    #found_bridge_pieces = 
    for i in range(len(bridge_cubes_ids)):
        tmp_chrom, end, success = dijkstra_cubes_multiple(cubes,bridge_cubes_ids[i],bridge_coords)
        #try:# For when there is not enough freeroaming cubes
        #except:
        #    print("ended early, not enough freeroaming cubes")
        #    return chrom
        if success:
            cubes.apply_chromosome(end_chromosome(tmp_chrom),True)
            chrom = np.concatenate([chrom,tmp_chrom])
        else:
            #print("cant move freeroaming cube")
            break
    return chrom

def unstuck_cube_using_bridge(cubes:ProgrammableCubes,stuck_id:int,bridge_cubes_ids:np.ndarray):
    chrom = create_bridge(cubes,stuck_id,bridge_cubes_ids)
    # move = force_random_move(stuck_id,cubes)
    # if move != -1:
    #     chrom = np.concatenate([chrom,[stuck_id,move]])
    # else:
    #     print("Cant move even with bridge")
    return chrom

def run_removal_of_stuck_cubes_on_udp(udp:programmable_cubes_UDP, max_tries = 5, use_also_on_cubes_at_wrong_place = False,verbose=False):
    ci = np.array(udp.final_cube_positions)
    chrom = []
    cubes = ProgrammableCubes(udp.final_cube_positions)
    wrong_type_ids, wrong_place_ids  = have_wrong_type_ids_from_udp(udp)
    if use_also_on_cubes_at_wrong_place: # QUICK TEST
        wrong_type_ids = np.concatenate([wrong_type_ids,wrong_place_ids])
    stuck = get_stuck_wrong_cubes(cubes,wrong_type_ids)
    iter = 0

    cubes_target = ProgrammableCubes(udp.target_cube_positions)
    freeroaming_from_target = get_freeroaming_cubes(cubes_target,np.arange(len(cubes.cube_position)))
    
    while len(stuck) > 0 and iter < max_tries:
        chosen_id = stuck[np.random.randint(0,len(stuck))]
        if verbose:
            print(f"Trying to unstuck {chosen_id} with neigbours {cubes.cube_neighbours[chosen_id]}")
        iter += 1


        has_one_neighbour = get_freeroaming_cubes(cubes,np.arange(len(cubes.cube_position)))
        which_are_at_freeroaming_of_target = np.array([contains_coord(cubes.cube_position,cubes_target.cube_position[freeroaming_from_target[i]]) for i in range(len(freeroaming_from_target))],dtype=int)
        freeroaming, idx_a, idx_b = np.intersect1d(has_one_neighbour, which_are_at_freeroaming_of_target, return_indices=True)

        #freeroaming = np.union1d(freeroaming, wrong_place_ids)
        #np.random.shuffle(freeroaming)

        chrom = np.concatenate([chrom,unstuck_cube_using_bridge(cubes,chosen_id,freeroaming)])

        udp.fitness(end_chromosome(chrom),ci)
        #udp.plot("ensemble",types)
        chrom = np.concatenate([chrom,find_chromosome_heuristic(udp)])
        # reset stuck cubes
        udp.fitness(end_chromosome(chrom),ci)
        wrong_type_ids, wrong_place_ids  = have_wrong_type_ids_from_udp(udp)
        if use_also_on_cubes_at_wrong_place: # QUICK TEST
            wrong_type_ids = np.concatenate([wrong_type_ids,wrong_place_ids])
        stuck = get_stuck_wrong_cubes(cubes,wrong_type_ids)
        if verbose:
            print(f"Still stuck {len(stuck)}")
    
    #print(len(chrom),udp.fitness(end_chromosome(chrom)))
    return chrom



def fill_empty_space_with_type(cubes:ProgrammableCubes,cube_types,coord,type,chrom,recursion=0):
    """
    experimental, does not work
    This method always edits the cubes variable
    
    """
    if recursion > 1:
        return np.array([],dtype=int),False
    local_chrom = np.array([],dtype=int)
    #cubes = ProgrammableCubes(cubes.cube_position)
    same_type_ids = np.where(cube_types == type)[0]
    random.shuffle(same_type_ids) # random order
    for id in same_type_ids:
        # Find cube to move closer
        original = np.array(cubes.cube_position[id])
        tmp_chrom, success = astar_cubes_distance(cubes,id,coord,dist=1)
        local_chrom = np.concatenate([local_chrom,np.array(tmp_chrom)])
        if success:
            # 1) try to move the cube even closer
            tmp_chrom2, success2 = astar_cubes(cubes,id,coord)
            cubes.apply_chromosome(end_chromosome(tmp_chrom),verbose=True)
            if success2:
                print(f"{id} can be moved to target coord")
                local_chrom = np.concatenate([local_chrom,np.array(tmp_chrom2)])
                return local_chrom,True
            # 2) if impossible find another cube to move closer
            if recursion == 1:
                continue
            tmp_chrom2, success2 = fill_empty_space_with_type(cubes,cube_types,coord,type,chrom,int(recursion+1))
            if success2:
                print(f"{id} allows another to move into target coord")
                # move at id back to original position
                back_chrom, success_back = astar_cubes(cubes,id,original)
                print(success_back)
                local_chrom = np.concatenate([local_chrom,np.array(tmp_chrom2),np.array(back_chrom)])
                return local_chrom,True
            cubes.apply_chromosome(end_chromosome(invert_chromosome_same_problem(tmp_chrom)),verbose=True)
        else:
            break
    print("found none")
    return np.array([],dtype=int),False

def fill_empty_space_using_bridge(cubes:ProgrammableCubes,cube_types,coord,type,chrom,recursion=0):
    """
    experimental, not finished
    This method always edits the cubes variable
    
    """
    same_type_ids = np.where(cube_types == type)[0]
    coords = get_relative_bridge_coords(random.randint(0,5)) + coord
    # TODO








# Chaining of algorithms to get to the solution


def apply_find_chromosome_multiple_times(udp:programmable_cubes_UDP,method=find_chromosome,count=1,chromosome = np.array([]),budget=np.inf,random_shuffle:bool = False,verbose=False):
    """ 
    udp has to be initialized
    """
    for i in range(count):
        tmp_chrom = method(udp,budget=budget,random_shuffle=random_shuffle)
        chromosome = np.concatenate([chromosome,tmp_chrom])
        fitness = udp.fitness(np.concatenate([tmp_chrom,np.array([-1])]),udp.final_cube_positions) # applied only the new chromosome
        if verbose:
            print(f"fitness (approximation, underestimate) {fitness}, mistakes: {analyze_first_and_second_mistakes(udp)}")
    return chromosome
