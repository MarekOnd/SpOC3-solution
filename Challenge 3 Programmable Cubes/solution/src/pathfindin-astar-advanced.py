
import numpy as np
from numba import njit
from heapq import heappush, heappop
from programmable_cubes_UDP import ProgrammableCubes
from programmable_cubes_UDP import programmable_cubes_UDP
from solution.src.misc import *
import random
from solution.src.pathfinding import *


# IDEA: astar but when stuck add also neighbour and surrounding rotations as possible "paths"