import argparse
import copy
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

plt.ion()

GRID_SIZE = (10, 10)
FRACTION_EMPTY = 0.1
NUM_STEPS = 50
STEPS_PER_PLOT = 1
DELAY_PER_PLOT = 1 / 12

# Similarity threshold is D in the brief.
SIMILARITY_THRESHOLD = 0.2
# Segregation threshold is T_segregation in the brief
SEGREGATION_THRESHOLD = 0.5
# Consensus threshold is T_consensus in the brief
CONSENSUS_THRESHOLD = np.clip(2 * SIMILARITY_THRESHOLD, 0, 1)
# Consensus weight is "alpha" in the brief
CONSENSUS_WEIGHT = 0.1
# empty point represented by value of 0 exactly
EMPTY = 0
# Cluster threshold is T_floodfill in brief
CLUSTER_THRESHOLD = 0.2


PARAMS = dict()
"""Dictionary to store parameters — these may later be overwritten by values read from the command line"""
PARAMS["num_steps"] = NUM_STEPS
PARAMS["grid_size"] = GRID_SIZE
PARAMS["fraction_empty"] = FRACTION_EMPTY
PARAMS["steps_per_plot"] = STEPS_PER_PLOT
PARAMS["delay_per_plot"] = DELAY_PER_PLOT
PARAMS["use_consensus"] = True
PARAMS["use_segregation"] = True
PARAMS["use_floodfill"] = True
PARAMS["similarity_threshold"] = SIMILARITY_THRESHOLD
PARAMS["segregation_threshold"] = SEGREGATION_THRESHOLD
PARAMS["consensus_threshold"] = CONSENSUS_THRESHOLD
PARAMS["consensus_weight"] = CONSENSUS_WEIGHT
PARAMS["cluster_threshold"] = CLUSTER_THRESHOLD
PARAMS["filename"] = None
PARAMS["outfile"] = None

def load_grid_from_file(filename):
    '''
    This function loads a grid of values from a file.

    :param filename: string representing the file path of the grid to be loaded.
    '''
    
    grid = []

    with open(filename, "r") as f:
        for line_count, line in enumerate(f):
            values = line.split(',')
            grid.append([float(val) for val in values])

    return np.array(grid)


def write_grid_to_file(grid, filename):

    np.savetxt(filename, grid, delimiter=",", fmt='%.2f')



def make_random_grid(params):
    """
    This function generates a random grid of values in the range (0, 1) of size grid_size.

    :param params: dictionary containing model parameters.
    """
    row_size, column_size = params["grid_size"]

    #fills a grid of given size with random floats 0 --> 1
    grid = np.random.rand(row_size, column_size)

    total_cells = grid.size

    #the number of required empty cells as an int
    empty_cell_num = int(total_cells * params["fraction_empty"])        

    assinged_empty_cells = 0

    #adds empty cells while there is not enough empty cells
    while(assinged_empty_cells < empty_cell_num):
        #randomly picks a position for empty cell
        empty_index_row = np.random.randint(0, row_size)
        empty_index_column = np.random.randint(0, column_size)

        #the cell cannot be already empty
        if(grid[empty_index_row, empty_index_column] != 0):
            grid[empty_index_row, empty_index_column] = 0
            assinged_empty_cells += 1


    return grid

def plot_floodfill(ax, grid, delay=0.0):
    '''
    This function plots the cluster alongside the intial and updating/final grid with
    cells of the same cluster sharing a colour on the plot

    :param ax: A matplotlib axis object. The plot will be drawn onto this axis.
    :param grid: A grid of cluster values with cells of the same cluster sharing an integer value.
    :param delay: An optional parameter, setting how long the plot will be visible for.
    '''
    #clear the previous state and add title
    ax.clear()
    ax.set_title("Clusters")

    #remove label on axes
    ax.set_xticks([])
    ax.set_yticks([]) 

    #create colour based on the dsicrete tab20
    #empties are set to black
    cmap = plt.cm.tab20.copy()
    cmap.set_under("black")   

    #create 2D array of sqaure cells
    ax.set_aspect("equal")  
    ax.imshow(grid, cmap=cmap, vmin=0.5)

    #delay allows user to see trends
    if delay > 0:
        plt.pause(delay)

def plot_grid(ax, grid, delay=0.0):
    """
    This function should plot the grid, using colour information to show the value of each cell in the grid.
    Empty cells should be set to a colour outside the colour range.

    :param ax: A matplotlib axis object. The plot will be drawn onto this axis.
    :param grid: A grid of values in the range (0, 1).
    :param delay: An optional parameter, setting how long the plot will be visible for.
                Defaults to 0 (plot stays visible until closed).
    """
    plotting_grid = grid.copy()

    #replace all 0's in the grid with -1's
    plotting_grid[plotting_grid == 0] = -1  
    
    #creates a colour map that is the same as coolwarm but for values under the range gives a black colour
    cmap = cm.coolwarm.copy()    
    cmap.set_under("black") 

    #clear previous state and add title
    ax.clear()
    ax.set_title("Updated grid")

    #removes lables on the axes
    ax.set_xticks([])
    ax.set_yticks([])  

    #ensures cells are square
    ax.set_aspect("equal")  

    #outputs the 2D array as an image
    ax.imshow(plotting_grid, cmap=cmap, vmin=0, vmax=1)

    #delay allows user to see trends
    if delay > 0:
        plt.pause(delay)
    else:
        ax.set_title("Final grid")


def consensus_update(grid, params):
    '''
    Compares all neighbours to each other and if neighbours are similar enough
    brings each other closer to a shared mean

    :param grid: A NumPy ndarray representing a grid of values in the range (0, 1).
    :param params: A dictionary storing all model parameters.
    '''

    grid_updated = grid.copy()

    grid_size = params["grid_size"]
    rows = grid_size[0]
    columns = grid_size[1]

    T_consensus = params["consensus_threshold"]
    alpha_consensus = params["consensus_weight"]

    #empty template the same size as the grid, where all interaction changes are accumulated
    weighting_accumulator = np.zeros_like(grid, dtype=float)   

    for i in range(rows):
        for j in range(columns):
            cell_value = grid[i,j]

            #do not apply consensus update to empty cell
            if(cell_value == 0):
                continue

            #values of neighbour cells (the 4 below and to the right to avoid double counting*)
            # *each neighbour must interact once if all eight neigbours are checked then everything is compared twice per iteration,
            #this ensures each neighbour has one comparision where both values are manipulated
            right = grid[i, (j+1) % columns]
            right_down = grid[(i+1) % rows, (j+1) % columns]
            centre_down = grid[(i+1) % rows, j]
            left_down = grid[(i+1) % rows, j-1]

            positions = [right, right_down, centre_down, left_down]
            position_coords = [[i, (j+1) % columns], [(i+1) % rows, (j+1) % columns], [(i+1) % rows, j], [(i+1) % rows, j-1]]
            
            #iterates through the 4 neighbours
            for k in range (4):
                neighbour_value = positions[k]
                
                #checks the nieghbours is a valid option to communicate with
                if (neighbour_value != 0) and (abs(neighbour_value - cell_value) <= T_consensus):
                    r_neighbour, c_neighbour = position_coords[k]

                    #contributions from each neighbour interaction are added to the corresponding position on the accumulative grid
                    weighting_accumulator[i,j] += alpha_consensus*(neighbour_value - cell_value)
                    weighting_accumulator[r_neighbour, c_neighbour] += alpha_consensus*(cell_value - neighbour_value)
            
            
    #represents adding to the original opinion, all the changes due to neighbour interaction at once       
    grid_updated = grid + weighting_accumulator 
    return grid_updated


def segregation_update(grid, params):
    '''
    Compares each cell to it's neighbours to see if similar or different. 
    Determines how similar the cell is to all it's neighbours and below a threshold similarity decides to move.
    All cells that need to move find a random empty cell to move to. 

    :param grid: A NumPy ndarray representing a grid of values in the range (0, 1).
    :param params: A dictionary storing all model parameters.
    '''
    grid_updated = grid.copy()

    grid_size = params["grid_size"]
    rows = grid_size[0]
    columns = grid_size[1]

    D_similarity = params["similarity_threshold"]
    T_segregation = params["segregation_threshold"]

    #array for cells that need to move due to being in a neighbourhood that are too different from
    cells_move_required = []

    #array for valid to move positions
    empty_positions_coords = []

    #finds all empty position in the grid
    for i in range(rows):
        for j in range(columns):
            if grid[i, j] == 0:
                empty_positions_coords.append((i, j))

    for i in range(rows):
        for j in range(columns):
            #variables for the neighbour states
            similar_count = 0
            different_count = 0

            cell_value = grid[i,j]

            #do not apply consensus update to empty cells
            if(cell_value == 0):
                continue

            #values of all 8 neighbour cells
            left = grid[i, j-1]
            left_up = grid[i-1, j-1]
            centre_up = grid[i-1, j]
            right_up = grid[i-1, (j+1) % columns]
            right = grid[i, (j+1) % columns]
            right_down = grid[(i+1) % rows, (j+1) % columns]
            centre_down = grid[(i+1) % rows, j]
            left_down = grid[(i+1) % rows, j-1]

            positions = [left, left_up, centre_up, right_up, right, right_down, centre_down, left_down]

            #test every neighbour to see how similar to them the cell is
            #if not similar enough to neighbourhood add cell to the need to be moved list
            for k in range (8):
                neighbour_state = segregation_cell_similarity_check(cell_value, positions[k], D_similarity)
                if(neighbour_state == 1):
                    different_count += 1
                if(neighbour_state == 2):
                    similar_count += 1
            
            #checks for non-empty neighbours then calculates similarity
            if(similar_count + different_count > 0):
                similarity = similar_count / (similar_count + different_count)
            else:
                #the cell is surronded by empty cells so has no reason to move
                similarity = 1 

            #checks if neighbourhood is too different
            if(similarity <= T_segregation):
                cells_move_required.append((cell_value, i, j))

    #iterates thorugh each cell that must move
    for cell_value, cell_x, cell_y in cells_move_required:
        #no empty cells indicate nothing can change
        if not empty_positions_coords:
            break

        #a random empty cell is chosen and its coords unpacked
        target_index = np.random.randint(len(empty_positions_coords))
        x_target, y_target = empty_positions_coords[target_index]

        #cell and empty cell are swapped
        grid_updated[x_target, y_target] = cell_value
        grid_updated[cell_x, cell_y] = 0
        
        #the empty position array is updated according to the previous swap
        del empty_positions_coords[target_index]
        empty_positions_coords.append((cell_x, cell_y))



    return grid_updated

def segregation_cell_similarity_check(cell_value, position, D_similarity):
    '''
        This function checks if neighbouring cells are empty, and similar. Then returns
        an integer which conveys the information of the determined result.

        :param cell_value: The value of the cell that is to compare neighbours
        :param position: The value of the neighbour cell that is to be compared
        :param D_similarity: A threshold value below which two neighbours are considered similar
    '''
    if (position != 0):
        if(abs(position - cell_value) <= D_similarity):
            return 2    #similar
        return 1        #different
    return 0            #empty

def update(grid, params):
    """Apply one timesteps worth of updates to the grid.

    :param grid: A NumPy ndarray representing a grid of values in the range (0, 1).
    :param params: A dictionary storing all model parameters.
    """

    if params["use_consensus"]:
        grid = consensus_update(grid, params)

    if params["use_segregation"]:
        grid = segregation_update(grid, params)

    if params["use_floodfill"]:
        global clustered_grid, number_clusters
        clustered_grid, number_clusters = floodfill_update(grid, params)

    return grid


def calculate_mean_similarity(grid, params):
    '''
    This function caculates the similarity of all non-empty cells (people) with 
    respect to the surronding non-empty neighbours, giving a mean similarity of the population.

    :param grid: A NumPy ndarray representing a grid of values in the range (0, 1).
    :param params: A dictionary storing all model parameters.
    '''

    grid_size = params["grid_size"]
    rows = grid_size[0]
    columns = grid_size[1]

    D_similarity = params["similarity_threshold"]

    total_similarity = 0
    non_empty_cells = 0

    for i in range(rows):
        for j in range(columns):
            similar_count = 0
            different_count = 0

            cell_value = grid[i,j]

            #do not apply consensus update to empty cell
            if(cell_value == 0):
                continue

            non_empty_cells += 1    
            
            #values of neighbour cells
            left = grid[i, j-1]
            left_up = grid[i-1, j-1]
            centre_up = grid[i-1, j]
            right_up = grid[i-1, (j+1) % columns]
            right = grid[i, (j+1) % columns]
            right_down = grid[(i+1) % rows, (j+1) % columns]
            centre_down = grid[(i+1) % rows, j]
            left_down = grid[(i+1) % rows, j-1]

            positions = [left, left_up, centre_up, right_up, right, right_down, centre_down, left_down]

            #checks each neighbour to see if similar or different
            for k in range (8):
                neighbour_state = segregation_cell_similarity_check(cell_value, positions[k], D_similarity)

                if(neighbour_state == 1):
                    different_count += 1
                elif(neighbour_state == 2):
                    similar_count += 1
            
            #append the similarity of this cell to the total similarity if any non-empty cells exist
            #if surronded by empty cells similarity is 0 but this is the same as not appending
            if(similar_count + different_count > 0):
                total_similarity += similar_count / (similar_count + different_count)

    #if all cells are empty then return a similarity of zero to avoid division by zero
    if(non_empty_cells == 0):
        return 0.0 
    
    #calculation of mean only taking into account only people (non-empty cells)
    total_similarity = total_similarity / (non_empty_cells)
            

    return total_similarity

def floodfill_update(grid, params):
    '''
    This function iterates through each cell in the grid and then cycles through all connected neighbours
    to find all similar cells that form a cluster. Then the next cell that is not in a cluster is searched to form a cluster.

    :param grid: A NumPy ndarray representing a grid of values in the range (0, 1).
    :param params: A dictionary storing all model parameters.
    '''

    #the cluster grid is simply the grid of opinions thats searched through
    #when values are added to a cluster the cell in this grid is made empty (set to zero)
    #so that cells are not repeatedly added to different clusters or searched if already in a cluster
    #must be global so changes in the nested function (when cells are added to a cluster)
    #can be taken into account when coming out of nested function back into a for loop for example - where a search continues
    global clustered_number_grid
    clustered_number_grid = grid.copy()

    #create a grid where the cluster number is going to be stored in each cell
    output_grid = np.zeros_like(grid)

    grid_size = params["grid_size"]
    rows = grid_size[0]
    columns = grid_size[1]

    #list to hold all the clusters, with each index holding all the co-ordinates
    #of all values within a cluster
    all_clusters = []

    #iterate through each position in the grid and check the cell is not already in a cluster
    for i in range(rows):
        for j in range(columns):
            if(clustered_number_grid[i,j] != 0):
                cell = clustered_number_grid[i, j]
                
                #list containg the x,y co-ordinate for each cell in a cluster
                cluster = cluster_search(cell, i, j, params)

                #add the cluster to the list of all clusters
                all_clusters.append(cluster)


    #two similar variables where the cluster_num represents all clusters including clusters of size 1
    #but cluster_count used for counting includes only custers with size of greater than 1
    cluster_num = 1
    cluster_count = 0

    #iterate through each cluster and for each cluster set each coresponding co-ordinate on the output grid
    #equal to the cluster number the cell is part of
    for cluster in all_clusters:
        cell_num_check = 0

        for x, y in cluster:
            cell_num_check += 1
            output_grid[x,y] = cluster_num

        #must be at least 2 cells to count as cluster
        if(cell_num_check > 1):
            cluster_count +=1
        cluster_num+=1    

    return output_grid, cluster_count


def cluster_search(cell, i, j, params):
    '''
    Used as a part of the flood fill function to repeatedly search neighbours of a given cell and
    return a list of co-ords of cell in the given cluster

    :param cell: The value of the cell being searched
    :param i, j: Indicate the position of the cell that is being searched
    :param params: A dictionary storing all model parameters.
    '''
    
    #needs to be modified so must be declared global
    global clustered_number_grid

    grid_size = params["grid_size"]
    rows = grid_size[0]
    columns = grid_size[1] 

    T_floodfill = params["cluster_threshold"]

    #when any cell is added to a cluster it's cell on the main cluster grid is set to zero 
    #this is so it can't be iterated over repeatedly
    clustered_number_grid[i,j] = 0


    cluster = [(i, j)]

    #values of neighbour cells
    #mod operators used to wrap around edges, very important for a nested function like this
    left = clustered_number_grid[i%rows, (j-1)%columns]
    left_up = clustered_number_grid[(i-1)%rows, (j-1)%columns]
    centre_up = clustered_number_grid[(i-1)%rows, j%columns]
    right_up = clustered_number_grid[(i-1)%rows, (j+1)%columns]
    right = clustered_number_grid[i%rows, (j+1)%columns]
    right_down = clustered_number_grid[(i+1)%rows, (j+1)%columns]
    centre_down = clustered_number_grid[(i+1)%rows, j%columns]
    left_down = clustered_number_grid[(i+1)%rows, (j-1)%columns]

    positions = [left, left_up, centre_up, right_up, right, right_down, centre_down, left_down]
    positions_coords = [[i%rows, (j-1)%columns], [(i-1)%rows, (j-1)%columns], [(i-1)%rows, j%columns], [(i-1)%rows, (j+1)%columns], [i%rows, (j+1)%columns], [(i+1)%rows, (j+1)%columns], [(i+1)%rows, j%columns], [(i+1)%rows, (j-1)%columns]]

    #iterate through each of a cells 8 number and check they are non-empty
    for k in range(8):
        if(positions[k] > 0):
            #check neighbourghing cell is similiar enough
            if(abs(positions[k] - cell) < T_floodfill):
                #call this function again for the neighbour 
                #add the returned cluster of cells to the existing list of cluster of cells
                cluster.extend(cluster_search(positions[k], positions_coords[k][0], positions_coords[k][1], params))

    #the cluster list now contains all the connected similar neighbours spanning from the starting cell
    return cluster







# TEST FUNCTIONS — BELOW #####################################################


def test_make_random_grid():

    PARAMS["grid_size"] = (3, 3)
    PARAMS["fraction_empty"] = 0
    grid = make_random_grid(PARAMS)

    assert isinstance(
        grid, (np.ndarray)
    ), "Error: grid is not numpy array"
    assert np.shape(grid) == PARAMS["grid_size"], "Grid shape doesn't match"

    PARAMS["grid_size"] = (20, 10)
    PARAMS["fraction_empty"] = 0
    grid = make_random_grid(PARAMS)

    assert isinstance(
        grid, (np.ndarray)
    ), "Error: grid is not numpy array"
    assert np.shape(grid) == PARAMS["grid_size"], "Grid shape doesn't match"

    PARAMS["grid_size"] = (10, 10)
    PARAMS["fraction_empty"] = 0
    grid = make_random_grid(PARAMS)

    any_are_zero = np.equal(grid, 0).any()
    assert (
        not any_are_zero
    ), "Grid with fraction_empty of 0 should have no zeros"

    PARAMS["grid_size"] = (10, 10)
    PARAMS["fraction_empty"] = 1
    grid = make_random_grid(PARAMS)

    all_are_zero = np.equal(grid, 0).all()
    assert all_are_zero, "Grid with fraction_empty of 1 should be all zeros"

    PARAMS["grid_size"] = (10, 10)
    PARAMS["fraction_empty"] = 0.5
    grid = make_random_grid(PARAMS)

    zero_count = np.equal(grid, 0).sum()
    expected_zero_count = PARAMS["fraction_empty"] * np.size(grid)
    assert (
        abs(zero_count - expected_zero_count) < 5
    ), f"Grid with fraction_empty = 0.5 should have close to 50 zeros, got zero_count = {zero_count}."


def test_consensus_update():

    shape_3x3 = (3, 3)
    PARAMS["grid_size"] = shape_3x3
    grid = 0.4 * np.ones(shape_3x3)
    updated_grid = consensus_update(grid, PARAMS)
    zero_count = np.equal(updated_grid, 0).sum()

    assert np.array_equal(
        np.shape(updated_grid), shape_3x3
    ), "Consensus update should not change grid size"
    assert np.array_equal(
        grid, updated_grid
    ), "Grid with all values the same should not be changed"
    assert (
        zero_count == 0
    ), "Consensus update should not change number of zero cells"

    PARAMS["consensus_weight"] = 0.5
    PARAMS["grid_size"] = shape_3x3
    grid = np.array([[0.4, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.6]])
    updated_grid = consensus_update(grid, PARAMS)
    zero_count = np.equal(updated_grid, 0).sum()

    assert np.array_equal(
        np.shape(updated_grid), shape_3x3
    ), "Consensus update should not change grid size"
    assert (
        zero_count == 7
    ), "Consensus update should not change number of zero cells"
    assert (
        updated_grid[0, 0] == 0.5
    ), "Cell (0, 0) should equal 0.5 after update with weight of 0.5 — do you have circular boundary conditions?"
    assert (
        updated_grid[2, 2] == 0.5
    ), "Cell (2, 2) should equal 0.5 after update with weight of 0.5 — do you have circular boundary conditions?"


def test_segregation_update():
    shape_3x3 = (3, 3)
    PARAMS["grid_size"] = shape_3x3
    grid = np.array([[0.4, 0.0, 0.0], [0.0, 0.4, 0.0], [0.0, 0.0, 0.0]])
    updated_grid = segregation_update(grid, PARAMS)
    zero_count = np.equal(updated_grid, 0).sum()

    assert np.array_equal(
        np.shape(updated_grid), shape_3x3
    ), "Segregation update should not change grid size"
    assert (
        zero_count == 7
    ), "Segregation update should not change number of zero cells"
    assert (
        updated_grid[0, 0] == 0.4
    ), "Cell (0, 0) should equal 0.4 after segregation update"
    assert (
        updated_grid[1, 1] == 0.4
    ), "Cell (1, 1) should equal 0.4 after segregation update"

    PARAMS["segregation_threshold"] = 1.0
    grid = np.array([[0.1, 0.0, 0.0], [0.0, 0.9, 0.0], [0.0, 0.0, 0.0]])
    updated_grid = segregation_update(grid, PARAMS)
    zero_count = np.equal(updated_grid, 0).sum()

    assert np.array_equal(
        np.shape(updated_grid), shape_3x3
    ), "Segregation update should not change grid size"
    assert (
        zero_count == 7
    ), "Consensus update should not change number of zero cells"
    assert (
        updated_grid[0, 0] != 0.1
    ), "Cell (0, 0) should not equal 0.1 after a segregation update"
    assert updated_grid[1, 1] != 0.9, "Cell (1, 1) should not equal 0.9"

    # You may wish to add additional tests to confirm your segregation update is working.


def test_calculate_mean_similarity():

    grid = np.array([[0.1, 0.1, 0.1],
                     [0.1, 0.1, 0.1],
                     [0.1, 0.1, 0.1]])
    
    mean_similarity = calculate_mean_similarity(grid, PARAMS)
    assert calculate_mean_similarity(grid, PARAMS) == 1.0, f"Mean similarity should be 1.0, but is actually {mean_similarity}"

    grid = np.array([[0.0, 0.1, 0.0],
                     [0.1, 0.1, 0.1],
                     [0.0, 0.1, 0.0]])
    
    mean_similarity = calculate_mean_similarity(grid, PARAMS)
    assert calculate_mean_similarity(grid, PARAMS) == 1.0, f"Mean similarity should be 1.0, but is actually {mean_similarity}"

    PARAMS['similarity_threshold'] = 0.5
    grid = np.array([[0.0, 0.1, 0.0],
                     [0.1, 0.9, 0.1],
                     [0.0, 0.1, 0.0]])
    
    mean_similarity = calculate_mean_similarity(grid, PARAMS)
    assert mean_similarity == 0.5625, f"Mean similarity should be 0.5625, but is actually {mean_similarity}"

    PARAMS['similarity_threshold'] = 0.01
    grid = np.array([[0.0, 0.2, 0.0],
                     [0.3, 0.4, 0.5],
                     [0.0, 0.6, 0.0]])
    
    mean_similarity = calculate_mean_similarity(grid, PARAMS)
    assert mean_similarity == 0.0, f"Mean similarity should be 0.0, but is actually {mean_similarity}"



def run_model_tests():
    test_make_random_grid()
    print("Make random grid tests passed")
    test_consensus_update()
    print("Consensus update tests passed")
    test_segregation_update()
    print("Segregation update tests passed")
    test_calculate_mean_similarity()
    print("Calculate mean similarity tests passed")
    print("All tests passed — well done!")


def run_floodfill_tests():
    shape_3x3 = (3, 3)
    PARAMS["grid_size"] = shape_3x3


    grid = np.array([[0.4, 0.0, 0.0], [0.0, 0.4, 0.0], [0.0, 0.0, 0.0]])

    updated_grid, number_clusters = floodfill_update(grid, PARAMS)
    assert number_clusters == 1, f"Expected 1 clusters larger than 1, but got {number_clusters}"


    grid = np.array([[0.1, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.8]])

    updated_grid, number_clusters = floodfill_update(grid, PARAMS)
    assert number_clusters == 0, f"Expected 0 clusters larger than 1, but got {number_clusters}"

# FUNCTIONS FOR RUNNING CODE — BELOW #########################################


def get_parameters(params):
    parser = argparse.ArgumentParser(
        description=" Individual Programming Project for CPA 2025/26"
    )
    parser.add_argument("--use_consensus", action="store_true")
    parser.add_argument("--use_segregation", action="store_true")
    parser.add_argument("--test_model", action="store_true")
    parser.add_argument("--use_floodfill", action="store_true")
    parser.add_argument("--test_floodfill", action="store_true")
    parser.add_argument("-num_steps", type=int)
    parser.add_argument("-steps_per_plot", type=int)
    parser.add_argument("-delay_per_plot", type=float)
    parser.add_argument("-grid_size", nargs=2, type=int)
    parser.add_argument("-similarity_threshold", type=float)
    parser.add_argument("-segregation_threshold", type=float)
    parser.add_argument("-consensus_threshold", type=float)
    parser.add_argument("-consensus_weight", type=float)
    parser.add_argument("-file", type=str)
    parser.add_argument("-outfile", type=str)

    args = parser.parse_args()

    if args.use_consensus:
        params["use_consensus"] = args.use_consensus

    if args.use_segregation:
        params["use_segregation"] = args.use_segregation

    if args.use_floodfill:
        params["use_floodfill"] = args.use_floodfill

    if args.test_model:
        old_params = copy.deepcopy(params)
        run_model_tests()
        params = old_params

    if args.test_floodfill:
        old_params = copy.deepcopy(params)
        run_floodfill_tests()
        params = old_params

    if args.num_steps:
        params["num_steps"] = args.num_steps

    if args.steps_per_plot:
        params["steps_per_plot"] = args.steps_per_plot

    if args.delay_per_plot:
        params["delay_per_plot"] = args.delay_per_plot

    if args.grid_size:
        params["grid_size"] = args.grid_size

    if args.similarity_threshold:
        if 0 <= args.similarity_threshold <= 1:
            params["similarity_threshold"] = args.similarity_threshold
        else:
            print("Error: similarity threshold must be between 0 and 1")
            exit()

    if args.segregation_threshold:
        if 0 <= args.segregation_threshold <= 1:
            params["segregation_threshold"] = args.segregation_threshold
        else:
            print("Error: segregation threshold must be between 0 and 1")
            exit()

    if args.consensus_threshold:
        if args.consensus_threshold >= 0 and args.consensus_threshold <= 1:
            params["consensus_threshold"] = args.consensus_threshold
        else:
            print("Error: consensus threshold must be between 0 and 1")
            exit()

    if args.consensus_weight:
        if args.consensus_weight >= 0 and args.consensus_weight <= 1:
            params["consensus_weight"] = args.consensus_weight
        else:
            print("Error: consensus weight must be between 0 and 1")
            exit()

    if args.file:
        params["filename"] = args.file

    if args.outfile:
        params["outfile"] = args.outfile

    return params



def main(params):
    # Either load a grid file or generate a grid randomly.
    if params["filename"]:
        grid = load_grid_from_file(params["filename"])
        # Edit the lines below to update the parameter settings.
        grid_size = grid.size
        params["grid_size"] = grid.shape

        zero_num = np.sum(grid == 0)
        params["fraction_empty"] = zero_num/grid_size
        print("Grid of size ", grid.shape, " loaded from file")
    else:
        grid_size = params["grid_size"]
        grid = make_random_grid(params)
        print("Created grid of size", grid_size)

    # Create a figure and plot the initial grid
    fig = plt.figure()
    ax = fig.add_subplot(131)
    plot_grid(ax, grid, 1)
    ax.set_title("Initial grid")

    # Update model repeatedly
    print("Running model updates")
    ax2 = fig.add_subplot(132)

    if(params["use_floodfill"]):
        ax3 = fig.add_subplot(133)


    for i in range(params["num_steps"]):
        grid = update(grid, params)
        # Only show every nth (n = 1 by default) update.
        if i % params["steps_per_plot"] == 0:
            plot_grid(ax2, grid, params["delay_per_plot"])

            if(params["use_floodfill"]):
                plot_floodfill(ax3, clustered_grid, params["delay_per_plot"])
                print(number_clusters)

            print("Simulated upto step: ", i)

    print("Simulation complete after ", i, " steps")

    mean_similarity = calculate_mean_similarity(grid, params)
    print("Mean similarity is", mean_similarity)

    if params["outfile"]:
        write_grid_to_file(grid, params["outfile"])
        print("Output grid written to ", params["outfile"])

    #plot the final grid, with a different title
    plot_grid(ax2, grid, 0)

    #ensure plots stay after program finishes
    plt.ioff()
    plt.show()
    return grid


if __name__ == "__main__":
    params = get_parameters(PARAMS)
    main(params)
