# File:         8_puzzle.py
# Author:       Jeremy B. Kimball
# Date:         1/25/2023
# Description:  Python script to solve the "8 puzzle" game using breadth first search, depth first search, and a_star search algorithims. 

import numpy as np
from collections import deque # used to implement a double ended queue
import time as t # used to time the execution of the program
from heapdict import heapdict # used to implement a priority queue

# Defining a class to store node info
class Node:
    def __init__(self, state = None, parent = None, move = None, cost = 0, heuristic = 0):
        # state: current state of the puzzle
        self.state = state
        # parent: parent node of the current node
        self.parent = parent
        # move: move used to reach the current node from its parent node
        self.move = move
        # cost: cost to reach the current node from the start node
        self.cost = cost
        # heuristic: estimated cost to reach the goal node
        self.heuristic = heuristic

# Function to move up the blank tile
def move_up(state):
    '''Moves the blank tile up in the inputted state and returns the new state. If the blank tile is in the top row, it returns an array of zeros.
    Input: state - the current state of the puzzle
    Output: moved - the new state of the puzzle
    '''
    moved = np.copy(state) # create a copy of the input state
    blank_row, blank_col = np.where(moved == 0) # find the location of the blank tile
    if blank_row > 0: # if the blank tile is not in the top row
        # move the blank tile up
        moved[blank_row, blank_col] = moved[blank_row - 1, blank_col]
        moved[blank_row - 1, blank_col] = 0
        return moved
    else:
        return np.zeros_like(moved)

# Function to move down the blank tile
def move_down(state):
    '''Moves the blank tile down in the inputted state and returns the new state. If the blank tile is in the bottom row, it returns an array of zeros.
    Input: state - the current state of the puzzle
    Output: moved - the new state of the puzzle
    '''
    moved = np.copy(state)
    blank_row, blank_col = np.where(moved == 0)
    if blank_row < 2: # if the blank tile is not in the bottom row
        # move the blank tile down
        moved[blank_row, blank_col] = moved[blank_row + 1, blank_col]
        moved[blank_row + 1, blank_col] = 0
        return moved
    else:
        return np.zeros_like(moved) # returns an array of zeros for the directions in which it is not possible to move

# Function to move the blank tile right
def move_right(state):
    '''
    Moves the blank tile right in the inputted state and returns the new state. If the blank tile is in the rightmost column, it returns an array of zeros.
    Input: state - the current state of the puzzle
    Output: moved - the new state of the puzzle
    '''
    moved = np.copy(state)
    blank_row, blank_col = np.where(moved == 0)
    if blank_col < 2: # if the blank tile is not in the rightmost column
        # move the blank tile right
        moved[blank_row, blank_col] = moved[blank_row, blank_col + 1]
        moved[blank_row, blank_col + 1] = 0
        return moved
    else:
        return np.zeros_like(moved)

# Function to move the blank tile left
def move_left(state):
    '''
    Moves the blank tile left in the inputted state and returns the new state. If the blank tile is in the leftmost column, it returns an array of zeros.
    Input: state - the current state of the puzzle
    Output: moved - the new state of the puzzle
    '''
    moved = np.copy(state)
    blank_row, blank_col = np.where(moved == 0)
    if blank_col > 0: # if the blank tile is not in the leftmost column
        # move the blank tile left
        moved[blank_row, blank_col] = moved[blank_row, blank_col - 1]
        moved[blank_row, blank_col - 1] = 0
        return moved
    else:
        return np.zeros_like(moved)

# Function to expand the node. For an inputted node object, it returns all of the possible children.
def expand_node(node):
    '''
    Expands the inputted node and returns a list of all of the possible children.
    Input: node - the node to be expanded
    Output: children - a list of all of the possible children of the inputted node
    '''
    children = []
    expansion_cost = node.cost + 1
    current_state = node.state
    # Compute the possible children of the node
    children.append(Node(move_up(current_state), node, "Up", expansion_cost))
    children.append(Node(move_down(current_state), node, "Down", expansion_cost))
    children.append(Node(move_left(current_state), node, "Left", expansion_cost))
    children.append(Node(move_right(current_state), node, "Right", expansion_cost))
    children = [i for i in children if not(np.array_equiv(i.state, np.zeros((3, 3))))] # list comprehension to remove the arrays of zeros
    return children

# Function to use the breadth first search to find the path to the goal state
def breadth_first(state, target):
    '''
    Uses the breadth first search algorithm to find the path to the goal state.
    Input: state - the initial state of the puzzle
           target - the goal state of the puzzle
    Output: path - the path to the goal state
            explored - the number of nodes explored
            solution_cost - the cost of the solution
            time - the time it took to find the solution
    '''
    # Initialize start time, initial node, and queues/sets for tracking states
    start_time = t.time() # start the timer
    initial = Node(state)
    path = [] 
    open_queue = deque([initial]) 
    explored = 0
    seen = {tuple(initial.state.flatten())}

    # while there are nodes in the open queue
    while open_queue:
        current = open_queue.popleft() # Get the next node from the open queue
        explored += 1
        
        # Check if the current node is the target state
        if np.array_equal(current.state, target):
            solution_cost = current.cost
            while(current.parent!=None):
                path.insert(0,current.move) # Construct the solution path
                current=current.parent
                time = t.time() - start_time # Calculate the elapsed time
            return solution_cost, explored, time, path # Return the results

        # Expand the children of the current node
        for child in expand_node(current):
            # If the child state has not been seen, add it to the open queue and mark it as seen
            if tuple(child.state.flatten()) not in seen:
                open_queue.append(child)
                seen.add(tuple(child.state.flatten()))

    return None, t.time() - start_time # if the puzzle is not solvable return 'None' & the time taken to expand all nodes
    

# Function to use the depth first search to find the path to the goal state
def depth_first(state, target):
    '''
    Uses the depth first search algorithm to find the path to the goal state.
    Input: state - the initial state of the puzzle
           target - the goal state of the puzzle
    Output: path - the path to the goal state
            explored - the number of nodes explored
            solution_cost - the cost of the solution
            time - the time it took to find the solution
    '''
    # Initialize start time, initial node, and queues/sets for tracking states
    start_time = t.time() # start the timer
    initial = Node(state)
    path = []
    open_queue = deque([initial])
    explored = 0
    seen = {tuple(initial.state.flatten())}

    # while there are nodes in the open queue
    while open_queue:
        current = open_queue.pop() # Get the next node from the open queue
        explored += 1
        # Check if the current node is the target state
        if np.array_equal(current.state, target):
            solution_cost = current.cost
            while(current.parent!=None):
                path.insert(0,current.move) # Construct the solution path
                current=current.parent
                time = t.time() - start_time # Calculate the elapsed time
            return solution_cost, explored, time # Return the results

        # Expand the current node
        for child in expand_node(current):
            # If the child state has not been seen, add it to the open queue and mark it as seen
            if tuple(child.state.flatten()) not in seen:
                open_queue.append(child)
                seen.add(tuple(child.state.flatten()))

    return None, t.time() - start_time # if the puzzle is not solvable return 'None' & the time taken to expand all nodes


# Heuristic function to calculate the nunm of misplaced tiles (for A* search)
def heuristic_num_misplaced_tiles(state, goal_state):
    '''
    Calculates the number of misplaced tiles in the inputted state.
    Input: state - the state to be evaluated
           goal_state - the goal state of the puzzle
    Output: h - the number of misplaced tiles
    '''
    h = np.count_nonzero(np.not_equal(state, goal_state))
    return h

# Heuristic function that sums the distance of each tile from its loaction in the goal state (for A* search)
def heuristic_distance_from_goal(state, goal):
    '''
    Calculates the sum of the distances of each tile from its location in the goal state.
    Input: state - the state to be evaluated
           goal - the goal state of the puzzle
    Output: distance - the sum of the distances of each tile from its location in the goal state
    '''
    state_indices = {val: np.where(state == val) for val in np.unique(state)}
    goal_indices = {val: np.where(goal == val) for val in np.unique(goal)}
    distance = 0
    for val in np.unique(state):
        if val == 0:
            continue
        curr_i, curr_j = state_indices[val]
        goal_i, goal_j = goal_indices[val]
        distance += abs(curr_i - goal_i) + abs(curr_j - goal_j)
    return int(distance)

# Function to use the A* search to find the path to the goal
def a_star(state, target):
    '''
    Uses the A* search algorithm to find the path to the goal state.
    Input: state - the initial state of the puzzle
           target - the goal state of the puzzle
    Output: path - the path to the goal state
            explored - the number of nodes explored
            solution_cost - the cost of the solution
            time - the time it took to find the solution
    '''
   # Initialize start time, initial node, and queues/sets for tracking states
    start_time = t.time() # start the timer
    initial = Node(state)
    path = []
    open_queue = heapdict() # using "heapdict" for the open queue
    open_queue[initial] = initial.heuristic + initial.cost
    seen = {tuple(initial.state.flatten()): initial.cost}
    node_map = {tuple(initial.state.flatten()): initial}
    explored = 0

    #while there are still nodes in the open queue
    while open_queue:
        current = open_queue.popitem()[0] # pop the node with the lowest f-value
        explored += 1
        if np.array_equal(current.state, target):
            # If the current node state is the goal state
            solution_cost = current.cost # the cost of reaching the goal state is the cost of the current node
            while(current.parent!=None):
                path.insert(0,current.move) # Construct the solution path
                current=current.parent
                time = t.time() - start_time # calculate teh elapsed time
            return solution_cost, explored, path, time # return the solution cost, number of states explored, path, and running time

        # Expand the current node
        for child in expand_node(current):
            child_state = tuple(child.state.flatten())
            # if the child's state has already been seen
            if child_state in seen: 
                # If the cost of the child is lower than the previous cost of the same state
                if child.cost < seen[child_state]:
                    if node_map[child_state] in open_queue:
                        del open_queue[node_map[child_state]] # delete old version of state from open queue
                    seen[child_state] = child.cost # update the cost in the 'seen' dictionary
                    open_queue[child] = child.heuristic + child.cost # add child to the open queue
                    node_map[child_state] = child
            else:
                child.heuristic = heuristic_distance_from_goal(child.state, target) # calculate the heuristic value of the child
                open_queue[child] = child.heuristic + child.cost # assign the f-value of the child as the sum of its heuristic and cost
                seen[child_state] = child.cost # add the child's state and cost to the 'seen' dictionary
                node_map[child_state] = child

    return None, t.time() - start_time # if the puzzle is not solvable return 'None' & the time taken to expand all nodes


# Function to generate a random 8-puzzle layout that is solvable
def generate_random_8puzzle():
    state = np.random.permutation(9).reshape((3,3))
    while not isSolvable(state):
        state = np.random.permutation(9).reshape((3,3))
    return state.reshape((3,3))

# functions getInvCount() & isSolvable() adapted from: https://www.geeksforgeeks.org/check-instance-8-puzzle-solvable/
def getInvCount(arr):
    inv_count = 0
    empty_value = 0
    for i in range(0, 9):
        for j in range(i + 1, 9):
            if arr[j] != empty_value and arr[i] != empty_value and arr[i] > arr[j]:
                inv_count += 1
    return inv_count
 
def isSolvable(puzzle) :
    puzzle = puzzle.tolist()
    # Count inversions in given 8 puzzle
    inv_count = getInvCount([j for sub in puzzle for j in sub])
 
    # return true if inversion count is even.
    return (inv_count % 2 == 0)

if __name__ == "__main__":
    # Defining the goal state
    goal_state = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])

    # Defining the initial state (uncomment the state you want to test)

    #initial_state = generate_random_8puzzle()
    #initial_state = np.array([[1, 3, 8], [5, 2, 7], [6, 0, 4]]) # cant solve!!! (impossible)
    initial_state = np.array([[1, 2, 3],[7, 0, 5],[4, 6, 8]]) #intial state from assignment (18 moves)
    #initial_state = np.array([[4, 1, 3],[0, 2, 5],[7, 8, 6]]) #5 moves away
    #initial_state = np.array([[4, 3, 5],[2, 1, 6],[7, 8, 0]]) #10 moves away
    #initial_state = np.array([[3, 0, 5],[4, 2, 1],[7, 8, 6]]) #15 moves away
    #initial_state = np.array([[8, 6, 7],[2, 5, 4],[3, 0, 1]]) #31 moves away. hardest possible layout.

    # Printing the initial state, running each search algorithim, and printing the results
    print('Initial State:', initial_state)
    #print('breadth_first', breadth_first(initial_state, goal_state))
    #print('depth_first', depth_first(initial_state, goal_state))
    print('a_star solution:', a_star(initial_state, goal_state))

