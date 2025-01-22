
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib.patches as mpatches
import copy





class QubitMappingClass():
    
    def __init__(self, numNodes, numQubits, numEPR_threshold, initial_mapping=None):
        self.numNodes = numNodes
        self.numQubits = numQubits
        self.numEPR_threshold = numEPR_threshold

        self.ball_to_box = {}  # ball to box mapping
        self.box_to_ball = {}  # box to ball mapping
        self.EPR_pairs = {}  # EPR pairs mapping
        self.EPR_pool = [f"EPR-{i}" for i in range(numEPR_threshold)]  # Pool of EPR IDs

        if initial_mapping is None:
            initial_mapping = self.generate_random_initial_mapping()

        # Initialize with given mapping
        if initial_mapping is not None:
            for ball, box in initial_mapping.items():
                if (ball > numQubits-1 or box > numNodes - 1):
                    raise Exception("Ball or box out of limit.")
                self.ball_to_box[ball] = box
                self.box_to_ball[box] = ball
        else:   
            raise Exception("Error - initial mapping is None")

             

    def get_box(self, ball):
        if ball not in self.ball_to_box:
            raise Exception(f"No box found for ball {ball}.")
        return self.ball_to_box[ball]

    def get_ball(self, box):
        if box not in self.box_to_ball:
            return self.box_to_ball.get(box, None)
        return self.box_to_ball[box]

    def generate_EPR_pair(self, box1, box2):
        if len(self.EPR_pool) == 0:
            raise Exception("No more EPR IDs available in the pool.")
        
        if (box1 > self.numNodes - 1 or box2 > self.numNodes - 1):
                    raise Exception("Ball or box out of limit.")
        
        
        
        epr_id = self.EPR_pool.pop(0)  # Get the first available ID and remove it from the pool

        # Update the mappings
        self.ball_to_box[epr_id] = [box1, box2]
        self.box_to_ball[box1] = epr_id
        self.box_to_ball[box2] = epr_id
        self.EPR_pairs[epr_id] = [box1, box2]

    def destroy_EPR_pair(self, epr_id):
        if epr_id not in self.EPR_pairs:
            raise Exception(f"No EPR pair with ID {epr_id} exists.")

        # Remove the pair from the mappings
        self.EPR_pairs.pop(epr_id)
        boxes = self.ball_to_box[epr_id]  #might have been updated boxes
        self.ball_to_box.pop(epr_id)
        self.box_to_ball.pop(boxes[0])
        self.box_to_ball.pop(boxes[1])
        # Return the ID to the EPR pool
        self.EPR_pool.append(epr_id)
        self.EPR_pool.sort()  # Keep the pool sorted for predictability

    def query_EPR_pair(self, epr_id):
        if epr_id not in self.EPR_pairs:
            raise Exception(f"No EPR pair with ID {epr_id} exists.")
        # Return the boxes associated with the EPR pair
        # Fetch the current boxes associated with the EPR pair from ball_to_box mapping
        boxes = self.ball_to_box[epr_id]
        # Update the EPR_pairs mapping
        self.EPR_pairs[epr_id] = boxes
        # Return the updated boxes
        return boxes
    
    def generate_random_initial_mapping(self):
        if self.numQubits > self.numNodes:
            raise ValueError("Number of logical qubits cannot be greater than the number of physical qubits.")
        physical_qubits = list(range(self.numNodes))
        random.shuffle(physical_qubits)
        initial_mapping = {i: physical_qubits[i] for i in range(self.numQubits)}
        return initial_mapping
    



