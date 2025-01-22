
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib.patches as mpatches
import copy

from QPUClass import QPUClass
from DAGClass import DAGClass
from QubitMappingClass import QubitMappingClass
from Constants import Constants





class SystemStateClass():
    
    def __init__(self, G, my_DAG, qubit_mapping):
        self.epsilon = 1
        self.G = G
        self.my_DAG = my_DAG
        self.qm = qubit_mapping
        self.update_frontier()
        self.distance_metric = self.calculate_distance_metric() # this metric decides the moving reward - what actions did make the qubits that should come together closer?
        self.distance_metric_prev = self.distance_metric # keep track of the previous distance to calculate the reward (the difference)
        self.cur_mask = self.calculate_mask()  # initialize the current mask

 
    def is_action_possible(self, link):
        # Now we need to check both nodes involved in the link
        return self.G.nodes[link[0]]['weight'] == 0 and self.G.nodes[link[1]]['weight'] == 0 


    def reduce_cooldowns(self):
        for node in self.G.nodes:
            if self.G.nodes[node]['weight'] > 0:
                self.G.nodes[node]['weight'] -= 1


    def perform_action(self, action, link):
        #print(action)
        performed_score = False #make it true only when you indeed performed a score
        # check the cd and produce error if not score or stop. Remember scores we do not produce error since they happen automatically
        if (action != "stop" and action != "SCORE" and action != "tele-gate" and (not self.is_action_possible(link))) : 
            for node in self.G.nodes:
                print(self.G.nodes[node]['weight'])
            
            # print("Warning! error with valid action selection: ", action)
            raise ValueError(f"Action cannot be performed due to cooldown: {action}")
        if action == "GENERATE":
            self.generate(link)
        elif action == "SWAP":
            self.swap(link)
        elif action == "SCORE":
            performed_score = self.score(link)
            print('*************************WE SCORE!!************************')
            print("state is: ", self.convert_self_to_state_vector())
            self.update_frontier()
        elif action == "tele-gate":
            performed_score = self.tele_gate(link)
            print('*************************WE TELEGATE!!************************')
            print("state is: ", self.convert_self_to_state_vector())
            self.update_frontier()
        elif action == "tele-qubit":
            self.tele_qubit(link)
        elif action == "stop":
            self.stop()
        else:
            raise ValueError(f"Unknown action: {action}")
        return performed_score

    def generate(self, link):
        if self.G.edges[link]['label'] != "quantum":
            raise ValueError("GENERATE can only be performed on quantum links.")
        if (self.qm.get_ball(link[0]) != None or self.qm.get_ball(link[1]) != None):
            raise ValueError("GENERATE can only be performed on empty link qubits.")
        self.G.nodes[link[0]]['weight'] = Constants.COOLDOWN_GENERATE
        self.G.nodes[link[1]]['weight'] = Constants.COOLDOWN_GENERATE
        if random.random() < Constants.ENTANGLEMENT_PROBABILITY:  
            self.qm.generate_EPR_pair(*link)



    def swap(self, link):
        if self.G.edges[link]['label'] == "quantum":
            raise ValueError("SWAP cannot be performed on quantum links.")  
        box1, box2 = link
        ball1 = self.qm.get_ball(box1)
        ball2 = self.qm.get_ball(box2)
        # Function to handle swapping of EPR pairs and normal balls
        def handle_swap(box_from, box_to, ball):
            if ball is None:
                return
            # Check if ball is part of EPR pairs
            if ball in self.qm.EPR_pairs:
                self.qm.ball_to_box[ball].remove(box_from)
                self.qm.ball_to_box[ball].append(box_to)
                temp_boxes = self.qm.ball_to_box[ball]              
                self.qm.EPR_pairs[ball] = temp_boxes       #update the EPR pairs as well

            else:
                self.qm.ball_to_box[ball] = box_to

        # If both boxes are empty, don't perform a swap
        if ball1 is None and ball2 is None:
            return
        
        # Temporary store balls before swapping in box_to_ball mapping
        temp_ball1 = ball1
        temp_ball2 = ball2

        if box1 in self.qm.box_to_ball:
            self.qm.box_to_ball.pop(box1)
        if box2 in self.qm.box_to_ball:
            self.qm.box_to_ball.pop(box2)
        
        # Handle swap of ball1 from box1 to box2
        if temp_ball1 is not None:
            self.qm.box_to_ball[box2] = temp_ball1
            handle_swap(box1, box2, temp_ball1)

        # Handle swap of ball2 from box2 to box1
        if temp_ball2 is not None:
            self.qm.box_to_ball[box1] = temp_ball2
            handle_swap(box2, box1, temp_ball2)

        self.G.nodes[link[0]]['weight'] = Constants.COOLDOWN_SWAP
        self.G.nodes[link[1]]['weight'] = Constants.COOLDOWN_SWAP


    def virtual_swap(self, link):
        box2, other_box = link             #other_box has nothing in it here
        ball2 = self.qm.get_ball(box2)
        if ball2 is None:
            return
        # Function to handle swapping of EPR pairs and normal balls
        def handle_virtual_swap(box_from, box_to, ball):
            # Check if ball is part of EPR pairs
            if ball in self.qm.EPR_pairs:
                self.qm.ball_to_box[ball].remove(box_from)
                self.qm.ball_to_box[ball].append(box_to)
                temp_boxes = self.qm.ball_to_box[ball]              
                self.qm.EPR_pairs[ball] = temp_boxes       #update the EPR pairs as well
            else:
                self.qm.ball_to_box[ball] = box_to

        # Temporary store ball before swapping in box_to_ball mapping
        temp_ball2 = ball2
        if box2 in self.qm.box_to_ball:
            self.qm.box_to_ball.pop(box2)

        # Handle swap of ball2 from box2 to other_box
        self.qm.box_to_ball[other_box] = temp_ball2
        handle_virtual_swap(box2, other_box, temp_ball2)



    
    def score(self, link):
        max_cd = 0
        performed_score = False # Have scored
        if self.G.edges[link]['label'] == "quantum":
            raise ValueError("SCORE cannot be performed on quantum links.")
        for (ball1, ball2, _) in self.frontier:
            if (ball1, ball2) in [(self.qm.get_ball(link[0]), self.qm.get_ball(link[1])), (self.qm.get_ball(link[1]), self.qm.get_ball(link[0]))]:
                #self.topo_order.remove((ball1, ball2, _))
                self.my_DAG.remove_node((ball1, ball2)) #it will understand to remove the fist layer that appears 
                max_cd = max(self.G.nodes[link[0]]['weight'], self.G.nodes[link[1]]['weight'])
                self.G.nodes[link[0]]['weight'] = max_cd + Constants.COOLDOWN_SCORE
                self.G.nodes[link[1]]['weight'] = max_cd + Constants.COOLDOWN_SCORE # since scores happen automatically
                performed_score = True
                print("-----we scored", ball1, ball2,"----------")
                return performed_score
        if (not performed_score):
            raise ValueError("could not score.")


    def stop(self):
        self.reduce_cooldowns()


    # in the tele_gate action, note that the link referes to a "virtual" link between EPR pairs
    # gets as input the positions of the EPR pair and scores using any pair of neighbors (if possible)
    def tele_gate(self, link):
        flag = False # Have performed tele-gate
        box1, box2 = link
        max_cd = 0
        ball1, ball2 = self.qm.get_ball(box1), self.qm.get_ball(box2)
        if ball1 not in self.qm.EPR_pairs or ball2 not in self.qm.EPR_pairs or ball1 != ball2:
            raise ValueError("tele-gate can only happen between EPR pairs.")

        neighbors_ball1 = set(self.qm.get_ball(neighbor) for neighbor in self.G.neighbors(box1))
        neighbors_ball2 = set(self.qm.get_ball(neighbor) for neighbor in self.G.neighbors(box2))

        for (ball1_frontier, ball2_frontier, _) in self.frontier: 
            if ((ball1_frontier in neighbors_ball1 and ball2_frontier in neighbors_ball2) or (ball1_frontier in neighbors_ball2 and ball2_frontier in neighbors_ball1)):
                #self.topo_order.remove((ball1_frontier, ball2_frontier, _))
                self.my_DAG.remove_node((ball1_frontier, ball2_frontier)) #it will understand to remove the fist layer that appears 
                #print(ball1)
                self.qm.destroy_EPR_pair(ball1)
                max_cd = max(self.G.nodes[self.qm.get_box(ball1_frontier)]['weight'], self.G.nodes[self.qm.get_box(ball2_frontier)]['weight'], self.G.nodes[link[0]]['weight'], self.G.nodes[link[1]]['weight'])
                self.G.nodes[self.qm.get_box(ball1_frontier)]['weight'] = max_cd + Constants.COOLDOWN_TELE_GATE   # since scores are automatic
                self.G.nodes[self.qm.get_box(ball2_frontier)]['weight'] = max_cd + Constants.COOLDOWN_TELE_GATE
                self.G.nodes[link[0]]['weight'] = max_cd + Constants.COOLDOWN_TELE_GATE
                self.G.nodes[link[1]]['weight'] = max_cd + Constants.COOLDOWN_TELE_GATE
                print("-------we telegate", ball1_frontier, ball2_frontier,"----------")
                flag = True
                return flag
        if (not flag):
            raise ValueError("tele-gate could not be performed.")
        return flag            

    
    #needs a link between an EPR particle and a non EPR particle. It teleports the nonEPR qubit to the position that the other half EPR is.
    def tele_qubit(self, link):
        box1, box2 = link
        if self.qm.get_ball(box1) not in self.qm.EPR_pairs and self.qm.get_ball(box2) not in self.qm.EPR_pairs:
            raise ValueError("tele-qubit needs a half of EPR pair.")
        if self.qm.get_ball(box1) in self.qm.EPR_pairs and self.qm.get_ball(box2) not in self.qm.EPR_pairs:
            print("------Telequbit performed in", self.qm.get_ball(box2), "using", self.qm.get_ball(box1), "------------------")
            EPR_box = copy.deepcopy(self.qm.query_EPR_pair(self.qm.get_ball(box1)))
            EPR_box.remove(box1)
            other_box = EPR_box[0] #where is the other EPR half
            if (self.G.nodes[other_box]['weight'] != 0):     
                raise ValueError("tele-qubit cannot be performed due to cooldown")
            self.qm.destroy_EPR_pair(self.qm.get_ball(box1))
            self.virtual_swap((box2, other_box))    # box2 contains the qubit and other box contains the box of the EPR half
            self.G.nodes[other_box]['weight'] = Constants.COOLDOWN_TELE_QUBIT
        elif self.qm.get_ball(box2) in self.qm.EPR_pairs and self.qm.get_ball(box1) not in self.qm.EPR_pairs:
            print("------Telequbit performed in", self.qm.get_ball(box1), "using", self.qm.get_ball(box2), "------------------")
            EPR_box = copy.deepcopy(self.qm.query_EPR_pair(self.qm.get_ball(box2)))
            EPR_box.remove(box2)
            other_box = EPR_box[0] #where is the other EPR half
            if (self.G.nodes[other_box]['weight'] != 0):     
                raise ValueError("tele-qubit cannot be performed due to cooldown")
            ##print((box1, other_box))
            self.qm.destroy_EPR_pair(self.qm.get_ball(box2))
            self.virtual_swap((box1, other_box))
            self.G.nodes[other_box]['weight'] = Constants.COOLDOWN_TELE_QUBIT
        self.G.nodes[link[0]]['weight'] = Constants.COOLDOWN_TELE_QUBIT
        self.G.nodes[link[1]]['weight'] = Constants.COOLDOWN_TELE_QUBIT  #########SINCE YOU KILLED IT, YOU SHOULD PERFORM THE ACTIONS LATER ON?



    def update_frontier(self):
        # nodes with no incoming edges
        nodes_no_predecessors = set(self.my_DAG.DAG.nodes()) - {node for _, adj_list in self.my_DAG.DAG.adjacency() for node in adj_list.keys()}
        # update frontier
        self.frontier = nodes_no_predecessors



    def calculate_mask(self):  #calculates mask - saves the correct true-falls values depending on whether a link/action can be done and provides a vector form with 0s and 1s with the correct order asked by the agent
        mask = []
        for edge in self.G.edges():  #initialize as if every action is possible
            if (self.G.edges[edge]['label'] != "quantum"):
                self.G.edges[edge]['mask_tele_qubit'] = True   
                self.G.edges[edge]['mask_swap'] = True
            else: self.G.edges[edge]['mask_generate'] = True  

        # first check cooldowns
        for edge in self.G.edges():  
            if not self.is_action_possible(edge): #Action cannot be performed due to cooldown        #####HERE ADD DONT GENERATE IF READHED THE THRESHOLD
                if (self.G.edges[edge]['label'] != "quantum"):
                    self.G.edges[edge]['mask_tele_qubit'] = False   
                    self.G.edges[edge]['mask_swap'] = False
                else: self.G.edges[edge]['mask_generate'] = False  

        # now check link by link whether swap is available. The following implement a soft constraint to boost the efficiency of learning.
            # specifically, do not swap if both of the qubits are None (do not swap empty boxes)
        for edge in self.G.edges():  
            #Check: do we have empty boxes? then do not swap
            if (self.G.edges[edge]['label'] != "quantum"):
                if (self.qm.get_ball(edge[0]) == None and self.qm.get_ball(edge[1]) == None):
                    self.G.edges[edge]['mask_swap'] = False 
    

        # now check link by link whether tele-qubit is available (hard constraints)
        for edge in self.G.edges():  
            #Check: telequbit should have EPR's one 1 side, also the other EPR half should not have cooldown positive
            if (self.G.edges[edge]['label'] != "quantum"):
                #check 1 should have half EPR in one side
                if self.qm.get_ball(edge[0]) not in self.qm.EPR_pairs and self.qm.get_ball(edge[1]) not in self.qm.EPR_pairs:  #"tele-qubit needs a half of EPR pair."
                    self.G.edges[edge]['mask_tele_qubit'] = False 
                #check 2 the other EPR half should be also without cooldown  
                if self.qm.get_ball(edge[0]) in self.qm.EPR_pairs and self.qm.get_ball(edge[1]) not in self.qm.EPR_pairs:
                    EPR_box = copy.deepcopy(self.qm.query_EPR_pair(self.qm.get_ball(edge[0])))
                    EPR_box.remove(edge[0])
                    other_box = EPR_box[0] #where is the other EPR half
                    if (self.G.nodes[other_box]['weight'] != 0):     
                        self.G.edges[edge]['mask_tele_qubit'] = False #tele-qubit cannot be performed due to cooldown"
                elif self.qm.get_ball(edge[1]) in self.qm.EPR_pairs and self.qm.get_ball(edge[0]) not in self.qm.EPR_pairs:
                    EPR_box = copy.deepcopy(self.qm.query_EPR_pair(self.qm.get_ball(edge[1])))
                    EPR_box.remove(edge[1])
                    other_box = EPR_box[0] #where is the other EPR half
                    if (self.G.nodes[other_box]['weight'] != 0):     
                        self.G.edges[edge]['mask_tele_qubit'] = False #tele-qubit cannot be performed due to cooldown"
                #check 3 added later - it implements the soft constraint for efficiency and it cannot implement a telequbit to teleport an empty qubit
                if (self.qm.get_ball(edge[0]) == None or self.qm.get_ball(edge[1]) == None):
                    self.G.edges[edge]['mask_tele_qubit'] = False # tele-qubit better not be performed since we teleport empty qubits
            
    
            else: #Check whether generate is possible
                if (self.qm.get_ball(edge[0]) != None or self.qm.get_ball(edge[1]) != None):    #"GENERATE can only be performed on empty link qubits."
                    self.G.edges[edge]['mask_generate'] = False  
                if len(self.qm.EPR_pool) == 0:                                                  #No more EPR IDs available in the pool.
                    self.G.edges[edge]['mask_generate'] = False  

    
        # Generate mask! Now mask as a vector of 0s and 1s should be created
        # First handle 'simple' links
        for edge in self.G.edges(data=True):  # Include edge data in the iteration
            if edge[2]['label'] != "quantum":  # Check if the label is 'simple'
                mask.append(int(edge[2]['mask_swap']))  # Append 'mask_swap' first
                mask.append(int(edge[2]['mask_tele_qubit']))  # Then append 'mask_tele_qubit'

        # Then handle 'quantum' links (or any link that is not 'simple')
        for edge in self.G.edges(data=True):  # Reiterate to maintain the order
            if edge[2]['label'] == "quantum":  # Check if the label is quantum
                mask.append(int(edge[2].get('mask_generate', False)))  # Append 'mask_generate'

                #print(edge, int(edge[2]['mask_generate']))

        mask = [1] + mask # the 1 at the beginning symbolizes the 'stop' action which is always available.
        # Now 'mask' contains the ordered values as requested
        return mask   


    # created and returns a random action as a vector (e.g., {'edge':(1,2), 'action':SWAP })  #note that from our heuristic design we can only implement SWAP, tele-qubit and generate. 
    def get_random_action(self):     
        # Assuming 'mask' is already created as per calculate_mask() function
        # Create a list of indices where mask is 1
        possible_actions_indices = [i for i, x in enumerate(self.cur_mask) if x == 1]
        # Randomly select one index from possible actions
        selected_index = random.choice(possible_actions_indices)
        random_action = self.decode_action_fromNum(selected_index)
        # Now `random_action` contains the randomly selected edge and action type
        return random_action

    def generate_random_actions_debug(self):
        random_action = 0
        possible_actions_indices = [i for i, x in enumerate(self.cur_mask) if x == 1]
        random_action = random.choice(possible_actions_indices)
        return random_action
    


    def decode_action_fromNum(self, action_num):  #decode a number to the correct action using the self.G.edges command and that first we have swap and then tele-qubit - then we have the generate actions
        # Determine the corresponding edge and action by making the same ordering as how the mask was generated
        edge_action_pairs = []
        for edge in self.G.edges(data=True):
            if edge[2]['label'] != "quantum":
                edge_action_pairs.append((edge, 'SWAP'))
                edge_action_pairs.append((edge, 'tele-qubit'))
        for edge in self.G.edges(data=True):  
            if edge[2]['label'] == "quantum":  # Check if the label is quantum
                edge_action_pairs.append((edge, 'GENERATE'))
        #The selected edge and action
        edge_action_pairs = [([], 'stop')] + edge_action_pairs #exacty as how the mask created the action 'stop' - at the beginning
        selected_edge, selected_action = edge_action_pairs[action_num]
        action = {'edge': selected_edge[:2], 'action': selected_action}  # edge is a tuple (u, v), action is a string (we slice the first 2 since the selected edge has also the labels due to data=true argument)
        return action 


    




    # step the emulator given a specific action
    def step_given_action(self, action_num):

        matching_scores = [] # here we will store the edges that were picked with the autocomplete method of scoring (scores and tele-gates automatically done after an action)
        reward = 0
        #self.cur_mask = self.calculate_mask() # assume that the mask has been updated before you come here - at the initilization we have the initial mask
    
        taken_action = self.decode_action_fromNum(action_num)
        cur_state = copy.deepcopy(self)           
        self.perform_action(taken_action['action'], taken_action['edge'])  #make action and change self (state)

        # Fill any scores or tele-gates that can happen immediately after the action of this time slot
        matching_scores = []  #which links were triggered for scores and telegate
        matching_scores,cur_reward = self.fill_matching(matching_scores)   ## Here we auto fill with the scores and tele-gates! The possible scores and tele-gate actually are implemented here automatically!
        reward += cur_reward                   
        
        
        self.distance_metric = self.calculate_distance_metric() # this metric decides the moving reward - what actions did make the qubits that should come together closer?
        dif_score = 0
        if (reward == 0 and action_num != 0): #it did not score and it is not stop
            dif_score = self.distance_metric_prev - self.distance_metric
            reward = dif_score * Constants.DISTANCE_MULT 
        elif (action_num == 0): #we did stop
            reward = Constants.REWARD_STOP
        self.distance_metric_prev = self.distance_metric #the previous for the next one

    
        self.cur_mask = self.calculate_mask()  # SOS UPDATE the mask with the new changes/new state
        
        flagSuccess = False
        if len(self.my_DAG.DAG.nodes) == 0 : 
            reward = Constants.REWARD_EMPTY_DAG
            flagSuccess = True
        #return reward, self, nx.is_empty(self.my_DAG.DAG)
        return reward, self, flagSuccess
        


    


    #It provides a matching with the possible scores and tele-gates that can happen according to the state.
    def fill_matching(self,matching):
        cur_reward = 0
        # Make a copy of current links in the system excluding those labeled "quantum"
        all_links = [link for link in self.G.edges() if self.G.edges[link].get('label') != 'quantum']

        # Iterate over all links for SCORE action - can be done more efficiently by checking the frontier
        for link in all_links:
            box1, box2 = link
            ball1, ball2 = self.qm.get_ball(box1), self.qm.get_ball(box2)

            # Check if the SCORE action can be performed on this link
            # !!we do not check whether action is possible any more we just increase the cd
            for (ball1_frontier, ball2_frontier, _) in self.frontier:
                if (ball1_frontier, ball2_frontier) in [(ball1, ball2), (ball2,ball1)]:
                    # Perform the SCORE action and append link to matching
                    self.perform_action('SCORE', link)      
                    cur_reward += Constants.REWARD_SCORE
                    matching.append(link)

        # Separate loop to iterate over EPR pairs for tele-gate action
        for ball in list(self.qm.EPR_pairs.keys()):
            link = self.qm.query_EPR_pair(ball)   # get the boxes that contain the EPR pair
            box1, box2 = link
            neighbors_ball1 = set( self.qm.get_ball(neighbor) for neighbor in self.G.neighbors(box1) if  self.G.edges[(box1, neighbor)].get('label') != 'quantum')
            neighbors_ball2 = set(self.qm.get_ball(neighbor) for neighbor in self.G.neighbors(box2) if  self.G.edges[(box2, neighbor)].get('label') != 'quantum') #changed here!!!

            # Iterate over the frontier
            for (ball1_frontier, ball2_frontier, _) in self.frontier:
                # Check if the frontier balls are neighbors to the boxes and if action is possible
                # do not check whether the action is possible between the boxes of the frontier's balls
                if ((ball1_frontier in neighbors_ball1 and ball2_frontier in neighbors_ball2) or
                (ball1_frontier in neighbors_ball2 and ball2_frontier in neighbors_ball1)):
                    # Perform the tele-gate action and append link to matching
                    self.perform_action('tele-gate', link) 
                    cur_reward += Constants.REWARD_SCORE
                    matching.append(link)
                    break
        return matching,cur_reward
    






    def calculate_distance_between_balls(self, ball1, ball2, temp_G):
        # Find the boxes corresponding to ball1 and ball2
        box1 = self.qm.get_box(ball1)
        box2 = self.qm.get_box(ball2)
        epr_links_used = []
        # Calculate the shortest path
        try:
            shortest_path = nx.shortest_path(temp_G, source=box1, target=box2, weight='weight')
            path_length = nx.path_weight(temp_G, shortest_path, weight='weight')
            # Check for EPR links in the path
            for i in range(len(shortest_path) - 1):
                if temp_G.edges[( shortest_path[i], shortest_path[i+1]) ]['virtual'] == True:
                    epr_links_used.append((shortest_path[i], shortest_path[i+1]))
        except nx.NetworkXNoPath:
            # In case there is no path between the two nodes
            path_length = float('inf')

        return path_length, epr_links_used


   


    def calculate_distance_metric(self):
        distance_metric = 0  # Reset the distance metric
        # Create a temporary graph for distance calculation
        temp_G = self.G.copy()

        for edge in temp_G.edges():
            temp_G.edges[edge]['weight'] = 1 # every link will count as distance 1
            temp_G.edges[edge]['virtual'] =  False
            if (temp_G.edges[edge]['label'] == "quantum"): 
                temp_G.edges[edge]['weight'] = Constants.DISTANCE_QUANTUM_LINK  # make it harder to traverse quantum links (they require EPR pair generation conseptually)
        

        # Add links for every EPR pair
        for epr_id, (box1, box2) in self.qm.EPR_pairs.items():
            edge = (box1,box2)
            # Add a "virtual" link
            if (box1,box2) not in self.G.edges:
                temp_G.add_edge(box1, box2, weight = Constants.DISTANCE_BETWEEN_EPR, label="virtual", virtual=True)
            elif (temp_G.edges[edge]['label'] == "quantum"):
                temp_G.edges[edge]['virtual'] = True
                temp_G.edges[edge]['weight'] = Constants.DISTANCE_BETWEEN_EPR # from quantum reduce it temporarily to 1 since we have an entanglement there, remember to increase again when this entanglement is used
            
                

        # Iterate over the frontier to calculate distances
        for (ball1, ball2, _) in self.frontier:
            distance, epr_links_used = self.calculate_distance_between_balls(ball1, ball2, temp_G)
            distance_metric += distance
            # Remove used EPR links from temp_G
            for link in epr_links_used:
                
                if link not in self.G.edges:
                    temp_G.remove_edge(*link)
                elif (temp_G.edges[link]['label'] == "quantum"):
                    temp_G.edges[link]['weight'] = Constants.DISTANCE_QUANTUM_LINK # previous entanglement is used so get it back
                    temp_G.edges[link]['virtual'] = False

        return distance_metric

    






    #convert from the actual state class object to the state vector as the RL agent wants it
    def convert_self_to_state_vector(self):
        # Initialize a list with None (or a placeholder) for each possible index
        my_list = [-1] * (self.qm.numNodes) 
        # Populate the list using dictionary keys as indices
        for key, value in self.qm.box_to_ball.items():
            # Instead of EPR-x we now need just the number (i.e., numNodes+x as an index)
            # Check if value is a string
            if isinstance(value, str):
            # Attempt to extract the number part if the format is as expected
                prefix, num_str = value.split('-')
                value = int(num_str) + self.my_DAG.numQubits # Convert the numerical part to an integer (max logical qubit since there does not exist such and after)
            my_list[key] = value
        single_numbers_topo_list = [element for tup in self.my_DAG.topo_order for element in tup]  #break (x,y,z) tuple inside topo_order to x,y,z (x,y qubits and z the layer)
        #the above is needed for breaking into the state space vector
        state_vector = my_list + single_numbers_topo_list
        N = self.qm.numNodes + 3*self.my_DAG.numGates # N is the size of a correct state vector
        if len(state_vector) < N:
            #print("test")
            state_vector.extend([-2] * (N - len(state_vector)))
        return state_vector



    def plot_qubit_mapping(self,pos_G,rew_display, distance_metric_disp, action_disp,topo_disp, frontier_disp):
        G_temp = self.G.copy()
        # Update node labels and color mapping
        labels = {}
        colors = []
        for box in G_temp.nodes:
            weight = G_temp.nodes[box]['weight']  # Get node weight
            if box in self.qm.box_to_ball.keys(): # the box contains a ball
                ball = self.qm.box_to_ball[box]
                if ball in self.qm.EPR_pairs:  # the ball is part of an EPR pair
                    labels[box] = f'{ball} ({weight})'  # Append weight to the label
                    colors.append('cyan')
                else:  # the ball is not part of an EPR pair
                    labels[box] = f'Q-{ball} ({weight})'  # Append weight to the label
                    colors.append('red')
            else:  # the box does not contain a ball
                labels[box] = f'No ({weight})'  # Append weight to the label
                colors.append('green')


        plt.figure(figsize=(10, 8)) 
        # Plot graph
        nx.draw_networkx(G_temp, pos_G, node_color=colors, labels=labels, with_labels=False, font_size=6, node_size=1000)
        nx.draw_networkx_labels(G_temp, pos_G, labels=labels, font_size=9)
        # Show quantum links
        quantum_links = [(u,v) for (u,v,d) in G_temp.edges(data=True) if d['label'] == 'quantum']
        nx.draw_networkx_edges(G_temp, pos_G, edgelist=quantum_links, width=2, alpha=0.5, edge_color='red')

        # Add text at the bottom of the figure
        plt.text(0.5, 1.1, 'reward='+str(rew_display) + ",\n" 'distance_metric='+str(distance_metric_disp) + ",\n" + 'action='+str(action_disp) , ha='center', va='top', transform=plt.gca().transAxes)
        plt.text(0.5, -0.1, 'dag_left='+str(topo_disp) + ",\n"+'frontier='+str(frontier_disp), ha='center', va='bottom', transform=plt.gca().transAxes)
        plt.show()