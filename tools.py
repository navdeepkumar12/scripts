import numpy as np
import sys
import pickle
import pm
import datetime 
import time
import matplotlib.pyplot as plt
import os

def check(command):
    try:
        dummy = eval(command)
        return True
    except:
        return False    

def norm(A):
    norm  = [np.sum(a*a) for a in A]
    return np.sqrt(np.sum(norm)), norm 
class progress():
    def __init__(self,max,intervals=100):
        self.max = max
        self.intervals = intervals
        self.progress = 0
        self.iteration = 0
    def forward(self,increment=1):   
        self.iteration += increment
        self.temp = int(self.iteration*self.intervals/self.max)
        if self.temp > self.progress:
            self.progress = self.temp
            fprint('Progress {}/{}'.format(self.progress,self.intervals))

def cprint(txt, color = 'red'):
    if color == 'red':
        print('\033[91m'+'{}'.format(txt)+'\033[0m')
    if color == 'green':
        print('\033[92m'+'{}'.format(txt)+'\033[0m')
    if color == 'blue':
        print('\033[94m'+'{}'.format(txt)+'\033[0m')    


def fprint(txt):
    sys.stdout.flush()
    sys.stdout.write('\r  {}'.format(txt))
        
# class cprint:
    
#     HEADER = '\033[95m'
#     OKBLUE = '\033[94m'
#     OKGREEN = '\033[92m'
#     WARNING = '\033[93m'
#     FAIL = '\033[91m'
#     END = '\033[0m'
#     BOLD = '\033[1m'
#     UNDERLINE = '\033[4m'    

#     def r(txt):
#         print('\033[91m'+'{}'.format(txt)+'\033[0m')
#     def g(txt) :
#         print('\033[92m'+'{}'.format(txt)+'\033[0m')


    
def load_Q(adress):
    Q = pickle.load(open(adress, "rb"))
    print(">tools:load_Q:- Q loaded from {}".format(adress))
    return Q

def dump_Q(file, adress):
    pickle.dump(file, open(adress, 'wb'))
    if os.path.exists(os.getcwd()+ '/'+ adress):
        print('>tools:dump_Q:- File dumped at {}'.format(os.getcwd()+ '/' + adress))
 

# def Q(state,player):
#     if int(player) ==1:
#         return [0,1,1,9,3,5,5,7,7]
#     if int(player) ==2:
#         return [0,2,6,2,6,4,6,6,4]
#     else:
#         #print("{}->Q:- Error: state ={}, player ={}".format(sys.argv[0],state,player))

def initialize_Q():
    Q = {}
    for i in range(3**9):
        temp = np.base_repr(i,base=3)
        temp = "".join(['0'*(9-len(temp)),temp])
        value_temp = np.zeros(9)
        for act, num in enumerate(temp):
            if(num == '0'):
                value_temp[act] = pm.init_reward  
            else:
                value_temp[act] = pm.imr
        Q[temp] = value_temp

    #filename = str(datetime.datetime.now())[:-7] + 'Q'    
    #pickle.dump(Q,open(filename, 'wb'))       
    #print('{}->initialize_Q:- Zero initialization of Q')
    return Q

def string_to_array(state_in_string):
    return(np.array(list(map(int,i))))

def finish_check_array_in_bool():
    finish_check = ['111000000',
                '000111000',
                '000000111',
                '100100100',
                '010010010',
                '001001001',
                '100010001',
                '001010100']

    for i, s in enumerate(finish_check):
        finish_check[i] = list(map(int,s))

    finish_check = np.array(finish_check,dtype=np.int8)
    finish_check = np.array(finish_check,dtype = bool)
    return finish_check


def Toggle(state, a=0): #Toggle state or player
    if a ==1:  #Do nothing if player 1 is inputed
        return state
    state = str(state)
    state = np.array(list(map(int,state)))
    state = np.where(state>0, 3-state, state)
    state = ''.join(list(map(str,state)))
    return state
   

def Result(state):
    #Checks win condition
    #Args:- state,   Return:- winning player and no. of winning condition
    # Return Signal 1,2 player won, 0 no body won, 3 draw, 9 both player won condition
    state = np.array(list(map(int,state)))
    finish_check = finish_check_array_in_bool()
    
    #Draw condition
    if np.sum(state >0) ==9:
        #print("{}->Result:- Match TIE ".format(sys.argv[0]))
        return [3, 0]
    
    W = []
    for i in finish_check:
        if(np.sum(state[i] == 1) ==3):
            W.append(1)
        elif(np.sum(state[i] == 2) == 3):
            W.append(2)

    if len(W)==0:
        return 0,0
    if  np.mean(W) == W[0]:
        #print('{}->Result:- Player {} won in {} conditions'.format(sys.argv[0],W[0], len(W)))
        return W[0], len(W)
    if len(W) > 1:
        #print('{}->Result:- Multiple win condition {}'.format(sys.argv[0],W))
        return 9, len(W)

def array_to_string(array):
    temp = list(map(str,array))
    return ''.join(temp)    

def SAS(state,action,player):
    state = ''.join(list(map(str, state)))# converting to string  
    state1 = state  #just for printing purpose
    result = Result(state)
    #All sanity checks befor change in state
    if int(action) ==9:
        #print('\033[91m'+ 'SAS:- action = 9, game over , returned state {}'.format(state)+'\033[0m')
        return state
    if (state[int(action)] == '0' and len(state) == 9 and abs(2*int(player) -3) ==1 and result[0]==0):
        state = list(state)
        state[int(action)] = str(player)
        #print('SAS:- SAPS = {}-{}{}-{}'.format(state1,action,player,state))
        return ''.join(state)
    #Condition doesnt match to change the state
    else :
        print('\033[91m'+'{}->SAS:- Invalid move, state ={}, action = {},player = {}, Result {}'.format(sys.argv[0],state,action,player,result )+'\033[0m')
        return state
    

def Chance(state):
    state = np.array(list(map(int,state)))  # state in array
    t1 = np.sum(state ==1)
    t2 = np.sum(state ==2)
    if abs(t1-t2) >1:
        #print('\033[91m'+ 'Chance:- Invalid state {}'.format(state)+'\033[0m')
        return 0
    if t1>t2 or (t1 == t2 and np.random.rand(1)<0.5): #if equal wp 1/2
        return 2
    else:
        return 1
    
def Action(state, state_value):
    # Arg:-  state, state_action_value or player, slackness constant for value cutofff, epsilon greedy
    # Return:- chosen action , value, random 0 or greedy 1
    #Sanity Check
    # if len(state)!=9 or 
    # Checking for terminating state
    result = Result(state)
    if result[0] != 0:
        #print('{}->Action:- Terminating state {}'.format(sys.argv[0],state))
        return 9
       
    if type(state_value) in {str, int} : #if  player is inputed instead of state_values.
        if state_value in {1,2,'1','2'}:
            player = state_value
            state_value = Q(state, player)
            #print('{}->Action:- Player {} is inputed, q_value  {}calculated from Q function'.format(sys.argv[0],player,state_value))
        
    state = np.array(list(map(int,state))) #state in array
    valid_action = list(np.where(state==0)[0])
    # Sanilty check
    if len(valid_action) < 1: 
        print("{}->Action:- No valid availabel, state ={}".format(sys.argv[0],state))
        return 9
    #choosing action
    valid_q = np.choose(valid_action,state_value)
    max_q = np.max(valid_q)
    toss = np.random.rand(1)
    if toss < pm.epsilon:
        a = np.random.choice(valid_action)
        q = state_value[a]
        best_a_q =[(i,j) for i,j in zip(valid_action,valid_q) if j >= max_q]
        #print('{}->Action:- Expoloratory action(a,q) ={}'.format(sys.argv[0],(a,q)))
        #print('{}->Action:- Best possible action(a,q) ={}'.format(sys.argv[0],best_a_q))
        return a
    else:
        best_a_q =[(i,j) for i,j in zip(valid_action,valid_q) if j >= max_q*(1- np.sign(max_q)*pm.delta)]
        k = np.random.choice(np.arange(len(best_a_q)))
        a = best_a_q[k][0]
        q = best_a_q[k][1]
        #print('{}->Action:- Action among best values'.format(sys.argv[0]))
        #print('{}->Action:- Valid_action ={}, valid_q ={}'.format(sys.argv[0],valid_action,valid_q))
        #print('{}->Action:- Best possible (a,q) = {} with delta ={}, selected (a,q) = {}'.format(sys.argv[0],best_a_q, pm.delta, (a,q)))
        return a


def Symmetry(state,action):
    #Calculates symmetry
    # Args:- state, action:   Return:- List of symmetric states and actions
    state = list(map(int,state))
    action = int(action)
    #symmetries
    ind = [0,1,2,3,4,5,6,7,8]
    v  = [6,7,8,3,4,5,0,1,2]
    h  = [2,1,0,5,4,3,8,7,6]
    d1 = [0,3,6,1,4,7,2,5,8]
    d2 = [8,5,2,7,4,1,6,3,0]
    r3 = [2,5,8,1,4,7,0,3,6]
    r2 = [8,7,6,5,4,3,2,1,0]
    r1 = [6,3,0,7,4,1,8,5,2]   #rotation 90 clockwise
    
    S = [ind,v,h,d1,d2,r1,r2,r3]
       
    L = []
    for sym in S:
        i = np.choose(sym,state)
        i = ''.join(map(str,i))
        j = sym.index(action)
        j = str(j)
        L.append([i,j])

    return L            

def Path(state,Q):
    state = ''.join(list(map(str,state)))
    # p1 = Chance(s1)
    # if p1 ==0:
    #     return []
    # p2 = Toggle(p1)
    # a1 = Action(s1, Q[s1])
    # s2 = SAS(s1,a1,p1)
    # a2 = Action(s2, Q[s2])
    # s3 = SAS(s2,a2,p2)
    # a3 = Action(s3,Q[s1])
    # s4 = SAS(s3,a3,p1) 
    # a4 = Action(s4,Q[s2])
    # H = [[s1,a1,p1],[s2,a2,p2],[s3,a3,p1],[s4,a4,p2]]
    
    H = []
    player = Chance(state)
    if player in {1,2,'1','2'}:
        for i in range(pm.path_length):
            action = Action(state, Q[state])
            H.append([state,action,player])
            state  = SAS(state, action, player)
            player = Toggle(player)
    #print('{}->Path:- State path generated ={}'.format(sys.argv[0],H))
    return H

def Symmetry_update(state, action, Q, del_q):
    
    L = Symmetry(state,action)
    if pm.symmetry_update == True:
        for i,[state,action] in enumerate(L):
            state = ''.join(list(map(str,state)))
            action = int(action)
            if Q[state][action] == pm.imr:
                print('\033[91m'+'{}->Symmetry_update:- Invalid move, state ={}, action = {}, symmetry={},del_q ={}'.format(sys.argv[0],state,action,i,del_q )+'\033[0m')
        
            Q[state][action] += del_q
    else:
        state = ''.join(list(map(str,state)))
        action = int(action)
        Q[state][action]  += del_q        
    return Q    

        
def Update(H,Q):
    l = len(H)
    H = [H[::2],H[1::2]]
    
    for J in H:
        [s1,a1,p] = J[-1]
        [s2,a2,p] = J[-2]
        r1 = Result(s1)[0]
        r2 = Result(s2)[0]
        if int(p) ==1:
            for i in range(len(J)-1):
                [s2,a2,p] = J[-i-2]
                r2 = Result(s2)[0]

                if r1==0 and r2==0:
                    del_q =  pm.alpha*(Q[s1][a1] - Q[s2][a2])
                    Q = Symmetry_update(s2, a2, Q, del_q)
                    #print('Update:- del_q = {}, [s2,a2,p]=[{},{},{}] '.format(del_q,s2,a2,p))
                if r1 in {1,2,3} and r2 ==0:
                    del_q = pm.alpha*(pm.reward[str(p)+str(r1)]- Q[s2][a2])
                    Q = Symmetry_update(s2, a2, Q, del_q)
                    #print('Update:- del_q = {}, [s2,a2,p]=[{},{},{}] '.format(del_q,s2,a2,p))
                [s1,a1,p,r1] = [s2,a2,p,r2]  
            
        if int(p) ==2:
            for i in range(len(J)-1):
                [s2,a2,p] = J[-i-2]
                r2 = Result(s2)[0]

                if r1==0 and r2==0:
                    del_q = pm.alpha*(Q[Toggle(s1)][a1] - Q[Toggle(s2)][a2])
                    Q = Symmetry_update(Toggle(s2),a2,Q,del_q)
                    #print('Update:- del_q = {}, [s2,a2,p]=[{},{},{}] '.format(del_q,s2,a2,p))
                if r1 in {1,2,3} and r2 ==0:
                    del_q = pm.alpha*(pm.reward[str(p)+str(r1)]- Q[Toggle(s2)][a2])
                    Q = Symmetry_update(Toggle(s2),a2,Q,del_q)
                    #print('Update:- del_q = {}, [s2,a2,p]=[{},{},{}] '.format(del_q,s2,a2,p))

                [s1,a1,p,r1] = [s2,a2,p,r2]     
    return Q , del_q               


def Valid_start_state(state):
    p = Chance(state)
    if p in {1,2,'1','2'} and Result(state)[0]  in {0,'0'}:
        return 1
    # else:
        #print('\033[91m'+ 'Valid_start_state:- Invalid start state {}'.format(state)+'\033[0m')
     

        
        

##OTHER PLOTING FUNCTION
def value_list(Q,fil='Null'):
    Q = list(Q)
    Q = np.array(Q)
    Q = Q.flatten()
    
    if fil != 'Null':
        Q = filter(lambda d: d != fil, Q)
        Q = list(Q)
    
    return Q
def Filter(Q, fil ='Null') :
    if type(Q) ==str:
        Q = load_Q(Q)
        Q = Q.values()
        Q = value_list(Q,fil)
    if type(Q) == dict:
        Q = Q.values()
        Q = value_list(Q,fil) 
    if type(Q) == list or np.ndarray:
        Q = value_list(Q,fil)
    else:
        print('not a address str nor Q dictionary, nor list.array')
    
    return Q  


def index(address):
    
    if os.path.exists(address):
        index = np.loadtxt(address)
        index = int(index) +1
    else :
        index = 10
    np.savetxt(address, [index])

    return index
    
    
            