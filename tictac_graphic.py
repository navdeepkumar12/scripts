#import imutils
import cv2
import sys
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import tools as tl
import pm

def location(state_in_string, size = 3):
    if len(state_in_string) != int(size*size):
        tl.cprint('grapchic:location:- state_in_string is of length {}, non-square'.format(len(state_in_string)))
        
    circle_int_location = []
    star_int_location  = []    
    for num, loc in enumerate(state_in_string):
        if loc =='1':
            circle_int_location.append(num)
        if loc == '2':
            star_int_location.append(num)
    return circle_int_location, star_int_location

def location_to_cordinate(player_int_location, size = 3, box = 200):
    coordinate = []
    margin = int(box/2)
    for l in player_int_location:
        x = box*(l%size) + margin     # X coordinate
        y = box*int(np.floor(l/size)) + margin  # Y coordinate 
        coordinate.append((x,y))
        # print('x,y,l)= ({},{},{})'.format(x,y,l))
    return coordinate

def star(output, centre):
    t = 40
    (x,y) = centre
    output = cv2.line(output, (x-t, y-t), (x+t, y+t), (200,0,0), thickness=5, lineType=8, shift=0)
    output = cv2.line(output, (x-t, y+t), (x+t, y-t), (200,0,0), thickness=5, lineType=8, shift=0)
    return output

def circle(output,centre,thickness =2, color = (0,0,255)):
    cv2.circle(output, centre , 40 , (0, 0, 255), thickness)    
    return output

def wline(output, finish_check_state_in_string):
    temp = np.array(map(int,finish_check_state_in_string))
    temp = np.where(temp > 1)
    coordinate = location_to_cordinate(temp)
    (x1,y1) = coordinate[0]
    (x2,y2) = coordinate[2]
    output = cv2.line(output, (x1, y1), (x2, y2), (100,100,0), thickness=5, lineType=8, shift=0)
    

def plot(output, coordinate1, coordinate2)  :
    for centre1 in coordinate1:
        output = circle(output, centre1 )
    for centre2 in coordinate2:
        output = star(output, centre2)
    return output



def grid(state_in_string):
    image = cv2.imread(pm.tictac_board_adress)
    (h,w,d) = image.shape
    #print('shape of image', (h,w,d))
    output = image.copy()

    circle_int_location, star_int_location = location(state_in_string)
    coordinate1 = location_to_cordinate(circle_int_location)
    coordinate2 = location_to_cordinate(star_int_location)
    print('coordinate1 =',coordinate1,'coordinate2 =', coordinate2)
    output = plot(output,coordinate1,coordinate2)    
    #cv2.imshow(state_in_string, output)
    # cv2.imwrite('grid/'+state_in_string+'.png', output)
    return output
#state_in_string = sys.argv[1]
#grid(state_in_string)


##OTHER PLOTING FUNCTION
def dict_to_list(D,fil='Null'):
    if type(D)==dict:
        D = D.values()
        D = list(D)
        D = np.array(D)
        D = D.flatten()
        if fil != 'Null':
            D = filter(lambda d: d != fil, D)
            D = list(D)
        return D
def Q_hist(Q, fil ='Null') :
    if type(Q) ==str:
        Q = tl.load_Q(Q)
    if type(Q) == dict:
        D = dict_to_list(Q,fil) 

    plot = plt.hist(D)
    return plot


# #### --------------------GO -------------------------------##


def check(img,size):
    size = size
    box = int(pm.imsize/size)
    margin = int(box/2)

    for i in range(size):
        x1 = i*box + margin
        y1 = margin
        x2 = i*box + margin
        y2 = pm.imsize-margin
        #(x1, y1), (x2, y2)
        img = cv2.line(img, (x1, y1), (x2, y2), (100,100,0), thickness=10, lineType=8, shift=0)
        img = cv2.line(img, (y1, x1), (y2, x2), (100,100,0), thickness=10, lineType=8, shift=0)

    return img

def board(size):
    img = cv2.imread(pm.go_board_adress)
    img = check(img,size)
    cv2.imwrite(pm.output_adress +'/board'+str(size)+'.png',img )
    return img

def go(state_in_string):
    #sanity check
    if type(state_in_string) == list:
        state_in_string = np.array(state_in_string)
    if type(state_in_string) == np.ndarray:
        state_in_string = state_in_string.flatten()
        state_in_string = ''.join(map(str,map(str,state_in_string)))
    if type(state_in_string) != str:
        tl.cprint('graphic:go:- wrong datatype of string/state')

    size = int(np.ceil((np.sqrt(len(state_in_string)))))
    print('size = {}, len = {}'.format(size,len(state_in_string) ))
    box = int(pm.imsize/size)
    margin = int(box/2)
    radius = int(0.8*margin)

    circle_int_location, star_int_location = location(state_in_string, size=size)
    coordinate1 = location_to_cordinate(circle_int_location,size=size, box=box)
    coordinate2 = location_to_cordinate(star_int_location,size=size, box=box)
    print('coordinate1 =',coordinate1,'coordinate2 =', coordinate2) 
    
    output = board(size)
    for centre1 in coordinate1:
        output = cv2.circle(output, centre1 , radius , (0, 0, 255), -1)  
    for centre2 in coordinate2:
        output = cv2.circle(output, centre2 , radius , (255, 0,0), -1)  

    cv2.imwrite(pm.output_adress +'/'+state_in_string +'.jpg', output)    
    return output

    
    # for i,s in enumerate(state_in_string):
    #     x = 