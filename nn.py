import numpy as np 
import pm
import tools as tl 
from scipy import signal as sg
import builtins
import logging as log
log.basicConfig(level=log.INFO)

class default:
    opt = 'adam'

class pad:
    def dim(m,mode):
        if mode == 'full':
            a,b = m-1,-(m-1)  #x_front,x_back,   p[a:b]=x,  b is negative, a is positive
        if mode == 'valid':
            a,b = 0,0
        if mode == 'same':
            a = int((m-1)/2)
            b = -(m-1-a)  
        return a,b

    def pad1d(x,m,mode='full'):
        x = x.copy()      #filter shape m
        x = np.array(x)
        a,b = pad.dim(m,mode)
        p = np.zeros(x.shape[0]+a-b) # b is negative
        if b==0: b=None
        p[a:b] = x
        return p
    def unpad1d(x,m,mode='full'):
        x = x.copy()
        a,b = pad.dim(m,mode)     #filter shape m
        if b==0: b=None
        x = x[a:b]             # b is negative
        return x
    def pad2d(x,f_shape,mode='full'):
        x = x.copy()
        m,n = f_shape       #filter shape 
        k,l = x.shape
        a,b = pad.dim(m,mode)
        c,d = pad.dim(n,mode)
        p = np.zeros((k+a-b, l+c-d)) # b, d is negative
        if b==0: b=None
        if d==0: d=None    
        p[a:b,c:d] = x
        return p    
    def unpad2d(x,f_shape,mode='full'):
        x = x.copy() 
        m,n = f_shape   #filter shape
        a,b = pad.dim(m,mode)
        c,d = pad.dim(n,mode)
        if b==0: b=None
        if d==0: d=None  
        x =x[a:b,c:d]       # b, d is negative
        return x    
        
class optimizer():
    def __init__(self, eta=0.001,beta1 = 0.9, beta2 = 0.999,epsilon=10**(-8)):
        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon =epsilon
        self.t = 0     #time stamp
        self.m = 0    #first order moment aka momentum
        self.v = 0    #second order moment
        self.dw = None
        self.delta = []
        self.beta1t = 1
        self.beta2t = 1

class vanila(optimizer):
    # Vanila gradient descent, just update by given gradient
    def forward(self,dw):   
        self.t = self.t + 1
        self.dw = dw
        self.delta = self.eta*self.dw
        return self.delta
         

class momentum(optimizer):
    def __init__(self):
        super().__init__()
    def forward(self,dw):
        self.t = self.t + 1
        self.dw = dw
        self.m = self.beta1*self.m + (1-self.beta1)*self.dw
        self.beta1t = self.beta1t*self.beta1      # = self.beta1**t
        self.temp3 = self.m/(1-self.beta1t) # =m, Bias correction, as it was initilized to 0
        self.delta = self.eta*self.temp3
        return self.delta


class adam(optimizer):
    
    def forward(self,dw):
        self.t = self.t + 1
        self.dw = dw.copy()
        self.m = self.beta1*self.m + (1-self.beta1)*self.dw
        self.v = self.beta2*self.v + (1-self.beta2)*(self.dw*self.dw)
        self.beta1t = self.beta1t*self.beta1      # = self.beta1**t
        self.beta2t = self.beta2t*self.beta2      # = self.beta2**t
        self.temp3 = self.m/(1-self.beta1t) # =m, Bias correction, as it was initilized to 0
        self.temp4 = self.v/(1-self.beta2t)  # =v,  Bias correction
        self.delta = self.eta*self.temp3/(np.sqrt(self.temp4)+self.epsilon)
        return self.delta

class adamax(optimizer):
    def __init__ (self):
        super().__init__()
        self.first_time = True
        
    def initilize(self):
        if self.first_time == True:
            self.shape = self.dw.shape
            self.v = np.ones(self.shape) 
            self.first_time == False   

    def forward(self,dw):
        self.t = self.t + 1
        self.dw = dw ; self.initilize()    # takes shape of dw and makes v same shape
        self.m = self.beta1*self.m + (1-self.beta1)*self.dw
        self.v = np.max(np.array([self.beta2*self.v, np.abs(self.dw)]), axis=0) # = max(beta*v,|dw|) 
        self.beta1t = self.beta1t*self.beta1      # = self.beta1**t, for m correction
        self.temp3 = self.m/(1-self.beta1t) # =m, Bias correction, as it was initilized to 0
        self.temp4 = self.v        # =v , Bias correction not required, as it gets first input from real data
        self.delta = self.eta*self.temp3/(self.temp4+self.epsilon)  
        return self.delta


class param():
    def __init__(self):
        self.shape = None
        self.w =  []
        self.name = 'param'

class ones(param):
    def forward(self,shape):
        self.shape = shape    
        self.w = np.ones(self.shape)
        return self.w

class zeros(param):
    def forward(self,shape):
        self.shape = shape
        self.w = np.zeros(self.shape)  
        return self.w

class uniform(param):
    def forward(self,shape):
        self.shape = shape
        self.w = np.random.random(self.shape)
        return self.w

class normal(param):
    def forward(self,shape):
        self.shape = shape
        self.w = eval('np.random.randn'+ str(self.shape))
        return self.w

class he(param):
    def forward(self,shape):
        self.shape = shape 
        #self.scale = np.sqrt(self.shape[0]+np.prod(self.shape[1:]))   # for lin and conv2d, conv3d, ?for conv1d, add1
        self.scale = np.sqrt(np.prod(self.shape))
        self.w = eval('np.random.randn'+ str(self.shape))/self.scale
        print('nn:param:he:- param initilized HE random normal')
        return self.w


class layer():

    def __init__(self, name='layer', shape=None, opt=None, param = 'he', trainable = True):
        self.name, self.shape, self.trainable = name, shape, trainable
        self.x, self.y, self.dy, self.dx , self.dw, self.delta = np.repeat(None,6)
        self.w = np.array([0])
        self.default = {} # To know if arguments are by default value
        self.set_opt(opt)
        # Set param and sanity check
        if type(param) ==str:  #  param init is inputed
            self.param = eval(param+'()')
            if shape != None:  # Dosn't init weights of Relu, cre , etc
                self.init_param()
        if type(param) in {list, np.ndarray}: # set param and sanity check
            if self.shape == None or self.shape == param.shape:
                self.set_param(param)         
            else: tl.cprint('layer:init:- layer shape ={} not matched with param shape {}'.format(self.shape,param.shape))
    
    def set_opt(self,opt):
        if opt == None: opt,self.default['opt'] = default.opt, True  
        if type(opt) == str: self.opt = eval(opt+'()')
        else: self.opt = opt
    
    def set_param(self, w):
        self.w = np.array(w)
        self.shape = self.w.shape
            
    def init_param(self, shape = None, param=None): 
        if param != None:   # changing __init__ param init
            self.param = eval(param+'()')   
        if shape != None: # changing __init__ param shape
            self.shape = shape    
        self.w = self.param.forward(self.shape)
    
    def forward(self,x):
        pass
   
    def backward(self,dy):
        pass
   
    def update(self):
        if self.trainable:
            self.delta = self.opt.forward(self.dw)
            self.w = self.w - self.delta
    
    def unbias(self):
        self.mean = np.mean(self.w)
        self.w = self.w - self.mean
    
    def normalize(self,temp = 1):
        self.std = np.std(self.w)
        if self.std >1:
            log.warning('nn.layer.normalize:- std = {}, layer_shape ={}'.format(self.std, self.shape))
            self.w = temp*self.w/self.std 
            

    # def clip(self,min = pm.param.min, max = pm.param.max):
    #     self.min, self.max  = min, max  
    #     if np.ptp(self.w)> self.max - self.min :
    #         log.warning('ptp = {}, shape = {}'.format(np.ptp(w),self.shape ))
    #     self.w = np.clip(self.w, self.min, self.max)    

class linear(layer):
    def __init__(self,shape=None, opt=None):
        super().__init__(name=self.__class__.__name__, opt=opt,shape=shape)
    def forward(self,x):
        self.x = x.copy()
        self.input_shape = x.shape
        self.reshape = x.ndim >1
        if self.reshape:
            self.x = self.x.reshape([-1])
        self.y = self.x@self.w
        return self.y

    def backward(self, dy):
        self.dy = dy.copy()
        self.dx = self.dy@np.transpose(self.w) 
        if self.reshape:
            self.dx = self.dx.reshape(self.input_shape) 
        self.dw = np.outer(self.x, self.dy)
        return self.dx
    

 
class relu(layer):
    def __init__(self):
        super().__init__(trainable=False,name= self.__class__.__name__)
        
    def forward(self,x):
        self.x = x.copy()
        self.y = np.where(self.x<0,0,self.x)
        return self.y

    def backward(self,dy):
        self.dy = dy.copy()
        self.dx = np.where(self.y<0,0,1)*self.dy
        return self.dx
    

class loss():
    def __init__(self,name=None):
        self.name = self.__class__.__name__
        self.x, self.y, self.dy, self.dx , self.dw, self.delta = np.repeat(None,6)
        self.w = np.array([0])
        self.Y = [] # list of previous loss
    def forward(self,x):
        pass
    def backward(self,dy):
        pass
    def update(self):
        pass
    def loss(self,x,label):
        return self.forward(x,label), self.backward()
    

class mse(loss):
    def __init__(self):
        super().__init__(name= self.__class__.__name__)
    
    def forward(self,x, label):
        self.label = label.copy()  
        self.x = x.copy()
        self.y = np.sum((self.x - self.label)*(self.x - self.label))/2
        self.Y.append(self.y)
        return self.y

    def backward(self):
        self.dx = self.x - self.label         
        return self.dx   

class cre(loss):
    def __init__(self):
        super().__init__(name= self.__class__.__name__)
        
    def forward(self,x, label):
        self.label = label.copy()  
        self.x = x.copy()
        self.label_entropy = -np.log2(self.label+0.0001)*self.label    #0.0001 is added to avoid log(0)
        self.x_entropy = -np.log2(self.x+0.0001)*self.label
        self.y = self.x_entropy  - self.label_entropy
        self.y = np.sum(self.y)   #KL divergence
        self.Y.append(self.y)
        return self.y
    
    def backward(self) :   
        self.dx = -self.label*(1/(self.x+0.0001))       # 0.0001 is added to avoid insanely large value of 1/x     
        return self.dx   


class softmax(layer):
    def __init__(self):
        super().__init__(trainable=False,name= self.__class__.__name__)
    
    def forward(self,x):
        self.x = x.copy()
        self.exp = np.exp(self.x)    
        self.esum = np.sum(self.exp)
        self.y = self.exp/self.esum
        return self.y

    def backward(self, dy):
        self.dy = dy.copy()
        self.avg = np.sum(self.dy*self.y)
        self.dx =  self.y*(self.dy - self.avg)  
        return self.dx 



class sigmoid(layer):
    def __init__(self):
        super().__init__(trainable=False, name= self.__class__.__name__)
    
    def forward(self, x):
        self.x = x.copy()
        self.y = 1/(1+np.exp(-x))
        return self.y

    def backward(self, dy):
        self.dy = dy.copy()
        self.dx = self.dy*self.y*(1-self.y)    
        return self.dx

class add(layer):
    def __init__(self, opt=None):
        super().__init__(name= self.__class__.__name__, opt=opt)
        self.w = 0
    def forward(self,x):
        self.x = x.copy()
        self.y = self.x + self.w
        return self.y
    
    def backward(self, dy): 
        self.dy = dy.copy()
        self.dw = self.dy 
        self.dx = self.dy
        return self.dx  

class hadamard(layer):
    def __init__(self):
        super().__init__(name= self.__class__.__name__)    
    def forward(self,x):
        self.x = x.copy()
        self.y = self.x*self.w      
        return self.y

    def backward(self, dy):
        self.dy = dy
        self.dx = self.dy*self.w
        self.dw =self.dy*self.x
        return self.dx

        
class convolve(layer):
    def __init__(self,mode='valid', shape=None, opt=None):
        self.mode = mode
        super().__init__(name= self.__class__.__name__, opt=opt,shape=shape)  
    def forward(self,x):
        self.x = x.copy()
        self.m = self.shape
        self.X = pad.pad1d(self.x, self.m, self.mode)  #padding
        self.y = sg.correlate(self.X, self.w, mode='valid')     
        return self.y

    def backward(self, dy):
        self.dy = dy.copy()
        self.dX = sg.convolve(self.dy, self.w, mode='full')  #padded grad
        self.dw = sg.correlate(self.X, self.dy, mode='valid')
        self.dx = pad.unpad1d(self.dX, self.m, mode=self.mode) #unpadded grad
        return self.dx

    
class convolve2d(layer): 
    def __init__(self,mode='valid',shape=None, opt=None):
        super().__init__(name= self.__class__.__name__,opt=opt,shape=shape) 
    def forward(self,x):
        self.x = x.copy()
        self.k, self.m, self.n = self.shape # k filter, (m,n) filter shape
        self.X = pad.pad2d(x,(self.m, self.n), self.mode)  #padded x = X
        self.y = np.array([sg.correlate2d(self.X, w, 'valid') for w in self.w]) # correlation   
        return self.y

    def backward(self, dy):
        self.dy = dy.copy()
        self.temp = np.array([sg.convolve2d(dy,w, mode='full') for dy,w in zip(self.dy,self.w)]) # high dim, padded dX
        self.dX = np.sum(self.temp, axis=0)           #padded dX
        self.dw = np.array([sg.correlate2d(self.X, dy, mode='valid') for dy in self.dy]) 
        self.dx = pad.unpad2d(self.dX, (self.m, self.n), self.mode)   #padded dx
        return self.dx
   
    


class convolve3d(layer): 
    def __init__(self,mode = 'valid',shape=None, opt=None):
        super().__init__(name= self.__class__.__name__,opt=opt,shape=shape)         
    def forward(self,x):
        self.x = x.copy()
        #(k,l,m,n) k filter, l input dim, (m,n)filter shape
        self.k, self.l, self.m, self.n = self.shape
        self.X = np.array([pad.pad2d(x,(self.m, self.n), self.mode) for x in self.x]) # X = padded x ,dim=(l,m',n')
        self.Y = np.array([[sg.correlate2d(x2,w2,mode='valid') for x2,w2 in zip(self.X, w1)] for w1 in self.w]) # dim=(k,l,m',n')
        self.y = np.sum(self.Y, axis=1) # dim=(k,m',n')       
        return self.y

    def backward(self, dy):
        self.dy = dy.copy()
        self.dX = np.array([[sg.convolve2d(dy,w1,mode='full') for w1 in w] for dy,w in zip(self.dy,self.w)]) #padded dX, dim(k,l,m',n')
        self.dw = np.array([[sg.correlate2d(X, dy, mode='valid') for X in self.X] for dy in self.dy]) # dim(k,l,m,n)
        self.dX1 = np.sum(self.dX, axis=0)  #padded dx, dim(l,m',n')
        self.dx = np.array([pad.unpad2d(dX1, (self.m, self.n), self.mode) for dX1 in self.dX1])   #dim(l,m,n)
        return self.dx
   

    
 
class sequential():
    def __init__(self,layers = [], loss = cre(),opt=None,opt_force=False):
        self.n, self.layers, self.loss, self.opt, self.opt_force = len(layers), layers, loss, opt, opt_force
        self.set_opt(opt)
        self.name = ['SEQUENTIAL: ']+ [layer.name for layer in self.layers] + ['loss= {}'.format(loss.name)]
        
    def forward(self,x):
        self.x = x.copy()
        self.y = x.copy()
        for layer in self.layers:
            self.y = layer.forward(self.y)
        return self.y    

    def backward(self, dy):
        self.dy = dy.copy()
        self.dx = dy.copy()
        for layer in reversed(self.layers):
            self.dx = layer.backward(self.dx)
        return self.dx    
     
    def update(self):
        for layer in self.layers:
            layer.update()
    
    def set_opt(self,opt =None, force=False):
        self.opt, self.opt_force = opt, force
        if opt == None: return 
        if type(opt) !=str: log.warning('default.opt = {} should be string not {}, otherwise all\
            layer would share same opt class'.format(opt, type(opt)))
        if self.opt_force:  # forcefully set all layers to opt
            for layer in self.layers:
                layer.set_opt(self.opt)    
        else:        # ensure default opt from here and finetuning in layer
            for layer in self.layers:
                try:
                    if layer.default['opt'] == True: layer.set_opt(self.opt)
                except:
                    pass
   
    def fit(self,x,y):
        x = self.forward(x)
        loss = self.loss.forward(x,y)
        dx = self.loss.backward()
        dx = self.backward(dx)
        self.update()
        return loss

    def Fit(self,X,Y): 
        for x,y in zip(X,Y):
            self.fit(x,y)
    # def iter(self,command='name'):
    #     return [eval(str(layer)+'.' + str(command)) for layer in self.layers]

    def set_weights(self,W):
        for layer, w in zip(self.layers,W):
            layer.set_param(w)

    def get_weights(self):
        self.w = [layer.w for layer in self.layers]
        return self.w
    
    def get_activation(self):
        self.act = [layer.y for layer in self.layers]
        return self.act
   
    def gain(self):
        self.x_mean = [np.mean(np.abs(layer.x)) for layer in self.layers]  
        self.x_std  = [np.std(layer.x) for layer in self.layers]
        return [self.mean, self.std]

    def stat(self, data):
        self.stats = [stat(d) for d in data]
        return self.stats

    def stab(self):
        for layer in self.layers:
            layer.clip()
            layer.normalize()        

# class stablize():
#     def __init__(self, min=pm.param.min, max=pm.param.max):
#         self.min , self.max = min, max
#     def clip(self, w):
#         self.w = np.clip(w, self.min, self.a_max)
#         return self.w
#     def unbias(self,w):
#         self.mean = np.mean(w)
#         self.w = w-self.mean   
#         return self.w
#     def std(self,w):
#         self.std = np.std(w)
#         if self.w > 1:
#             self.w = self.w/self.std  
#         return self.w     


def stat(a):
    mean = np.mean(a)
    median = np.median(a)
    std = np.std(a)
    amax = np.max(a)
    amin  = np.min(a)
    ptp = np.ptp(a)
    sample = np.random.choice(a.reshape([-1]), np.min([10,np.size(a)])) 
    #samples min(10,size(a)) random values
    return ['mean', mean,'median', median, 'min', amin, 'max', amax,'std', std,'ptp', ptp]



# class gain:
#     def activaat(iter,'attribute'):
#         mean = [np.mean(n)]
