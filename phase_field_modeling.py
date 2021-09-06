#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 16:02:59 2021

@author: diauro
"""


import numpy as np 
from math import pi
import matplotlib.pyplot as plt
import numpy as np 
from scipy.fft import fft, ifft
import os


from timeit import default_timer as timer

plt.rcParams['figure.dpi'] = 120
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['figure.figsize'] = 5,5


path = '.../output'






def prepare_fft(n_x,n_y,d_x,d_y):
    nx = n_x
    ny = n_y
    
    
    dx = d_x
    dy = d_y
    
    
    
    s = (nx,ny)
    k2 = np.zeros(s)
    
    nx21 = nx/2 + 1
    ny21 = ny/2 + 1
    
    nx2 = nx + 1
    ny2 = ny + 1
    
    kx = np.zeros(nx2)
    ky = np.zeros(nx2)
    
    delkx = (2* pi)/ (nx * dx)
    delky = (2* pi)/ (ny * dy)
    
    
    
    for i in range(int(nx21)):
        j = i+1
        fk1 = (j-1) * delkx
        kx[i] = fk1
        kx[nx2-i-1] = -fk1 
    
    for j in range(int(ny21)):
        k = j+1
        fk2 = (k-1) * delky
        ky[j] = fk2
        ky[nx2-j-1] = -fk2
    
    for i in range(nx):
        for j in range(ny):
            k2[i,j] =  kx[i]**2 + ky[j]**2
                
      
            
    k4 = k2**2

    return kx,ky,k2,k4   
   



#Micro ch pre function 
# This function initializes the mictrostructure for given average composition modulated with a noise term to account 
#thermal fluctuation in Cahn-Hilliard equation.



def micro_ch_pre(nx,ny,c0,iflag):
    
    
    
    nxny = nx*ny
    noise = 0.02


    if iflag == 1:
        s = (nx,ny)
        con = np.zeros(s) 
        for i in range(nx):
            for j in range(ny):
                con[i,j] = c0 + noise*(0.5-random.random())
                
    else:          
        s = (nxny,1)
        con = np.zeros(s)   
        for i in range(nx):
            for j in range(ny):
                ii = (i-1)*nx+j
                con[ii] = c0 + noise*(0.5-random.random())
    return con



def micro_test(nx,ny,c0):
    
    
    
    nxny = nx*ny
    noise = 0
    c0 = 0
    lam = (2*pi) / nx

    s = (nx,ny)
    con = np.zeros(s) 
    for i in range(nx):
        for j in range(ny):
            
            con[i,j] = np.cos(lam*i)

    return con


#  Calculate energy



def calculate_energ(nx,ny,con,grad_coef):
    energ = 0.0
  
    for i in range(nx-1):
        ip = i+1
        for j in range(ny-1):
            jp = j+1
            energ = energ + con[i,j]**2*(1.0 - con[i,j])**2 + 0.5 * grad_coef*((con[ip,j]-con[i,j])**2+(con[i,jp]-con[i,j])**2)


    return energ




#Free energy ch V1

def free_energy_ch_v1(i,j,con):


    A = 1.0
    
    dfdcon =  A*(2.0 * con[i,j] * (1-con[i,j])**2 - 2.0*con[i,j]**2*(1.0-con[i,j]))
    
    return dfdcon



nx = 64
ny = 64
nxny = nx*ny
dx = 1.0
dy = 1.0


#time integration parameters

nstep = 100000


counter = nstep * nx

dtime = 0.01
ttime = 0.0 
coefA = 1.0



# Material specific Parameters
mobility = 1.0
c0 = 0.40
mob = 1.0
grad_coef = 0.5

# Prepare microstructure

iflag = 1

con = micro_ch_pre(nx, ny, c0,iflag)


# prepare fft 

y = prepare_fft(nx, ny, dx, dy)

k2 = y[2]
k4 = y[3]

con_list = []
energ_list = []
# evolve
tsave=0
s = (nx,ny)
dfdcon = np.zeros(s)
for istep  in range(1,nstep):
    ttime = ttime + dtime
    conk = np.fft.fft2(con)
    
    
# Derivation of free energy

    for i in range(nx):
        counter -=1
        if counter % 50000 == 0:
            print(counter)
        for j in range(ny):
        
            dummy = free_energy_ch_v1(i,j,con)
            dfdcon[i,j] = dummy
     
    dfdconk = np.fft.fft2(dfdcon)
 
        #  Time integration
        
    for i in range(nx):
        for j in range(ny):
            
            numer = dtime*mobility*k2[i,j]*dfdconk[i,j]
            denom = 1.0 +dtime*coefA*mobility*grad_coef*k4[i,j]
            
            conk[i,j] = (conk[i,j]-numer) / denom
    
    
    con = np.real(np.fft.ifft2(conk))
    
    
    
    # print result
    
    
    if counter % 200 == 0:
        
        plt.figure()
        plt.title('frame'+ str(tsave) )
        plt.imshow(con,cmap = 'jet')
        plt.colorbar()
        plt.savefig( os.path.join( path, "frame"  + str(tsave) + ".png" ),  dpi=70 , format="PNG")
        plt.close()

    energy = calculate_energ(nx, ny, con, grad_coef)
    energ_list.append(energy)
    tsave += 1


#%%


list_sum =[]

for i in range(len(con_list)):
    suma = np.sum(con_list[i])
    print(suma)
    suma.tolist()
    list_sum.append(suma)


#%%
x_sum = np.arange(0,len(con_list),1)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(x_sum, list_sum, color='tab:blue')
plt.show()

#%%

test_array = np.array(list_sum)-list_sum[0]

#%%


list_sum = np.array(list_sum)
plt.figure(2)
plt.plot(x_sum,list_sum)
plt.ylim(-0,4)
plt.show()
#%%


plt.figure(1)
plt.imshow(con_list[19998], cmap = 'jet')



#%%
