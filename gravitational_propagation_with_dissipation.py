import numpy as np
import matplotlib.pyplot as plt

# The definition of some wave packets I am gonna use. 

def gaussian(r, r0, sigma, n, P):
    
    return (-2*P*(r - r0)/sigma**2) * np.exp(-((r - r0)/sigma)**(2*n))

def harmonic(r, n, P):
    
    return -P*np.sin(np.pi*n*r)

# Some functions related to my evolution schemes. 

def rk4_step(x, t, f, h, *args):
    k1 = h*f(x, t, *args)
    k2 = h*f(x + 0.5*k1, t + 0.5*h, *args)
    k3 = h*f(x + 0.5*k2, t + 0.5*h, *args)
    k4 = h*f(x + k3, t+h, *args)
    xn = x + (k1 + 2*k2 + 2*k3 + k4)/6
    
    return xn

def rk2_step(x, t, f, h, *args):
    
    k1 = h*f(x, t, *args)
    k2 = h*f(x + 0.5*k1, t + 0.5*h, *args)
    xn = x + k2
    
    return xn

def iterative_crank_nicolson(x, t, f, h, *args):
    
    x1 = x + h*f(x, t, *args)
    x2 = x + 0.5*h*(f(x1, t, *args) + f(x, t, *args))
    x3 = x + 0.5*h*(f(x2, t, *args) + f(x, t, *args))
    
    return x3

# First I write out the expression of source terms with 4th order spatial derivatives
# and Kreiss Oliger dissipation terms. 

def uA(A, B, At, Bt, phi, pi, r):
    
    u = np.zeros(len(A), float)
    
    u[3:-3] = At[3:-3] + dissi/(64*deltax)*(A[6:] - 6*A[5:-1] + 15*A[4:-2] - \
     20*A[3:-3] + 15*A[2:-4] - 6*A[1:-5] + A[:-6])
    
    u[2] = At[2] + dissi/(64*deltax)*(A[5] - 6*A[4] + 15*A[3] - 20*A[2] + 16*A[1] - 6*A[0])
    
    u[1] = At[1] + dissi/(64*deltax)*(A[4] - 6*A[3] + 16*A[2] - 26*A[1] + 15*A[0])
    
    u[0] = At[0] + dissi/(64*deltax)*(2*A[3] - 12*A[2] + 30*A[1] - 20*A[0])
    
    u[-3] = At[-3] + dissi/(64*deltax)*(A[-6] - 6*A[-5] + 15*A[-4] - 20*A[-3] + 14*A[-2])
    
    u[-2] = At[-2] + dissi/(64*deltax)*(A[-5] - 6*A[-4] + 14*A[-3] - 14*A[-2])
    
    return u

def uB(A, B, At, Bt, phi, pi, r):
    
    u = np.zeros(len(A), float)
    
    u[3:-3] = Bt[3:-3] + dissi/(64*deltax)*(B[6:] - 6*B[5:-1] + 15*B[4:-2] - \
     20*B[3:-3] + 15*B[2:-4] - 6*B[1:-5] + B[:-6])
    
    u[2] = Bt[2] + dissi/(64*deltax)*(B[5] - 6*B[4] + 15*B[3] - 20*B[2] + 16*B[1] - 6*B[0])
    
    u[1] = Bt[1] + dissi/(64*deltax)*(B[4] - 6*B[3] + 16*B[2] - 26*B[1] + 15*B[0])
    
    u[0] = Bt[0] + dissi/(64*deltax)*(2*B[3] - 12*B[2] + 30*B[1] - 20*B[0])
    
    u[-3] = Bt[-3] + dissi/(64*deltax)*(B[-6] - 6*B[-5] + 15*B[-4] - 20*B[-3] + 16*B[-2] - 6*B[-1])
    
    u[-2] = Bt[-2] + dissi/(64*deltax)*(B[-5] - 6*B[-4] + 16*B[-3] - 26*B[-2] + 15*B[-1])
    
    u[-1] = Bt[-1] + dissi/(64*deltax)*(2*B[-4] - 12*B[-3] + 30*B[-2] - 20*B[-1])
    
    return u
    
def uAt(A, B, At, Bt, phi, pi, r):
    
    u = np.zeros(len(A), float)
    
    u[2:-2] = (-A[4:] + 16*A[3:-1] - 30*A[2:-2] + 16*A[1:-3] - A[:-4])/(12*deltax**2) + 2*np.pi*(phi[2:-2]**2 - \
     pi[2:-2]**2) + np.pi**2/4*(1 - np.exp(2*A[2:-2]))/np.cos(np.pi*r[2:-2]/2)**2
     
    u[1] = (-A[3] + 16*A[2] - 31*A[1] + 16*A[0])/(12*deltax**2) + 2*np.pi*(phi[1]**2 - pi[1]**2) \
    + np.pi**2/4*(1 - np.exp(2*A[1]))/np.cos(np.pi*r[1]/2)**2
    
    u[0] = (-2*A[2] + 32*A[1] - 30*A[0])/(12*deltax**2) - 2*np.pi * pi[0]**2 + \
    np.pi**2/4*(1 - np.exp(2*A[0]))/np.cos(np.pi*r[0]/2)**2
    
    u[-2] = (-A[-4] + 16*A[-3] - 29*A[-2])/(12*deltax**2) + 2*np.pi*(phi[-2]**2 - pi[-2]**2) \
    + np.pi**2/4*(1 - np.exp(2*A[-2]))/np.cos(np.pi*r[-2]/2)**2
    
    return u

def uBt(A, B, At, Bt, phi, pi, r):
    
    u = np.zeros(len(A), float)
    
    u[2:-2] = (-B[4:] + 16*B[3:-1] - 30*B[2:-2] + 16*B[1:-3] - B[:-4])/(12*deltax**2) + \
    (-B[4:] + 8*B[3:-1] - 8*B[1:-3] + B[:-4])/(12*deltax)*((-B[4:] + 8*B[3:-1] - 8*B[1:-3] + B[:-4])/(12*deltax) \
     + 2*np.pi/np.sin(np.pi*r[2:-2])) - Bt[2:-2]**2 + \
    np.pi**2/2*(1-np.exp(2*A[2:-2]))/np.cos(np.pi*r[2:-2]/2)**2
    
    u[1] = (-B[3] + 16*B[2] - 31*B[1] + 16*B[0])/(12*deltax**2) + \
    (-B[3] + 8*B[2] + B[1] - 8*B[0])/(12*deltax)*((-B[3] + 8*B[2] + B[1] - 8*B[0])/(12*deltax) \
     + 2*np.pi/np.sin(np.pi*r[1])) - Bt[1]**2 + \
    np.pi**2/2*(1-np.exp(2*A[1]))/np.cos(np.pi*r[1]/2)**2
    
    u[0] = (-2*B[2] + 32*B[1] - 30*B[0])/(12*deltax**2) - Bt[0]**2 + \
    np.pi**2/2*(1-np.exp(2*A[0]))/np.cos(np.pi*r[0]/2)**2
    
    u[-2] = (-B[-4] + 16*B[-3] - 31*B[-2] + 16*B[-1])/(12*deltax**2) + \
    (B[-4] - 8*B[-3] - B[-2] + 8*B[-1])/(12*deltax)*((B[-4] - 8*B[-3] - B[-2] + 8*B[-1])/(12*deltax) \
     + 2*np.pi/np.sin(np.pi*r[-2])) - Bt[-2]**2 + np.pi**2/2*(1-np.exp(2*A[-2]))/np.cos(np.pi*r[-2]/2)**2
    
    u[-1] = (-2*B[-3] + 32*B[-2] - 30*B[-1])/(12*deltax**2) + Bt[-1]**2
    
    return u
    
def uphit(A, B, At, Bt, phi, pi, r):
    
    u = np.zeros(len(A), float)
    
    u[3:-3] = dissi/(64*deltax)*(phi[6:] - 6*phi[5:-1] + 15*phi[4:-2] - 20*phi[3:-3] + \
     15*phi[2:-4] - 6*phi[1:-5] + phi[:-6]) + (-pi[5:-1] + 8*pi[4:-2] - 8*pi[2:-4] + pi[1:-5])/(deltax*12)
    
    u[2] = dissi/(64*deltax)*(phi[5] - 6*phi[4] + 15*phi[3] - 20*phi[2] + 14*phi[1]) \
    + (-pi[4] + 8*pi[3] - 8*pi[1] + pi[0])/(12*deltax)
    u[1] = dissi/(64*deltax)*(phi[4] - 6*phi[3] + 14*phi[2] - 14*phi[1]) \
    + (-pi[3] + 8*pi[2] + pi[1] - 8*pi[0])/(12*deltax)
    u[-3] = dissi/(64*deltax)*(phi[-6] - 6*phi[-5] + 15*phi[-4] - 20*phi[-3] + 14*phi[-2]) \
    + (pi[-5] - 8*pi[-4] + 8*pi[-2])/(12*deltax)
    u[-2] = dissi/(64*deltax)*(phi[-5] - 6*phi[-4] + 14*phi[-3] - 14*phi[-2]) + \
    (pi[-4] - 8*pi[-3] + pi[-2])/(12*deltax)
    
    return u

def upit(A, B, At, Bt, phi, pi, r):
    
    u = np.zeros(len(A), float)
    
    u[3:-3] = dissi/(64*deltax)*(pi[6:] - 6*pi[5:-1] + 15*pi[4:-2] - 20*pi[3:-3] + \
     15*pi[2:-4] - 6*pi[1:-5] + pi[:-6]) + phi[3:-3]*(-B[5:-1] + 8*B[4:-2] - 8*B[2:-4] + B[1:-5])/(12*deltax) + \
    (-phi[5:-1] + 8*phi[4:-2] - 8*phi[2:-4] + phi[1:-5])/(12*deltax) \
    + np.pi*phi[3:-3]/np.sin(np.pi*r[3:-3]) - Bt[3:-3]*pi[3:-3]
    
    u[2] = dissi/(64*deltax)*(pi[5] - 6*pi[4] + 15*pi[3] - 20*pi[2] + 16*pi[1] - 6*pi[0]) \
    + phi[2]*(-B[4] + 8*B[3] - 8*B[1] + B[0])/(12*deltax) + (-phi[4] + 8*phi[3] \
         - 8*phi[1])/(12*deltax) + np.pi*phi[2]/np.sin(np.pi*r[2]) - Bt[2]*pi[2]
    
    u[1] = dissi/(64*deltax)*(pi[4] - 6*pi[3] + 16*pi[2] - 26*pi[1] + 15*pi[0]) + \
    phi[1]*(-B[3] + 8*B[2] + B[1] - 8*B[0])/(12*deltax) + (-phi[3] + 8*phi[2] - phi[1])/(12*deltax) \
    + np.pi*phi[1]/np.sin(np.pi*r[1]) - Bt[1]*pi[1]
    
    u[0] = dissi/(64*deltax)*(2*pi[3] - 12*pi[2] + 30*pi[1] - 20*pi[0]) + \
    (-phi[2] + 8*phi[1])/(6*deltax) - Bt[0]*pi[0]
    
    u[-3] = dissi/(64*deltax)*(pi[-6] - 6*pi[-5] + 15*pi[-4] - 20*pi[-3] + 14*pi[-2]) \
    + phi[-3]*(-B[-1] + 8*B[-2] - 8*B[-4] + B[-5])/(12*deltax) + (-phi[-1] + 8*phi[-2] \
         - 8*phi[-4] + phi[-5])/(12*deltax) \
    + np.pi*phi[-3]/np.sin(np.pi*r[-3]) - Bt[-3]*pi[-3]
    
    u[-2] = dissi/(64*deltax)*(pi[-5] - 6*pi[-4] + 14*pi[-3] - 14*pi[-2]) + \
    phi[-2]*(B[-4] - 8*B[-3] - B[-2] + 8*B[-1])/(12*deltax) + \
    (phi[-4] - 8*phi[-3] + phi[-2])/(12*deltax) + np.pi*phi[-2]/np.sin(np.pi*r[-2]) - Bt[-2]*pi[-2]
    
    return u

def u_for_all(x, t, r):
    
    [A, B, At, Bt, phi, pi] = x
    u = np.array([uA(A, B, At, Bt, phi, pi, r), uB(A, B, At, Bt, phi, pi, r), uAt(A, B, At, Bt, phi, pi, r), \
                  uBt(A, B, At, Bt, phi, pi, r), uphit(A, B, At, Bt, phi, pi, r), upit(A, B, At, Bt, phi, pi, r)])
    
    return u

def Ricci_scalar(A, phi, pi, r):
    
    return np.pi**3*np.cos(np.pi*r/2)**2*np.exp(-2*A) * (phi**2 - pi**2) - 1.5*np.pi**2

def initial(A, r, r0, sigma, n, P):
    
    return (1 - np.exp(2*A))*np.pi/2*np.tan(np.pi*r/2) + 4*np.sin(np.pi*r) * gaussian(r, r0, sigma, n, P)**2

# Here I am calculating the initial conditions using RK4. 
# Set up parameters. 
    
P = 0.001
r0 = 0.2 
sigma = 0.05
n = 1
dissi = 10

cfl = 0.8
t = 0

Nx = 300

#Plist = []
#errorlist = []

#Nx = 500

#for i in range(10):

deltax = 1/Nx
deltat = deltax*cfl

r = np.linspace(0.0, 1.0, Nx+1)
phi = gaussian(r, 0.2, 0.05, 1, P) # The gradient of scalar field. 

pi = np.copy(phi) # Partial time derivative of scalar field. 

At = 8*np.pi * np.sin(np.pi*r/2) * np.cos(np.pi*r/2) * phi**2

A = np.zeros(Nx+1, float)

B = np.zeros(Nx+1, float)
Bt = np.zeros(Nx+1, float)

for i in range(Nx-1):

    A[i+1] = rk4_step(A[i], r[i], initial, r[i+1] - r[i], r0, sigma, n, P)

# Then I begin to calculate the evolution of time. 

x = np.array([A, B, At, Bt, phi, pi])

tfinal = 4.0
Nt = int(tfinal//deltat)

Ricci = np.empty([Nt+1, Nx+1], float)
Phi = np.empty([Nt+1, Nx+1], float)

# I only store Ricci scalar and gradient of scalar field for all spacetime points. 
# This is for the imshow. 

Ricci[0, :] = Ricci_scalar(A, phi, pi, r)
Phi[0, :] = phi

for i in range(Nt):
    
    t += deltat
    x = iterative_crank_nicolson(x, t-deltat, u_for_all, deltat, r)
    Ricci[i+1, :] = Ricci_scalar(x[0], x[4], x[5], r)
    Phi[i+1, :] = x[4]
    
[A, B, At, Bt, phi, pi] = x

error = np.sqrt(sum(np.abs((-Bt[4:] + 8*Bt[3:-1] - 8*Bt[1:-3] + Bt[:-4])/(12*deltax) + Bt[2:-2] * \
((-B[4:] + 8*B[3:-1] - 8*B[1:-3] + B[:-4])/(12*deltax) - (-A[4:] + 8*A[3:-1] - \
 8*A[1:-3] + A[:-4])/(12*deltax) + np.pi/(2*np.tan(np.pi*r[2:-2]/2))) - \
 At[2:-2]*((-B[4:] + 8*B[3:-1] - 8*B[1:-3] + B[:-4])/(12*deltax) + np.pi/np.sin(np.pi*r[2:-2])) \
 +4*np.pi*phi[2:-2]*pi[2:-2]))/(Nx-3))

# This part is for computing the relation between error and monitored parameters. 

#    Plist.append(P)
#   errorlist.append(error)
    
#   P += 0.001
    
    
#plt.plot(Plist, errorlist)

#plt.xlabel('scalar wave amplitude', fontsize = 15)
#plt.ylabel('error', fontsize = 15)

#plt.show()

print(error)

#%%

# Plot the propagation in Penrose diagrams. 
fig = plt.figure(dpi = 100, figsize = [36, 12])      
plt.imshow(Phi, origin='lower', extent = [0, 1, 0, tfinal])

plt.xlabel('r', fontsize = 15)
plt.ylabel('t', fontsize = 15)

plt.colorbar()
plt.title(r'$\Phi$', fontsize = 15)

plt.gray()
plt.show()

#%%

# Plot the cross-section. 
plt.plot(r, A)

plt.xlabel('r', fontsize = 15)
plt.ylabel('A(r)', fontsize = 15)

plt.title(f't = {tfinal}', fontsize = 15)

plt.show()
    


























