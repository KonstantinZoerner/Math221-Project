import numpy as np
def randQB_FP_auto(A, relerr,b,p,Omg):
    m,n=A.shape()
    Q=np.zeros(m,0)
    B=np.zeros(0,n)
    maxiter=50
    maxiter=np.min(maxiter,np.ceil(np.min(m,n)/3/b))
    l=b*maxiter
    while(p>0):
        G,_=np.linalg.qr(A@Omg)
        Omg,_=np.linalg.qr(A.T@G)
        p=p-1
    G=A@Omg
    H=A.T@G
    E=np.linalg.norm(A,'fro')**2
    E0=E
    threshold=relerr**2*E
    r=1
    flag=False
    for i in range(1,maxiter):
        t=


