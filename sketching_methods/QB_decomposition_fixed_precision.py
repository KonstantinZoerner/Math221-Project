import numpy as np
import torch
#from sketching_methods.sketching import sketch_SRTT
def randQB_FP_auto(A, relerr,b,p,Omg):
    m,n=A.shape
    Q=np.zeros((m,1))
    B=np.zeros((1,n))
    maxiter=50
    #print(m,n)
    maxiter=np.min((maxiter,np.ceil(np.min((m,n))/3/b)))
    l=b*maxiter
    while(p>0):
        G,_=np.linalg.qr(A@Omg)
        Omg,_=np.linalg.qr(A.T@G)
        p=p-1
    G=(A@Omg).type(torch.FloatTensor)
    H=A.T@G
    E=np.linalg.norm(A,'fro')**2
    E0=E
    threshold=relerr**2*E
    r=1
    flag=False
    for i in range(1,int(maxiter)):
        t=B@Omg[:,r:r+b-1]
        Y=G[:,r:r+b-1]
        Qi,R=np.linalg.qr(Y)
        Qi,R1=np.linalg.qr(Qi-Q@(Q.T@Qi))
        R=R1@R
        
        Bi=torch.from_numpy(np.linalg.inv(R.T))@((H[:,r:r+b-1]).T-(Y.T@Q)@B-t.T@B)
        if i==1:
            Q=Qi
            B=Bi
        else:
            Q=np.hstack((Q,Qi))
            B=np.vstack((B,Bi))
        r=r+b

        temp=E-np.linalg.norm(Bi,'fro')**2

        if temp<threshold:
            for j in range(1,b):
                E=E-np.linalg.norm(Bi[j,:])**2
                if E< threshold:
                    flag=True
                    break
        else:
            E=temp
        if flag:
            k=(i-1)*b+j
            break
    if not flag:
      
        print('fail to converge')
        return Q,B,E,i
    return Q,B,E,i
    
    


