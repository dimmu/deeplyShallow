#function to load the data
#uses pandas for faster loading ... 
#load data
import pandas as pd
from binascii import crc32
import numpy as np


data=pd.read_csv("data/training.1600000.processed.noemoticon.csv", encoding='latin-1', delimiter=',', header=-1)
labels=data[0]
text=data[5].get_values()

#test_data=pd.read_csv("data/testdata.manual.2009.06.14.csv",encoding='latin-1', delimiter=',', header=-1)



valIdx=np.arange(0,len(labels),100)
testIdx=np.arange(1,len(labels),100)
trainIdx=np.setxor1d(np.arange(0,len(labels)),np.append(valIdx,testIdx))

Y=np.zeros((len(labels),2))
Y[labels==0,0]=1
Y[labels==4,1]=1

trainText=text[trainIdx]
valText=text[valIdx]
testText=text[testIdx]

#make labels
trainY=Y[trainIdx]
valY=Y[valIdx]
testY=Y[testIdx]

dims=30000

valX=fastWordVec(valText,dims)
testX=fastWordVec(testText,dims)


#vectorize somehow? 

#rolling window

#unique words

#bi-words

#tri-words

def fastWordVec(txts, num_bags):
    m = len(txts)
    rtnIdx=[]
    rtnVal=[]
    thisBag={}
    #tmp = 0
    for i in range(0,m):
        tmp=txts[i].split()
        for j in range(0,len(tmp)): 
            lala = crc32(bytes(tmp[j], 'UTF-8'))
            if lala in thisBag:
                thisBag[lala]+=1
            else:
                thisBag[lala]=1
        for key in thisBag:
            rtnIdx.append([i,key])
            rtnVal.append(thisBag[key])
    return (rtnIdx, rtnVal)

def fastWordVec(txts, num_bags):
    m = len(txts)
    X = np.zeros((m,num_bags))
    #tmp = 0
    for i in range(0,m):
        tmp=txts[i].split()
        for j in range(0,len(tmp)): 
            lala = crc32(bytes(tmp[j], 'UTF-8'))
            X[i][lala%num_bags]+=1
    return X

def fastWindowVec(txts, window_size, num_bags):
    m = len(txts)
    X = np.zeros((m,num_bags))
    tmp = 0
    for i in range(0,m):
        for j in range(0,len(txts[i])-window_size): 
            tmp = crc32(bytes(txts[i][j:j+5], 'UTF-8'))
            X[i][tmp%num_bags]+=1
    return X


def windowVec( X, m ):
    rtn = []
    for i in range(0, len(X)):
        tmp = []
        for j in range(0, len(X[i])-m):
            tmp.append(crc32(bytes(X[i][j:j+5], 'UTF-8')))
        rtn.append(tmp)
        if i%1000 ==0 :
            print(".")
    return rtn

XXX=[];
for i in range(0, len(XX)):
    for j in range(0,len(XX[i])):
        tmp=XX[i][j]%3000
        XXX[i][tmp]+=1
    if i%1000==0:
        print(".")


def expGradient(X,Y,its, eta):
    (a,b)=X.shape
    w=np.zeros((b,1))/b
    wPlus=np.ones((b,1))/b
    wMinus=np.ones((b,1))/b
    upd=np.zeros((b,1))
    for i in range(0,its):
        thisPerm=np.random.permutation(a)
        ccc=0
        acc=0
        bal=0
        yMax=0;
        yMin=1;
        for j in range(0, a):
            idx=thisPerm[j]
            yP=X[idx].dot(w)
            if yP>yMax:
                yMax=yP
            if yP<yMin:
                yMin=yP
            #print("dim of x[idx] is ",X[idx].shape, " w: ", w.shape, " yp: ",yP.shape, " a ",a , " b ",b)
            
#            print("iteration ", i, " sample ", j, " predicted ",yP, " actual :",Y[idx])
            thisErr=abs(Y[idx]-yP)
            ccc+=thisErr
            bal+=Y[idx]
            if yP*Y[idx]>0:
                acc+=1
            updPlus=np.exp(-2*eta*(yP-Y[idx])*X[idx] )
            updMinus=np.exp(2*eta*(yP-Y[idx])*X[idx] )        
            #print("dim of updated is ", upd.shape)
            #for k in range(1,b):
            #    w[k]=w[k]*upd[k]
            wPlus=wPlus*np.reshape( updPlus,(b,1))
            wMinus=wMinus*np.reshape(updMinus,(b,1))
            w=wPlus-wMinus
            w = w/sum(abs(w))
            if j%1000 == 0:
                print("err ",i,",",j,":",ccc, " acc:",acc, " bal: ", bal, " max ",max(w), " min: ", min(w)," sum ",sum(w)," yMax: ", yMax," yMin: ",yMin)
                ccc=0
                acc=0
                bal=0
                yMax=0
                yMin=1
        return w