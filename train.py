from get_dataloader import train_loader,valid_loader,test_loader
import torch.nn as nn
from torch import optim
import torch
from timeit import default_timer as timer
import numpy as np
import argparse
from weather_model import Model

parser=argparse.ArgumentParser()
parser.add_argument('--scale',type=float,default=1)
parser.add_argument('--name',type=str)
args=parser.parse_args()

net=Model(args.scale)
net=net.to('cuda')
model_name=args.name+'.pkl'
n1=args.name+'trainacc.npy'
n2=args.name+'trainloss.npy'
n3=args.name+'validacc.npy'
n4=args.name+'validloss.npy'
loss_func=nn.NLLLoss()
optimizer=optim.Adam(net.parameters())
trainloss=[]
trainacc=[]
validloss=[]
validacc=[]
bestloss = np.Inf
for epoch in range(100):
    start1=timer()
    train_acc=0
    train_loss=0
    valid_acc=0
    valid_loss=0
    net.train()
    for i,(data,target)in enumerate(train_loader):
        data=data.cuda()
        target=target.cuda()
        out=net(data)
        loss=loss_func(out,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _,pred=torch.max(out,dim=1)
        correct=pred.eq(target.data.view_as(pred))
        acc=torch.mean(correct.type(torch.FloatTensor))
        acc=acc.item()*len(data)
        train_acc+=acc
        train_loss+=loss.item()
        progress=round(100*(i+1)/len(train_loader),2)
        time=round(timer()-start1,2)
        print(epoch,'\t',progress,'% complete',time,'second passed',end='\r')
    with torch.no_grad():
        net.eval()
        start2=timer()
        for i,(data,target) in enumerate(valid_loader):
            data=data.cuda()
            target=target.cuda()
            out=net(data)
            loss=loss_func(out,target)
            _,pred=torch.max(out,dim=1)
            correct=pred.eq(target.data.view_as(pred))
            acc=torch.mean(correct.type(torch.FloatTensor))
            acc=acc.item()*len(data)
            valid_acc+=acc
            valid_loss+=loss.item()
            progress = round(100 * (i + 1) / len(valid_loader), 2)
            time = round(timer() - start2, 2)
            print(epoch, '\t', progress, '% complete', time, 'second passed', end='\r')
    train_acc=100*train_acc/len(train_loader.dataset)
    valid_acc=100*valid_acc/len(valid_loader.dataset)
    train_loss=train_loss/len(train_loader.dataset)
    valid_loss=valid_loss/len(valid_loader.dataset)
    if valid_loss<bestloss:
        torch.save(net,model_name)
        bestloss=valid_loss
        bestacc=valid_acc
        bestepoch=epoch
    trainacc.append(train_acc)
    trainloss.append(train_loss)
    validacc.append(valid_acc)
    validloss.append(valid_loss)
    train_acc=round(train_acc,2)
    train_loss=round(train_loss,6)
    valid_acc=round(valid_acc,2)
    valid_loss = round(valid_loss, 6)
    print('epoch:',epoch,'train_acc:',train_acc,'train_loss:',train_loss,'valid_acc:',valid_acc,'valid_loss:',valid_loss)

print('best epoch:',bestepoch,' ','bestacc:',bestacc,'  ','bestloss:',bestacc)
net=torch.load(model_name)
with torch.no_grad():
    net.eval()
    test_acc=0
    test_loss=0
    for i, (data, target) in enumerate(test_loader):
        data = data.cuda()
        target = target.cuda()
        out = net(data)
        loss = loss_func(out, target)
        _, pred = torch.max(out, dim=1)
        correct = pred.eq(target.data.view_as(pred))
        acc = torch.mean(correct.type(torch.FloatTensor))
        acc = acc.item() * len(data)
        test_acc += acc
        test_loss += loss.item()
test_acc = 100 * test_acc / len(test_loader.dataset)
test_loss = test_loss / len(test_loader.dataset)

print('test_acc:',test_acc,'    ','test_loss:',test_loss)

trainacc=np.array(trainacc)
np.save(n1,trainacc)
trainloss=np.array(trainloss)
np.save(n2,trainloss)
validacc=np.array(validacc)
np.save(n3,validacc)
validloss=np.array(validloss)
np.save(n4,validloss)