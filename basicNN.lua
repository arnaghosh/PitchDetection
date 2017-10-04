require 'nn';
require 'image';
require 'optim';
require 'gnuplot';

model = require 'model.lua'

local file = io.open("xmXtext.txt")
l=0
data = torch.Tensor(200,32769)
labels = torch.Tensor(200)
for line in file:lines() do
	l=l+1
	local A = line:split(" ")
	data[l] = torch.Tensor(A)
	--print(l,#A)
end

data = data:resize(200,32769,1)
--data = data:resize(200*99,331)

local file = io.open("basicCnotes - Sheet1.csv")
l=0
i=0
for line in file:lines() do
	l=l+1
	local A = line:split(",")
	--print(string.byte(A[1],1)-64)
	local t = string.byte(A[1],1)-64
	--repeating GT for 4 seconds each
	--print(#labels,i+1, i+(4*99),l)
	labels[{{i+1,i+(4)}}] = t
	i=i+(4);
	--print(t)
	--print(l,#A)
end
--print(labels)

trainData = data
trainLabels = labels
trainData:add(-trainData:mean())
trainData:div(trainData:std())
print(trainData:mean(), trainData:std())
N=200
local theta,gradTheta = model:getParameters()
criterion = nn.ClassNLLCriterion()

local x,y

local feval = function(params)
if theta~=params then
theta:copy(params)
end
gradTheta:zero()
out = model:forward(x)
--print(#x,#out,#y)
local loss = criterion:forward(out,y)
local gradLoss = criterion:backward(out,y)
model:backward(x,gradLoss)
return loss, gradTheta
end

batchSize = 10

indices = torch.randperm(trainData:size(1)):long()
--trainData = trainData:index(1,indices)
--trainLabels = trainLabels:index(1,indices)

epochs = 25
print('Training Starting')
local optimParams = {learningRate = 0.001, learningRateDecay = 0.0000}
local _,loss
local losses = {}
for epoch=1,epochs do
    collectgarbage()
    print('Epoch '..epoch..'/'..epochs)
    for n=1,N, batchSize do
        x = trainData:narrow(1,n,batchSize)
        y = trainLabels:narrow(1,n,batchSize)
        --print(y)
        _,loss = optim.adam(feval,theta,optimParams)
        losses[#losses + 1] = loss[1]
    end
    local plots={{'Training Loss', torch.linspace(1,#losses,#losses), torch.Tensor(losses), '-'}}
    gnuplot.pngfigure('Training2.png')
    gnuplot.plot(table.unpack(plots))
    gnuplot.ylabel('Loss')
    gnuplot.xlabel('Batch #')
    gnuplot.plotflush()
    --permute training data
    indices = torch.randperm(trainData:size(1)):long()
    trainData = trainData:index(1,indices)
    trainLabels = trainLabels:index(1,indices)
end

torch.save('CNN.t7',model)
