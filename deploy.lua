require 'nn';
require 'image';
require 'optim';
require 'gnuplot';

--totalNumLines
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

testData = data
testlabels = labels
testData:add(-testData:mean())
testData:div(testData:std())

model = torch.load('linear.t7')

N = testData:size(1)
teSize = N

print('Testing accuracy')
correct = 0
class_perform = {0,0,0,0,0,0,0}
class_size = {0,0,0,0,0,0,0}
classes = {'A','B','C','D','E','F','G'}
for i=1,N do
    local groundtruth = testlabels[i]
    local example = torch.Tensor(32769,1);
    example = testData[i]
    class_size[groundtruth] = class_size[groundtruth] +1
    local prediction = model:forward(example)
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    --print(#example,#indices)
    --print('ground '..groundtruth, indices[1])
    if groundtruth == indices[1] then
        correct = correct + 1
        class_perform[groundtruth] = class_perform[groundtruth] + 1
    end
    collectgarbage()
end
print("Overall correct " .. correct .. " percentage correct" .. (100*correct/teSize) .. " % ")
for i=1,#classes do
   print(classes[i], 100*class_perform[i]/class_size[i] .. " % ")
end