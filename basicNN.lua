require 'nn';
require 'image';
require 'optim';
require 'gnuplot';

model = nn.Sequential()
model:add(nn.Linear(32769,100))
model:add(nn.ReLU())
model:add(nn.Linear(100,13))
model:add(nn.LogSoftMax())

local file = io.open("xmXtext.txt")
l=0
data = torch.Tensor(200,32769)
target = torch.Tensor(200)
for line in file:lines() do
	l=l+1
	local A = line:split(" ")
	data[l] = torch.Tensor(A)
	--print(l,#A)
end
