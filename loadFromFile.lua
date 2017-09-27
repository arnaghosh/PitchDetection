local file = io.open("xmXtext.txt")
l=0
T = torch.Tensor(200,32769)
for line in file:lines() do
	l=l+1
	local A = line:split(" ")
	T[l] = torch.Tensor(A)
	--print(l,#A)
end
print(T:size(),T:min(),T:max())