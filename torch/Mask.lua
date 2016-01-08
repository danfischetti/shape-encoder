local Mask, parent = torch.class('nn.Mask','nn.Module')

function Mask:__init(rows,cols)
	parent.__init(self)

	self.rows,self.cols = rows,cols

	self.output = torch.zeros(1,1,rows,cols)
	self.gradInput = torch.zeros(1,1,rows,cols)

	for i = 1,rows do
		for j = 1,cols do
			if (i%2==0 and j%2==0) then
						self.output[{1,1,i,j}] = 1
			end
		end
	end

end

function Mask:updateOutput(input)
	local size = input:size()
	return torch.expand(self.output,size[1],size[2],self.rows,self.cols)
end

function Mask:updateGradInput(input, gradOutput)
	local size = input:size()
	return torch.expand(self.gradInput,size[1],size[2],self.rows,self.cols)
end