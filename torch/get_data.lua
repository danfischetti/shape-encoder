require 'image'
json = require('json')
require('json.rpc')
require 'nn'

local filepath = '/ramcache/renders/'

params = {
  n_glimpses=4,
  n_samples=20,
  loc = {{0,0,0.1,0},{0,0.5,0.2,0},{0,1,-0.1,0},{0,-0.5,-0.2,0}},
  n_batches=24,
  batch_size = 56
  }

function get_view(l,index,n)
  json.rpc.call('http://localhost:9090','get_view',{l[1],l[2],l[3],l[4],index,n})
  return image.load(filepath .. 'm' .. n .. '.png')
end

function loadBatch(n)
  json.rpc.call('http://localhost:9090','loadObjects',{n})
end

x_in = {}
x = {}
y = {}


function main()
  for i = 1,params.n_glimpses do
    x_in[i] = {}
  end
  for i = 1,params.n_samples do
    x[i] = {}
    y[i] = {}
  end
  for a = 1,16 do
    for i = 1,params.n_batches do
      print(i)
      local indices = {}
      local i2 = i
      if i == 16 then
        i2 = 25
      end
      for k = 1,params.batch_size do
        index=4*(i2-1)+((k-1)%14)*100+math.floor((k-1)/14)
        indices[k]=index
      end
      loadBatch(i2)
      for j = 1,params.batch_size do
        for k = 1,params.n_glimpses do
          local loc = params.loc[k]
          local img = get_view(loc,indices[j],j):view(1,1,64,64)
          table.insert(x_in[k],img)
        end
        for k = 1,params.n_samples do
          local loc = torch.Tensor({0,0,0,0})
          local h = halton2()
          loc[2]=h[1]*2-1
          loc[3]=(h[2]*2-1)*.2
          local img = get_view(loc,indices[j],j):view(1,1,64,64)
          table.insert(x[k],img)
          table.insert(y[k],loc:view(1,4))
        end
      end
    end
  end
  module = nn.JoinTable(1)
  for n = 1,params.n_samples do
    x[n] = module:forward(x[n]):clone()
    y[n] = module:forward(y[n]):clone()
  end
  for n = 1,params.n_glimpses do
    x_in[n] = module:forward(x_in[n]):clone()
  end
  torch.save('targets.dat',x)
  torch.save('renders.dat',x_in)
  torch.save('target_locs.dat',y)
end

IND = 0

function halton(index,base)
  local result = 0
  local i = index
  local f = 1
  while i > 0 do
    f = f/base
    result = result+f*(i%base)
    i = math.floor(i/base)
  end
  return result
end

function halton2()
  IND=IND+1
  return {halton(IND,2),halton(IND,3)}
end

main()