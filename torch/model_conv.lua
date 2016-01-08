require 'nn'
require 'cunn'
require 'cutorch'
require 'cudnn'
cudnn.benchmark = true
require 'nngraph'
require 'Adam'
require 'Mask'
require 'base'
require 'Alert'

params = {
  nfeats={32,64,64},
  nfeats_decoder={32,64,64},
  out_size = 64*8*8,
  out_size_decoder = 64*8*8,
  conv_layers = 3,
  filtsize = 5,
  poolsize = 2,
  img_size=64,
  rnn_layers=2,
  l_size=4,
  n_glimpses=4,
  init_weight=0.05,
  batch_size = 4,
  max_grad_norm = 0.7
}



local function CGRU(x, prev_h, input_size, output_size)
  -- Calculate 2 gates in one go
  local i2h = cudnn.SpatialConvolution(input_size, 2*output_size,5,5,1,1,2,2)
  local h2h = cudnn.SpatialConvolution(input_size, 2*output_size,5,5,1,1,2,2)
  local gates = nn.CAddTable()({i2h(x), h2h(prev_h)})

  local i2h2 = cudnn.SpatialConvolution(input_size, output_size,5,5,1,1,2,2)
  local h2h2 = cudnn.SpatialConvolution(input_size, output_size,5,5,1,1,2,2)
  
  -- Use Narrrow to slice each gate and apply nonlinearity
  local u                = nn.Sigmoid()(nn.Narrow(2,1,output_size)(gates))
  local r                = nn.Sigmoid()(nn.Narrow(2,output_size+1,output_size)(gates))
  local update           = nn.CMulTable()({u, prev_h})
  local reset            = nn.CMulTable()({r, prev_h})

  local next_h           = nn.CAddTable()({update,
      nn.CMulTable()({
        nn.AddConstant(1,false)(nn.MulConstant(-1,false)(u)),
        nn.Tanh()(nn.CAddTable()({i2h2(x), h2h2(reset)}))
      })
  })

  return next_h
end

local function CGRU2(s, input_size)
  -- Calculate 2 gates in one go
  local i2h = cudnn.SpatialConvolution(input_size, 2*input_size,5,5,1,1,2,2)
  local gates = i2h(s)

  local i2h2 = cudnn.SpatialConvolution(input_size, input_size,5,5,1,1,2,2)
  
  -- Use Narrrow to slice each gate and apply nonlinearity
  local u                = nn.Sigmoid()(nn.Narrow(2,1,input_size)(gates))
  local r                = nn.Sigmoid()(nn.Narrow(2,input_size+1,input_size)(gates))
  local update           = nn.CMulTable()({u, s})
  local reset            = nn.CMulTable()({r, s})

  local next_s           = nn.CAddTable()({update,
      nn.CMulTable()({
        nn.AddConstant(1,false)(nn.MulConstant(-1,false)(u)),
        nn.Tanh()(i2h2(reset))
      })
  })

  return next_s
end

function core_network()
  local x  = nn.Identity()()
  local prev_s = nn.Identity()()
  local prev_l = nn.Identity()()

  local l_transform = nn.Linear(4,8*8*8)
  l_transform.weight:normal(0, params.init_weight)
 
  local g_x = x
  local g_l = nn.View(8,8,8):setNumInputDims(1)(l_transform(prev_l))

  local g = nn.JoinTable(1,3)({g_x,g_l})

  local h  = {[0] = g}

  local next_s = {}
  local split = {prev_s:split(params.rnn_layers)}

  for i = 1, params.rnn_layers do
    local prev_h = split[i]
    local next_c, next_h
    next_h = CGRU(h[i - 1], prev_h, 72, 72)
    table.insert(next_s, next_h)
    h[i] = next_h
  end

  local module = nn.gModule({x, prev_l, prev_s},
                                      {h[params.rnn_layers], nn.Identity()(next_s)})
  module:getParameters():normal(0, params.init_weight)

  return module
end

function combine_network()

  local h = nn.Identity()()
  local l = nn.Identity()()
  local l_transform1 = nn.Linear(4,128)

  local l_transform2 = nn.Linear(128,8*8*8)

  local g_l =  nn.Tanh()(l_transform1(l))
 
  local g_l = nn.View(8,8,8):setNumInputDims(1)(l_transform2(g_l))

  local g = nn.JoinTable(1,3)({h,g_l})
  
  g = CGRU2(g,80)
  g = CGRU2(g,80)
  g = CGRU2(g,80)

  y = nn.Narrow(2,1,64)(g)

  local module = nn.gModule({h,l},{y})

  module:getParameters():normal(0, params.init_weight)

  return module
end

function mask_module(batch,feats,rows,cols)
  local x = nn.Identity()()
  local mask = nn.Mask(batch,feats,rows,cols)(x)
  return nn.gModule({x},{nn.CMulTable(true)({x,mask})})
end
