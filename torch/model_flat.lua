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
  nfeats={32,64,128},
  nfeats_decoder={32,64,128},
  out_size = 128*8*8,
  out_size_decoder = 128*8*8,
  conv_layers = 3,
  filtsize = 5,
  poolsize = 2,
  img_size=64,
  rnn_layers=1,
  fc_layers=0,
  rnn_size=1024,
  g_size=1024,
  l_size=4,
  n_glimpses=4,
  init_weight=0.05,
  batch_size = 4,
  max_grad_norm = 2
}

local function rnn(x, prev_h, input_size, output_size)
  local i2h = nn.Linear(input_size, output_size)
  local h2h = nn.Linear(params.rnn_size, output_size)

  local next_h = nn.Tanh()(nn.BatchNormalization(output_size)(nn.CAddTable()({i2h(x), h2h(prev_h)})))

  return i2h, h2h, next_h
end

local function lstm(x, prev_c, prev_h, input_size, output_size)
  -- Calculate all four gates in one go
  local i2h = nn.Linear(input_size, 4*output_size)
  local h2h = nn.Linear(params.rnn_size, 4*output_size)
  local gates = nn.CAddTable()({i2h(x), h2h(prev_h)})
  
  -- Reshape to (batch_size, n_gates, hid_size)
  -- Then slize the n_gates dimension, i.e dimension 2
  local reshaped_gates = nn.Reshape(4,output_size)(gates)
  local sliced_gates = nn.SplitTable(2)(reshaped_gates)
  
  -- Use select gate to fetch each gate and apply nonlinearity
  local in_gate          = nn.Sigmoid()(nn.SelectTable(1)(sliced_gates))
  local in_transform     = nn.Tanh()(nn.BatchNormalization(params.rnn_size)(nn.SelectTable(2)(sliced_gates)))
  local forget_gate      = nn.Sigmoid()(nn.SelectTable(3)(sliced_gates))
  local out_gate         = nn.Sigmoid()(nn.SelectTable(4)(sliced_gates))

  local next_c           = nn.CAddTable()({
      nn.CMulTable()({forget_gate, prev_c}),
      nn.CMulTable()({in_gate,     in_transform})
  })
  local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

  return i2h, h2h, next_c, next_h
end

function core_network()
  local x  = nn.Identity()()
  local prev_s = nn.Identity()()
  local prev_l = nn.Identity()()

  local l_transform = nn.Linear(4,128)
  l_transform.weight:normal(0, params.init_weight)

  local transforms = {}
  transforms[0] = nn.Linear(params.out_size,params.g_size-128)
  transforms[0].weight:normal(0, params.init_weight)

  --local enc = encoder(3)
 
  local g_x = nn.BatchNormalization(params.g_size-128)(transforms[0](nn.View(params.out_size):setNumInputDims(3)(x)))
  local g_l = l_transform(prev_l)

  local g = nn.JoinTable(1,1)({g_x,g_l})

  --local j_transform = nn.Linear(2*params.g_size,params.g_size)
  --j_transform.weight:normal(0, params.init_weight)

  --g = nn.BatchNormalization(params.g_size)(j_transform(g))
  --g = nn.ReLU(true)(g)

  --for i = 1,params.fc_layers do
    --transforms[i] = nn.Linear(params.g_size,params.g_size)
    --g = nn.CMulTable()({transforms[i](g), g_l})
  --end

  --local h  = {[0] = nn.Dropout(0.8)(g)}
  local h  = {[0] = g}

  local next_s = {}
  local split = {prev_s:split(2 * params.rnn_layers)}
  --[[local split
  if params.rnn_layers == 1 then
    split = prev_s
  else
    split = {prev_s:split(params.rnn_layers)}
  end]]
  local i2h = {}
  local h2h = {}
  for i = 1, params.rnn_layers do
    local prev_c = split[2 * i - 1]
    local prev_h = split[2 * i]
    --local prev_h = split[i]
    local next_c, next_h
    if i == 1 then
      i2h[i], h2h[i], next_c, next_h = lstm(h[i - 1], prev_c, prev_h, params.g_size, params.rnn_size)
      --i2h[i], h2h[i], next_h = rnn(h[i - 1], prev_h, params.g_size, params.rnn_size)
    else
      i2h[i], h2h[i], next_c, next_h = lstm(h[i - 1], prev_c, prev_h, params.rnn_size, params.rnn_size)
      --i2h[i], h2h[i], next_h = rnn(h[i - 1], prev_h, params.rnn_size, params.rnn_size)
    end
    table.insert(next_s, next_c)
    table.insert(next_s, next_h)
    h[i] = next_h
  end

  local module = nn.gModule({x, prev_l, prev_s},
                                      {h[params.rnn_layers], nn.Identity()(next_s)})
  module:getParameters():normal(0, params.init_weight)

  for i = 0,params.fc_layers do
    transforms[i].bias:normal(0, params.init_weight/20)
  end
  for i = 1,params.rnn_layers do
    i2h[i].weight:normal(0,params.init_weight/10)
    h2h[i].weight:normal(0,params.init_weight/10)
    i2h[i].bias:normal(0,params.init_weight/200)
    h2h[i].bias:normal(0,params.init_weight/200)
  end
  return module
end

function combine_network()

  local slice_width = 64

  local h = nn.Identity()()
  local l = nn.Identity()()
  
  local transforms = {}
  transforms[1] = nn.Linear(params.rnn_size,params.out_size_decoder/4)
  transforms[2] = nn.Linear(params.out_size_decoder/4,params.out_size_decoder/4)
  transforms[3] = nn.Linear(params.out_size_decoder/4,params.out_size_decoder)
  local l_transforms = {}
  l_transforms[0] = nn.Linear(4,16)
  l_transforms[1] = nn.Linear(16,2*slice_width)
  l_transforms[2] = nn.Linear(2*slice_width,params.out_size_decoder/4)
  l_transforms[3] = nn.Linear(2*slice_width,params.out_size_decoder/4)
  --l_transforms[4] = nn.Linear(4*slice_width,4*slice_width)
  
  --local h1 = nn.Narrow(2,1,slice_width)(h)
  local g_l0 = nn.ReLU(true)(l_transforms[0](l))
  g_l0 = nn.ReLU(true)(l_transforms[1](g_l0))
  --g_l0 = nn.JoinTable(1,1)({g_l0,h1})
  --g_l0 = nn.ReLU(true)(nn.BatchNormalization(4*slice_width)(l_transforms[4](g_l0)))
  local g_l1 = l_transforms[2](g_l0)
  local g_l2 = l_transforms[3](g_l0)
 
  --local h2 = nn.Narrow(2,slice_width+1,params.rnn_size-slice_width)(h)
  local h2 = h
  --local h2 = h
  h2 = transforms[1](h2)
  h2 = nn.CMulTable()({h2,g_l1})
  --h2 = transforms[2](h2)
  --h2 = nn.BatchNormalization(params.out_size_decoder/4)(h2)
  h2 = nn.Tanh(true)(h2)
  --h2 = nn.Dropout(0.75)(h2)
  h2 = nn.CMulTable()({h2,g_l2})
  h2 = transforms[3](h2)
  --h2 = nn.BatchNormalization(params.out_size_decoder)(h2)
  local y = h2

  for i = 1,3 do
    transforms[i].weight:normal(0,params.init_weight)
    transforms[i].bias:zero()
  end
  for i = 0,3 do
    l_transforms[i].weight:normal(0,2*params.init_weight)
    l_transforms[i].bias:zero()
  end

  return nn.gModule({h,l},{nn.View(128,8,8):setNumInputDims(1)(y)})
end

function mask_module(batch,feats,rows,cols)
  local x = nn.Identity()()
  local mask = nn.Mask(batch,feats,rows,cols)(x)
  return nn.gModule({x},{nn.CMulTable(true)({x,mask})})
end
