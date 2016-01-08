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


--[[function encoder()
  local nfeats = params.nfeats
  local filtsize = params.filtsize
  local poolsize = params.poolsize
  nfeats[0] = 1

  local x_conv = {}
  x_conv[0] = x

  local net = nn.Sequential()

  local conv_transforms = {}

  for i = 1,params.conv_layers do
    conv_transforms[i] = cudnn.SpatialConvolution(nfeats[i-1], nfeats[i], filtsize, filtsize)
    net:add(conv_transforms[i])
    net:add(nn.ReLU(true))
    net:add(cudnn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))
  end

  local fc_transforms = {}
  net:add(nn.View(params.out_size))

  fc_transforms[1] = nn.Linear(params.out_size,params.out_size)

  net:add(fc_transforms[1])
  net:add(nn.ReLU(true))

  for i = 1,params.conv_layers do
    local init1 = 60/(nfeats[i-1]*params.filtsize*params.filtsize)
    conv_transforms[i].weight:normal(0,init1)
    conv_transforms[i].bias:normal(0, init1/120)
  end
  for i = 1,params.fc_layers do
    fc_transforms[i].bias:normal(0, params.init_weight/20)
  end

  return net
end

function decoder()
  local nfeats = {128,96,128}
  local filtsize = params.filtsize
  local net = nn.Sequential()
  local transforms = {}
  transforms[0] = nn.Linear(params.out_size,nfeats[3]*8*8)
  net:add(transforms[0])
  net:add(nn.View(nfeats[3],8,8))
  net:add(nn.SpatialUpSamplingNearest(2))
  net:add(mask_module(16,16))
  net:add(nn.SpatialZeroPadding(2,2,2,2))
  transforms[1] = cudnn.SpatialConvolution(nfeats[3], nfeats[2], filtsize, filtsize)
  net:add(transforms[1])
  net:add(nn.ReLU(true))
  net:add(nn.SpatialUpSamplingNearest(2))
  net:add(mask_module(32,32))
  net:add(nn.SpatialZeroPadding(2,2,2,2))
  transforms[2]=cudnn.SpatialConvolution(nfeats[2], nfeats[1], filtsize, filtsize)
  net:add(transforms[2])
  net:add(nn.ReLU(true))
  net:add(nn.SpatialUpSamplingNearest(2))
  net:add(mask_module(64,64))
  net:add(nn.SpatialZeroPadding(2,2,2,2))
  transforms[3] = cudnn.SpatialConvolution(nfeats[1], nfeats[1], filtsize, filtsize) 
  net:add(transforms[3])
  net:add(nn.ReLU(true))
  net:add(nn.SpatialZeroPadding(2,2,2,2))
  transforms[4] = cudnn.SpatialConvolution(nfeats[1], 1, filtsize, filtsize)
  net:add(transforms[4])
  net:add(nn.Sigmoid(true))
  
  transforms[0].weight:normal(0,params.init_weight)
  transforms[0].bias:zero()
  for i = 1,4 do
    transforms[i].weight:normal(0,params.init_weight/10)
    transforms[i].bias:zero()
  end
  transforms[4].weight:normal(0,params.init_weight/10)
  transforms[4].bias:zero()
  return net
end]]--

function mask_module(batch,feats,rows,cols)
  local x = nn.Identity()()
  local mask = nn.Mask(batch,feats,rows,cols)(x)
  return nn.gModule({x},{nn.CMulTable(true)({x,mask})})
end

function decoder_layer(feats_in,feats_out,width)
  local filtsize = params.filtsize
  local net = nn.Sequential()
  net:add(nn.SpatialUpSamplingNearest(2))
  --net:add(mask_module(width*2,width*2))
  net:add(nn.SpatialZeroPadding(2,2,2,2))
  transform = cudnn.SpatialConvolution(feats_in,feats_out, filtsize, filtsize)
  net:add(transform)
  net:add(nn.SpatialBatchNormalization(feats_out))
  net:add(nn.ReLU(true)) 
  transform.weight:normal(0,params.init_weight/10)
  transform.bias:zero()
  return net
end

function decoder_last_layer(feats_in,feats_out)
  local filtsize = params.filtsize
  local net = nn.Sequential()
  net:add(nn.SpatialZeroPadding(2,2,2,2))
  transform = cudnn.SpatialConvolution(feats_in, feats_out, filtsize, filtsize)
  net:add(transform)
  net:add(nn.SpatialBatchNormalization(feats_out))
  --net:add(nn.Sigmoid(true))
  transform.weight:normal(0,params.init_weight/10)
  transform.bias:zero()
  return net
end

function encoder_layer(feats_in,feats_out)
  local filtsize = params.filtsize
  local poolsize = params.poolsize
  local net = nn.Sequential()
  transform = cudnn.SpatialConvolution(feats_in, feats_out, filtsize, filtsize)
  net:add(nn.SpatialZeroPadding(2,2,2,2))
  net:add(transform)
  net:add(nn.SpatialBatchNormalization(feats_out))
  net:add(nn.ReLU(true))
  net:add(cudnn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))
  local init1 = 60/(feats_in*params.filtsize*params.filtsize)
  transform.weight:normal(0,init1)
  transform.bias:normal(0, init1/120)
  return net
end

function encoder(n_layers)
  local net = nn.Sequential()
  local feats = params.nfeats
  feats[0]=1
  for i = 1,n_layers do
    net:add(encoder_layer(feats[i-1],feats[i]))
    if i == 2 then
      net:add(nn.SpatialDropout(0.5))
    end
  end
  return net
end

function decoder(n_layers)
  local net = nn.Sequential()
  local feats = params.nfeats_decoder
  feats[0]=32
  local widths = {32,16,8}
   for i = n_layers,1,-1 do
    --[[if i == 2 then
      net:add(nn.SpatialDropout())
    end]]
    net:add(decoder_layer(feats[i],feats[i-1],widths[i]))
  end
  net:add(decoder_last_layer(feats[0],1))
  return net
end

function loc_transform(encoder,width,n_feats)

  local filtsize = params.filtsize

  local x = nn.Identity()()
  local l = nn.Identity()()

  local transforms = {}
  transforms[1] = nn.Linear(4,256)
  transforms[2] = nn.Linear(256,256)
  transforms[3] = nn.Linear(256,n_feats*width*width)
  transforms[4] = nn.Linear(256,n_feats*width*width)
  transforms[5] = cudnn.SpatialConvolution(n_feats, n_feats, filtsize, filtsize)
  transforms[6] = cudnn.SpatialConvolution(n_feats, n_feats, filtsize, filtsize)
  
  local l_2 = nn.ReLU(true)(transforms[1](l))
  local l_3 = nn.ReLU(true)(transforms[2](l_2))
  local l_map1 = nn.View(n_feats,width,width)(nn.Tanh(true)(transforms[3](l_3)))
  local l_map2 = nn.View(n_feats,width,width)(nn.Sigmoid(true)(transforms[4](l_3)))

  local x_2 = nn.SpatialZeroPadding(2,2,2,2)(nn.CMulTable()({x,l_map1}))
  local x_3 = nn.ReLU(true)(transforms[5](x_2))
  local x_4 = nn.SpatialZeroPadding(2,2,2,2)(nn.CMulTable()({x_3,l_map2}))
  local out = nn.ReLU(true)(transforms[6](x_4))

  for i = 1,4 do
    transforms[i].weight:normal(0,params.init_weight/10)
    transforms[i].bias:zero()
  end
  transforms[5].weight:normal(0,params.init_weight/10)
  transforms[5].bias:zero()
  transforms[6].weight:normal(0,params.init_weight/10)
  transforms[6].bias:zero()

  local module = nn.gModule({x,l},{out})

  local net = nn.ParallelTable()
  net:add(encoder)
  net:add(nn.Identity())

  local out_net = nn.Sequential()
  out_net:add(net)
  out_net:add(module)

  return out_net

end