require 'CAE'
require 'model_flat'
require 'image'

params.batch_size=8

model = {}
paramx={}
paramdx={}

function init()
	window1,painter1 = image.window()
  window2,painter2 = image.window()

  window1:show()
  window2:show()
	model.data = torch.load('renders6.dat')
	model.x = {}
  model.loc = {}
  model.x_in = {}
  model.x_enc = {}
  model.s = {}
  model.l = {}
  model.h = {}
  model.ds = {}
  model.start_s = {}
  model.h_err = torch.Tensor(params.batch_size,params.rnn_size):cuda()
  for j = 0, params.n_glimpses do
    model.s[j] = {}
    for d = 1, 2*params.rnn_layers do
      model.s[j][d] = torch.zeros(params.batch_size, params.rnn_size):cuda()
    end
  
  end
  for d = 1,2*params.rnn_layers do
    model.start_s[d] = torch.zeros(params.batch_size, params.rnn_size):cuda()
    model.ds[d] = torch.zeros(params.batch_size, params.rnn_size):cuda()
  end

	model.n = 1344
  model.targets = torch.load('targets5.dat')
  model.locations = torch.load('target_locs5.dat')
	model.train_size = torch.floor(model.n*0.8)
  model.train_batches = torch.floor(model.train_size/params.batch_size)
  model.train_size = model.train_batches*params.batch_size
  model.valid_size = torch.floor((model.n-model.train_size)/2)
  model.valid_batches = torch.floor(model.valid_size/params.batch_size)
  model.valid_size = model.valid_batches*params.batch_size
  model.test_size = model.n - (model.train_size + model.valid_size)

  model.in_loc = torch.Tensor{{0,0,0.1,0},{0,0.5,0.2,0},{0,1,-0.1,0},{0,-0.5,-0.2,0}}
  model.in_loc = model.in_loc:cuda()

  local n_layers=3
	model.encoder = encoder(n_layers)

	model.decoder = decoder(n_layers)

  model.encoder = model.encoder:cuda()
  model.decoder = model.decoder:cuda()
  model.combine_network = combine_network():cuda()

	model.core_network = core_network():cuda()

	cutorch.setDevice(1)
 
	model.criterion = nn.MSECriterion():cuda()
  paramx[1],paramdx[1] = model.encoder:getParameters()
  paramx[2],paramdx[2] = model.core_network:getParameters()
	paramx[3],paramdx[3] = model.combine_network:getParameters()
  paramx[4],paramdx[4] = model.decoder:getParameters()

  local p = torch.load('params_2MulLayers_1RNNLayer_noSplit.dat')
  paramx[1]:copy(p[1])
  paramx[2]:copy(p[2])
  paramx[3]:copy(p[3])

  model.encoders = g_cloneManyTimes(model.encoder, params.n_glimpses)
  model.networks = g_cloneManyTimes(model.core_network, params.n_glimpses)

	print('Model built.')

end

function encodeImages()
  model.x_enc = {}
  for i = 1,#model.x do
    module = nn.JoinTable(1)
    local data = model.x[i]
    local numBatches = data:size(1)/params.batch_size
    local temp = {}
    local net = model.encoder
    for j = 1,numBatches do
      table.insert(temp,net:forward(data[{{1+(j-1)*params.batch_size,j*params.batch_size}}]):view(params.batch_size,128*8*8))
    end
    model.x_enc[i] = module:forward(temp):clone():cuda()
  end
end

local function reset_ds()
  for d = 1, #model.ds do
    model.ds[d]:zero()
  end
end

local function fp(indices)
  g_replace_table(model.s[0], model.start_s)
  for i = 1, params.n_glimpses do
    model.x_in[i]=model.x[i][indices]
    model.x_enc[i]=model.encoders[i]:forward(model.x_in[i])
    model.l[i] = model.in_loc[i]:view(1,4):expand(params.batch_size,4)
    model.h[i], model.s[i] = 
      unpack(model.networks[i]:forward({model.x_enc[i], model.l[i], model.s[i-1]}))
  end
  
end

local function generate(loc,target)
  model.comb = model.combine_network:forward({model.h[params.n_glimpses],loc})
  model.rec = model.decoder:forward(model.comb)
  local err = model.criterion:forward(model.rec,target)
  return err
end

local function gen_err(loc,target)
  local rec_err = model.criterion:backward(model.rec,target)
  local comb_err = model.decoder:backward(model.comb,rec_err)
  model.h_err:add(1/#model.targets,model.combine_network:backward({model.h[params.n_glimpses],loc},comb_err)[1])
end

local function bp()

  local aux_factor = 0.3
 
  local l = model.l[params.n_glimpses]
  local s = model.s[params.n_glimpses - 1]
  local x = model.x_in[params.n_glimpses]
  local x_enc = model.x_enc[params.n_glimpses]
  
  local h_err = model.h_err:clone()
  
  local x_enc_err,l_err,s_err = unpack(model.networks[params.n_glimpses]:backward({x_enc,l,s},{h_err,model.ds}))

  local rec = model.decoder:forward(x_enc)
  local rec_err = model.criterion:backward(rec,model.x_in[params.n_glimpses])
  local x_dec_err = model.decoder:backward(x_enc,rec_err)
  x_enc_err:add(aux_factor,x_dec_err)

  local x_err = model.encoders[params.n_glimpses]:backward(x,x_enc_err)
  g_replace_table(model.ds, s_err)
  h_err:zero()
  cutorch.synchronize()
  for i = params.n_glimpses-1, 1, -1 do
    l = model.l[i]
    x = model.x_in[i]
    x_enc = model.x_enc[i]
    s = model.s[i - 1]
    x_enc_err,l_err,s_err = unpack(model.networks[i]:backward({x_enc,l,s},{h_err,model.ds}))
     
    rec = model.decoder:forward(x_enc)
    rec_err = model.criterion:backward(rec,model.x_in[i])
    x_dec_err = model.decoder:backward(x_enc,rec_err)
    x_enc_err:add(aux_factor,x_dec_err)

    x_err = model.encoders[i]:backward(x,x_enc_err)
    g_replace_table(model.ds, s_err)
    cutorch.synchronize()
  end
  local shrink_factor=1
  local norm_dw = 0
  for i = 1,#paramx do
    norm_dw = math.max(norm_dw,paramdx[i]:norm())
  end
  if norm_dw > params.max_grad_norm then
      shrink_factor = params.max_grad_norm / norm_dw
  end
  for i = 1,#paramx do
   paramdx[i]:mul(shrink_factor)
  end
end

function main()
  init()

  local adam = {}
  for i = 1,#paramx do
    adam[i] = nn.Adam(paramx[i],paramdx[i])
  end
  local best_err = 1000000000000000000000
  local loop = true
  local patience = 2
  local min_epochs = 10
  local t = 0
  local i = 0
  local disp_ind={}
  for j = 1,params.batch_size do
    table.insert(disp_ind,j)
    table.insert(disp_ind,j+params.batch_size)
    table.insert(disp_ind,j+2*params.batch_size)
    table.insert(disp_ind,j+3*params.batch_size)
  end
  while loop do
  	i = i+1
    local train_err = 0
    local valid_err = 0

    --16 copies of each training example (augmented)
    for n = 1,16 do
      local split = 4
      local offset = 0
        for j = 1,#model.networks do
          model.networks[j]:training()
          model.encoders[j]:training()
        end
        model.combine_network:training()
        model.decoder:training()
        local target = {}
        --number of batches to load into memory at once
        local train_chunk = math.ceil(model.train_batches/split)
        for j = 1,model.train_batches do

          if j%train_chunk == 1 then
            offset = j-1
            local ind = {{offset*params.batch_size+1+(n-1)*1344,params.batch_size*math.min(train_chunk+offset,model.train_batches)+(n-1)*1344}}
            for a = 1,#model.data do
              model.x[a] = model.data[a][ind]:cuda()
            end
            for a = 1,#model.locations do
              model.loc[a] = model.locations[a][ind]:cuda()
            end
            for a = 1,#model.locations do
              target[a] = model.targets[a][ind]:cuda()
            end
            collectgarbage()
          end

          --zero grad params
          for k = 1,#paramx do
            paramdx[k]:zero()
          end
          reset_ds()
          model.h_err:zero()

          local indices = {{1+(j-1-offset)*params.batch_size,(j-offset)*params.batch_size}}
          
          --forward propagate through recurrent network
          fp(indices)
          local to_img1 = {}
          local to_img2 = {}

          for k = 1,#model.targets do

            --forward and backward pass through combine network and decoder
            local err = generate(model.loc[k][indices],target[k][indices])
            train_err = train_err + err
            gen_err(model.loc[k][indices],target[k][indices])
            if(j%10 == 1 and k < 5) then
              table.insert(to_img1,target[k][indices]:clone())
              table.insert(to_img2,model.rec:clone())
            end
          end

          --backward through recurent network
          bp()
          for k = 1,#paramx do
            	adam[k]:step()
          end
          if(j%10 == 1) then
          	image.display{image = reorder(nn.JoinTable(1):forward(to_img1):split(1),disp_ind),win=painter1}
      		  image.display{image = reorder(nn.JoinTable(1):forward(to_img2):split(1),disp_ind),win=painter2}
          end
        end
        for j = 1,#model.networks do
          model.networks[j]:training()
          model.encoders[j]:training()
        end
        model.combine_network:training()
        model.decoder:training()

        --validation set forward pass
        split = 2
        local valid_chunk = math.ceil(model.valid_batches/split)
        for j = 1,model.valid_batches do

          if j%valid_chunk == 1 then
            offset = model.train_batches+j-1
            local ind = {{offset*params.batch_size+1+(n-1)*1344,params.batch_size*math.min(valid_chunk+offset,model.valid_batches+model.train_batches)+(n-1)*1344}}

            for a = 1,#model.data do
              model.x[a] = model.data[a][ind]:cuda()
            end
            for a = 1,#model.locations do
              model.loc[a] = model.locations[a][ind]:cuda()
            end
            for a = 1,#model.locations do
              target[a] = model.targets[a][ind]:cuda()
            end
            collectgarbage()
          end

          local indices = {{model.train_size+1+(j-1-offset)*params.batch_size,model.train_size+(j-offset)*params.batch_size}}
          fp(indices)

          local to_img1 = {}
          local to_img2 = {}
          
          for k = 1,#model.targets do
            local err = generate(model.loc[k][indices],target[k][indices])
            valid_err = valid_err + err
            if(j%10 == 1 and k < 5) then
              table.insert(to_img1,target[k][indices]:clone())
              table.insert(to_img2,model.rec:clone())
            end
          end
          if(j%10 == 1) then
          	image.display{image = reorder(nn.JoinTable(1):forward(to_img1):split(1),disp_ind),win=painter1}
            image.display{image = reorder(nn.JoinTable(1):forward(to_img2):split(1),disp_ind),win=painter2}
          end
        end
        collectgarbage()
    end
    train_err = train_err/(#model.targets*model.train_batches*10)
    valid_err = valid_err/(#model.targets*model.valid_batches*10)
    if i > min_epochs and train_err < valid_err then
      if valid_err < best_err then
      	t = 0
      	best_err = valid_err
      else
      	t = t + 1
      	if t > patience then
      		loop = false
      	end
      end
    end
    print('Epoch ' .. i .. ', Train err: ' .. train_err .. ' Valid err: ' .. valid_err)
  end
end

function showBatch(i,l)
  local x = model.train_set_x[{{i,i+7}}]
  local loc = torch.Tensor(l):view(1,4):expand(8,4):cuda()
  image.display(model.net:forward({x,loc}))
end

function testOne(ind)
  for j = 1,#model.networks do
    model.networks[j]:training()
  end
  local h={}
  g_replace_table(model.s[0], model.start_s)
  local x_in = {}
  local l = {}
  for i = 1, params.n_glimpses do
    x_in[i]=model.data[i][{{ind,ind+params.batch_size-1}}]:cuda()
    l[i] = model.in_loc[i]:view(1,4):expand(params.batch_size,4)
    print(l[i])
    h[i], model.s[i] = 
      unpack(model.networks[i]:forward({x_in[i], l[i], model.s[i-1]}))
    print(h[i][{{},{1,8}}])
  end

  local join = nn.JoinTable(1)

  image.display(join:forward(x_in))

  local out_loc = torch.Tensor{{0,0,0.1,0},{0,0.25,0,0},{0,0.5,0.2,0},{0,0.75,0,0},{0,1,-0.1,0},{0,-0.75,0,0},{0,-0.5,-0.2,0}}
  out_loc = out_loc:cuda()
  local comb_print = {}
  local x_out = {}
  for i = 1,out_loc:size(1) do
    print(out_loc[i])
    local comb = model.combine_network:forward({h[params.n_glimpses],out_loc[i]:view(1,4):expand(params.batch_size,4)})
    comb_print[i] = comb[{{},{1,3}}]:float():clone()
    local rec = model.decoder:forward(comb)
    x_out[i] = rec:clone()
  end
  image.display(join:forward(comb_print))
  image.display(join:forward(x_out))
end

function testMany(ind)
  local disp_ind={}
  for j = 1,params.batch_size do
    table.insert(disp_ind,j)
    table.insert(disp_ind,j+params.batch_size)
    table.insert(disp_ind,j+2*params.batch_size)
    table.insert(disp_ind,j+3*params.batch_size)
  end
  for j = 1,#model.networks do
    model.networks[j]:evaluate()
  end
  local h={}
  g_replace_table(model.s[0], model.start_s)
  local x_in = {}
  local x_enc = {}
  local l = {}
  for i = 1, params.n_glimpses do
    x_in[i]=model.data[i][{{ind,ind+params.batch_size-1}}]:cuda()
    x_enc[i]=model.encoders[i]:forward(x_in[i])
    l[i] = model.in_loc[i]:view(1,4):expand(params.batch_size,4)
    h[i], model.s[i] = 
      unpack(model.networks[i]:forward({x_enc[i], l[i], model.s[i-1]}))
  end
  local join = nn.JoinTable(1)
  image.display{image = reorder(join:forward(x_in):split(1),disp_ind),win=painter4}
  for i = 1,6 do
    for j = 1,80 do
      local loc = {0,-1+2*(j/80),0,0}
      local out_loc = torch.Tensor(loc):cuda()
      local x_out = {}
      local comb = model.combine_network:forward({h[params.n_glimpses],out_loc:view(1,4):expand(params.batch_size,4)})
      local rec = model.decoder:forward(comb)
      image.display{image = rec,win=painter3}
    end
  end
  for i = 1,10 do
    for j = 1,40 do
        local loc = {0,-1+2*(i/10),-0.2+.4*(j/40),0}
        local out_loc = torch.Tensor(loc):cuda()
        local x_out = {}
        local comb = model.combine_network:forward({h[params.n_glimpses],out_loc:view(1,4):expand(params.batch_size,4)})
        local rec = model.decoder:forward(comb)
        image.display{image = rec,win=painter3}
    end
    for j = 1,40 do
        local loc = {0,-1+2*(i/10),0.2-.4*(j/40),0}
        local out_loc = torch.Tensor(loc):cuda()
        local x_out = {}
        local comb = model.combine_network:forward({h[params.n_glimpses],out_loc:view(1,4):expand(params.batch_size,4)})
        local rec = model.decoder:forward(comb)
        image.display{image = rec,win=painter3}
    end
  end
end

function reorder(t,i)
  local out = {}
  for j = 1,#t do
    out[j]=t[i[j]]
  end
  return out
end

main()

window3,painter3 = image.window()
window3:show()
window4,painter4 = image.window()
window4:show()

require('trepl')()