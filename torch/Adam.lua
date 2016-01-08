local Adam = torch.class('nn.Adam')

function Adam:__init(params,gparams,alpha,beta1,beta2,epsilon,l)
	self.params = params
	self.gparams = gparams
	self.alpha = alpha or 0.0002
	self.beta1 = beta1 or 0.1
	self.beta2 = beta2 or 0.001
	self.epsilon = epsilon or 1e-8
	self.l = l or 1e-8
	self.t = 0
	self.m = params:clone():zero()
	self.v = params:clone():zero()
end

function Adam:step()
	self.t = self.t+1
	local b1_t = self.beta1*math.pow(self.l,self.t-1)
	self.m:mul(b1_t)
	self.m:add(1-b1_t,self.gparams)
	self.v:mul(self.beta2)
	self.v:addcmul(1-self.beta2,self.gparams,self.gparams)
	local m_bias = torch.div(self.m,1-b1_t)
	local v_bias = torch.div(self.v,1-self.beta2)
	self.params:addcdiv(-self.alpha,m_bias,torch.sqrt(v_bias) + self.epsilon)
	--self.params:add(-self.gparams*self.alpha)
end

