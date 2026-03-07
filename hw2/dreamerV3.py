from tensorflow.python.ops.gen_batch_ops import batch
from networkx import kneser_graph
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical, Independent
import numpy as np

def symlog(x):
    """
    Symmetric log transformation.
    Squashes large values while preserving sign and small values.
    y = sign(x) * ln(|x| + 1)
    """
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

class GRPBase(nn.Module):
    """Base class for GRP models"""
    def __init__(self, cfg):
        super(GRPBase, self).__init__()
        self._cfg = cfg

    def encode_text_goal(self, goal, tokenizer=None, text_model=None):
        import numpy as _np
        import torch as _torch
        if self._cfg.dataset.encode_with_t5:
            if tokenizer is None or text_model is None:
                raise ValueError("tokenizer and text_model must be provided when using T5 encoding")
            # TODO:    
            ## Provide the logic converting text goal to T5 embedding tensor
            pass
        else:
            pad = " " * self._cfg.max_block_size
            goal_ = goal[:self._cfg.max_block_size] + pad[len(goal):self._cfg.max_block_size]
            try:
                stoi = {c: i for i, c in enumerate(self._cfg.dataset.chars_list)}
                ids = [stoi.get(c, 0) for c in goal_]
            except Exception:
                ids = [0] * self._cfg.max_block_size
            return _torch.tensor(_np.expand_dims(_np.array(ids, dtype=_np.int64), axis=0), dtype=_torch.long, device=self._cfg.device)

    def process_text_embedding_for_buffer(self, goal, tokenizer=None, text_model=None):
        """
        Process text goal embedding for storing in the circular buffer.
        Returns a numpy array of shape (max_block_size, n_embd) without batch dimension.
        """
        import numpy as _np
        if tokenizer is None or text_model is None:
            raise ValueError("tokenizer and text_model must be provided when using T5 encoding")
        
        goal_ = _np.zeros((self._cfg.max_block_size, self._cfg.n_embd), dtype=_np.float32)
        input_ids = tokenizer(goal, return_tensors="pt").input_ids
        goal_t = text_model.encoder(input_ids).last_hidden_state.detach().cpu().numpy()
        goal_[:len(goal_t[0]), :] = goal_t[0][:self._cfg.max_block_size]
        return goal_


    def resize_image(self, image):
        """Resize image to match model input size"""
        import cv2
        import numpy as _np
        img = _np.array(image, dtype=_np.float32)
        img = cv2.resize(img, (self._cfg.image_shape[0], self._cfg.image_shape[1]))
        return img

    def normalize_state(self, image):
        """Normalize image to [-1, 1] range"""
        enc = ((image / 255.0) * 2.0) - 1.0
        return enc
    
    def preprocess_state(self, image):
        """Preprocess observation image"""
        img = self.resize_image(image)
        img = self.normalize_state(img)
        return img

    def preprocess_goal_image(self, image):
        """Preprocess goal image"""
        return self.preprocess_state(image)

    def decode_action(self, action_tensor):
        """Decode normalized actions to original action space"""
        import torch as _torch
        action_mean = _torch.tensor(np.repeat([self._cfg.dataset.action_mean], self._cfg.policy.action_stacking, axis=0).flatten(), 
                                   dtype=action_tensor.dtype, device=action_tensor.device)
        action_std = _torch.tensor(np.repeat([self._cfg.dataset.action_std], self._cfg.policy.action_stacking, axis=0).flatten(), 
                                  dtype=action_tensor.dtype, device=action_tensor.device)
        return (action_tensor * (action_std)) + action_mean

    def encode_action(self, action_float):
        """Encode actions to normalized space [-1, 1]"""
        import torch as _torch
        ## If the action_float has length greater than action_dim then use stacking otherwise just use normal standardiaztion vectors
        if action_float.shape[1] == len(self._cfg.dataset.action_mean):
            action_mean = _torch.tensor(self._cfg.dataset.action_mean, dtype=action_float.dtype, device=action_float.device)
            action_std = _torch.tensor(self._cfg.dataset.action_std, dtype=action_float.dtype, device=action_float.device)
            return (action_float - action_mean) / (action_std)  

        action_mean = _torch.tensor(np.repeat([self._cfg.dataset.action_mean], self._cfg.policy.action_stacking, axis=0).flatten(), 
                                   dtype=action_float.dtype, device=action_float.device)
        action_std = _torch.tensor(np.repeat([self._cfg.dataset.action_std], self._cfg.policy.action_stacking, axis=0).flatten(), 
                                  dtype=action_float.dtype, device=action_float.device)
        return (action_float - action_mean) / (action_std)
    
    def decode_pose(self, pose_tensor):
        """
        Docstring for decode_pose
        
        :param self: Description
        :param pose_tensor: Description
        self._decode_state = lambda sinN: (sinN * state_std) + state_mean  # Undo mapping to [-1, 1]
        """
        import torch as _torch
        pose_mean = _torch.tensor(self._cfg.dataset.pose_mean, dtype=pose_tensor.dtype, device=pose_tensor.device)
        pose_std = _torch.tensor(self._cfg.dataset.pose_std, dtype=pose_tensor.dtype, device=pose_tensor.device)
        return (pose_tensor * (pose_std)) + pose_mean
    
    def encode_pose(self, pose_float):
        """
        Docstring for encode_pose
        
        :param self: Description
        :param pose_float: Description
        self._encode_pose = lambda pf:   (pf - pose_mean)/(pose_std) # encoder: take a float, output an integer
        """
        import torch as _torch
        pose_mean = _torch.tensor(self._cfg.dataset.pose_mean, dtype=pose_float.dtype, device=pose_float.device)
        pose_std = _torch.tensor(self._cfg.dataset.pose_std, dtype=pose_float.dtype, device=pose_float.device)
        return (pose_float - pose_mean) / (pose_std)

class DreamerV3(GRPBase):
    def __init__(self, 
                 obs_shape=(3, 128, 128),  # Updated default to match your error # (C, H, W )
                 action_dim=6, 
                 stoch_dim=32, 
                 discrete_dim=32, 
                 deter_dim=512, 
                 hidden_dim=512, cfg=None):
        # TODO: Part 3.1 - Initialize DreamerV3 architecture
        ## Define encoder, RSSM components (GRU, prior/posterior nets), and decoder heads
        # Save argument first
        super().__init__(cfg)
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.stoch_dim = stoch_dim # nb of independent categorical variables make up z_t
        self.discrete_dim = discrete_dim # classes each categorical chooses among
        self.deter_dim = deter_dim # size of h_t
        self.hidden_dim = hidden_dim  # width of MLP knob
        self.embed_dim = hidden_dim # TODO: shouldn't this be a parameter?
        self.state_dim = deter_dim + stoch_dim * discrete_dim
        
        # Big net inside
        class CNNEncoder(nn.Module):
            def __init__(self, obs_shape, kernel_size, stride, embed_dim): 
                super().__init__()
                C, H, W = obs_shape
                out_dim = lambda w, k, p, s: (w - k + 2 * p) // s + 1
                
                out1 = out_dim(max(H, W), kernel_size, 0, stride)
                out2 = out_dim(out1, kernel_size, 0, stride)
                out3 = out_dim(out2, kernel_size, 0, stride)
                
                self.conv1 = nn.Conv2d(C, 32, kernel_size, stride)
                self.conv2 = nn.Conv2d(32, 64, kernel_size, stride)
                self.conv3 = nn.Conv2d(64, 128, kernel_size, stride)
                self.flatten = nn.Flatten(1)
                self.fc = nn.Linear(128 * out3 * out3, embed_dim)
            
            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = F.relu(self.conv2(x))
                x = F.relu(self.conv3(x))
                x = self.flatten(x)
                return self.fc(x)
            
            
        class CNNDecoder(nn.Module):
            def __init__(self, obs_shape, kernel_size, stride, state_dim):
                super().__init__()
                C, _, _ = obs_shape
                
                self.fc = nn.Linear(state_dim, 128 * 4 * 4)
                # H, W out formula: H_out ​= (H_in​−1) * stride[0] − 2×padding[0] + dilation[0]×(kernel_size[0]−1) + output_padding[0] + 1
                # Layer 1 = (4-1)*2 - 2*1 + 1*(4-1) + 0 + 1 = 8
                # Layer 2 = (8-1)*2 - 2*1 + 1*(4-1) + 0 + 1 = 16
                # Layer 3 = (16-1)*2 - 2*1 + 1*(4-1) + 0 + 1 = 32
                # Layer 4 = (32-1)*2 - 2*1 + 1*(4-1) + 0 + 1 = 64
                # Layer 5 = (64-1)*2 - 2*1 + 1*(4-1) + 0 + 1 = 128
                self.conv1t = nn.ConvTranspose2d(128, 64, kernel_size, stride, padding=1)  
                self.conv2t = nn.ConvTranspose2d(64, 32, kernel_size, stride, padding=1)  
                self.conv3t = nn.ConvTranspose2d(32, 16, kernel_size, stride, padding=1) 
                self.conv4t = nn.ConvTranspose2d(16, 8, kernel_size, stride, padding=1)  
                self.conv5t = nn.ConvTranspose2d(8, 4, kernel_size, stride, padding=1)     
                self.tanh = nn.Tanh()
                
            def forward(self, x):
                import torch.nn.functional as F
                B = x.shape[0]
                x = self.fc(x)
                x = torch.reshape(x, (B, 128, 4, 4))
                x = F.relu(self.conv1t(x))
                x = F.relu(self.conv2t(x))
                x = F.relu(self.conv3t(x))
                x = F.relu(self.conv4t(x))
                x = F.relu(self.conv5t(x))
                return self.tanh(x)
                
        kernel_size = 4             
        stride = 2
        # Input: imgs = (B, C, H, W)  where C, H, W come from obs_shape
        # Output: embeds = (B, embed_dim)
        self.encoder = CNNEncoder(obs_shape=obs_shape, kernel_size=kernel_size, stride=stride, embed_dim=self.embed_dim)
        
        # RSSM Component: GRU and prior/post MLP
        # Input: h_prev = (B, deter_dim), z_prev = (B, stoch_dim * discrete_dim), a_prev = (B, action_dim)
        # Input: x_prev = concat(a_prev, z_prev) = (B, action_dim + stoch_dim * discrete_dim)
        # Input: hidden_dim: hyperparam that determine # of features of the network
        # Output: h_current = (B, deter_dim)
        # Batch first = false, careful because (B) will not be the first dim!
        self.gru = nn.GRU(input_size=(self.stoch_dim * self.discrete_dim + self.action_dim), hidden_size=self.deter_dim, batch_first=True)
        
        # Input prior: h_current = (B, deter_dim)
        # Input post: h_current = (B, deter_dim) AND embeds (B, embed_dim)
        # Output: prior/post_logit = (B, stoch_dim * discrete_dim), same size as z_current!
        self.priorMLP = nn.Sequential(nn.Linear(self.deter_dim, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, self.stoch_dim * self.discrete_dim))
        self.postMLP = nn.Sequential(nn.Linear(self.deter_dim + self.embed_dim, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, self.stoch_dim * self.discrete_dim))
        
        # Input: concat(z_current, h_current) = (B, state_dim)
        # Output decoder: reconstruct_imgs = (B, C, H, W)?
        # Output reward, continue: rews, continues = (B, 1)
        self.decoder = CNNDecoder(obs_shape, kernel_size=kernel_size, stride=stride, state_dim=self.state_dim)
        self.rewardMLP = nn.Sequential(nn.Linear(self.state_dim, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, 1))
        self.continueMLP = nn.Sequential(nn.Linear(self.state_dim, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, 1))

    # ... [Helper methods same as before] ...

    def get_initial_state(self, batch_size, device):
        return {
            'h': torch.zeros(batch_size, self.deter_dim, device=device),
            'z': torch.zeros(batch_size, self.stoch_dim * self.discrete_dim, device=device),
            'z_probs': torch.zeros(batch_size, self.stoch_dim, self.discrete_dim, device=device)
        }

    def sample_stochastic(self, logits, training=True):
        # TODO: Part 3.1 - Implement stochastic sampling
        ## Sample from discrete categorical distribution using logits
        # The softmax must be applied per slot (each group of discrete_dim logits is its own categorical).
        # Input: prior/post_logit = (B, stoch_dim * discrete_dim), same size as z_current!
        logits = torch.reshape(logits, (-1, self.stoch_dim, self.discrete_dim))
        
        # Output z_soft: same shape as logits (B, stoch_dim, discrete_dim)
        # Output arg_max = (B, stoch_dim)
        # Output one_hot = (B, stoch_dim, discrete_dim)
        z_soft = F.softmax(logits, dim=-1)
        z_hard = F.one_hot(torch.argmax(logits, dim=-1), num_classes = self.discrete_dim).float()
    
        # reshape because h_t have shape (B, self.deter_dim)
        if not training:
            return z_hard.reshape(-1, self.stoch_dim * self.discrete_dim)   # pure discrete, no gradient tricks needed
        
        z_hat = z_hard - z_soft.detach() + z_soft   # straight-through for training
        return z_hat.reshape(-1, self.stoch_dim * self.discrete_dim) 

    def rssm_step(self, prev_state, action, embed=None):
        # TODO: Part 3.1 - Implement RSSM step
        ## Update deterministic state (h) with GRU, compute prior and posterior distributions
        # Input: prev_state = {"h", "z", "z_prob"} (from get_init) , action = a_prev = (B, action_dim), embed (for postMLP) = (B, embed_dim)
        # Prior input = h_prev; Post input = concat(h_prev, embed)
        h_prev, z_prev = prev_state["h"], prev_state["z"]
        x_prev = torch.concat((z_prev, action), dim=-1)
        
        # Input GRU: concat(z_prev, a_prev), h_prev
        # Input GRU = (B, seq_len, input_size) => x_prev = (B, input_size), x_prev.unsqueeze(1) to add time dim
        # Input GRU h_0 = (num_layers, B, hidden_size) => h_prev = (B, deter_dim), h_prev.unsqueeze(0) to add num_layer dim
        # Output GRU = (output, h_n), don't use output but h_n
        # Outpu GRU: h_cur = (num_layers, B, hidden_size), h_cur.squeeze(0) to become (B, deter_dim)
        _, h_cur = self.gru(x_prev.unsqueeze(1), h_prev.unsqueeze(0))
        h_cur = h_cur.squeeze(0)   
        
        prior_logit = self.priorMLP(h_cur)
        if embed is None: 
            # Imagination/no observation case
            post_logit = prior_logit
        else: 
            post_input = torch.concat((h_cur, embed), dim=-1)
            post_logit = self.postMLP(post_input)
        
        # self.training is from nn.Module, = True on calling train()
        z_cur = self.sample_stochastic(post_logit, training=self.training)
        z_prob = F.softmax(torch.reshape(post_logit, (-1, self.stoch_dim, self.discrete_dim)), dim=-1)
        return {
            "h": h_cur, 
            "z": z_cur,
            "post_logit": post_logit, 
            "prior_logit": prior_logit, 
            "z_prob": z_prob
        }

    def forward(self, observations, prev_actions=None, prev_state=None,
                mask_=True, pose=None, last_action=None,
                text_goal=None, goal_image=None):
        # TODO: Part 3.2 - Implement DreamerV3 forward pass
        ## Encode images, unroll RSSM, and compute reconstructions and heads
        # observations = (B, T, H, W, C)
        device = observations.device
        B, T = observations.shape[0], observations.shape[1]
        prev_state = self.get_initial_state(batch_size=B, device=device) if prev_state is None else prev_state
        
        reconstructions = []   # decoder output at each t
        rewards_hat = []       # rewardMLP output at each t
        continues_hat = []     # continueMLP output at each t
        prior_logits = []      # prior_logit from each rssm_step
        post_logits = []       # post_logit from each rssm_step
        
        for t in range(T):
            obs = torch.permute(observations[:, t], (0, 3, 1, 2))
            action = prev_actions[:, t]
            embd_obs = self.encoder(obs)
            # reset at the beginning of the loop
            prev_state = self.rssm_step(prev_state, action, embd_obs)
            
            x = torch.cat((prev_state["z"], prev_state["h"]), dim=-1)
            reconstructions.append(self.decoder(x))
            rewards_hat.append(self.rewardMLP(x))
            continues_hat.append(self.continueMLP(x))
            prior_logits.append(prev_state['prior_logit'])
            post_logits.append(prev_state['post_logit'])
        
        return {
            "reconstructions": torch.stack(reconstructions, dim=1), 
            "rewards": torch.stack(rewards_hat, dim=1),
            "continues": torch.stack(continues_hat, dim=1),
            "prior_logits": torch.stack(prior_logits, dim=1),
            "post_logits": torch.stack(post_logits, dim=1)
        }

    # [Imagine method remains mostly the same, ensuring valid input shapes for heads]
    def preprocess_state(self, image):
        """Preprocess observation image"""
        img = self.resize_image(image)
        img = self.normalize_state(img)
        ## Change numpy array from channel-last to channel-first
        img = np.transpose(img, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        # img = img.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        return img
    
    def compute_loss(self, output, images, rewards, dones, device):
        """
        Compute the total loss for DreamerV3 model training.
        
        Args:
            output: Dictionary containing model outputs (reconstructions, rewards, continues, priors_logits, posts_logits)
            images: Ground truth images tensor
            rewards: Ground truth rewards tensor
            dones: Ground truth done flags tensor
            device: Device to perform computations on
            pred_coeff: Coefficient for prediction losses (reconstruction + reward + continue)
            dyn_coeff: Coefficient for dynamics loss
            rep_coeff: Coefficient for representation loss
        
        Returns:
            Dictionary containing:
                - total_loss: Combined weighted loss
                - recon_loss: Reconstruction loss
                - reward_loss: Reward prediction loss
                - continue_loss: Continue prediction loss
                - dyn_loss: Dynamics loss (KL divergence)
                - rep_loss: Representation loss (KL divergence)
        """
        # TODO: Part 3.2 - Implement DreamerV3 loss computation
        ## Compute reconstruction, reward, KL divergence losses and combine them
        images = torch.permute(images, (0, 1, 4, 2, 3))
        if rewards.dim() == 2: 
            rewards = rewards.unsqueeze(-1)
            
        if dones.dim() == 2: 
            dones = dones.unsqueeze(-1).float()
        
        L_recon = F.mse_loss(output['reconstructions'], images)
        L_reward = F.mse_loss(output['rewards'], symlog(rewards))
        L_continue = F.binary_cross_entropy_with_logits(output["continues"].float(), (1 - dones))
        L_pred = L_recon + L_reward + L_continue
        
        post_logits = output["post_logits"].reshape((-1, self.stoch_dim, self.discrete_dim))
        prior_logits = output["prior_logits"].reshape((-1, self.stoch_dim, self.discrete_dim))
        L_dyn = F.kl_div(F.log_softmax(prior_logits, dim=-1), F.softmax(post_logits, dim=-1).detach(), reduction="batchmean")
        L_rep = F.kl_div(F.log_softmax(prior_logits, dim=-1).detach(), F.softmax(post_logits, dim=-1), reduction="batchmean")
        L_tot = L_pred + 0.5 * L_dyn + 0.1 * L_rep
        
        return {
            "total_loss": L_tot, 
            "recon_loss": L_recon, 
            "reward_loss": L_reward,
            "continue_loss": L_continue, 
            "dyn_loss": L_dyn, 
            "rep_loss": L_rep
        }
        

