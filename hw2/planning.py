from simple_world_model import SimpleWorldModel
from dreamerV3 import GRPBase
import torch


class Planner(GRPBase):
    """
    Base class for planners. Defines the interface for planning algorithms.
    """
    def __init__(self, cfg=None):
        super(Planner, self).__init__(cfg)

    def update(self, states, actions):
        """
        Update the planner's internal model or policy based on collected states and actions.
        This method can be overridden by planners that learn from data (e.g., PolicyPlanner).
        
        Args:
            states: Tensor of shape (B, state_dim) containing collected states
            actions: Tensor of shape (B, action_dim) containing collected actions
        """
        pass  # Default implementation does nothing
    
    def plan(self, initial_state, return_best_sequence=True):
        """
        Plan action sequences given an initial state.
        
        Args:
            initial_state: Dictionary containing initial state information
            return_best_sequence: If True, returns the best action sequence; else returns action mean
            
        Returns:
            actions: Tensor of shape (horizon, action_dim) with the planned action sequence
            predicted_reward: Float value of the expected cumulative reward for the planned sequence
        """
        raise NotImplementedError("Plan method must be implemented by subclasses")

class CEMPlanner(Planner):
    """
    Cross-Entropy Method (CEM) planner for model-based planning.
    Samples action sequences and uses a world model to find high-reward plans.
    """
    def __init__(self, 
                 world_model,
                 action_dim,
                 cfg):
        """
        Initialize CEM planner.
        
        Args:
            world_model: World model (DreamerV3 or SimpleWorldModel) used for imagining future trajectories
            action_dim: Dimensionality of the action space
            cfg: Configuration object
        """
        # TODO: Part 1.3 - Initialize CEM planner
        ## Set up world model reference and determine if using DreamerV3 or SimpleWorldModel
        super().__init__(cfg)
        self.world_model = world_model
        self.action_dim = action_dim
        self.cfg = cfg
        self.horizon = cfg.planner.horizon
        self.num_samples = cfg.planner.num_samples
        self.num_elites = cfg.planner.num_elites
        self.num_iterations = cfg.planner.num_iterations

    def plan(self, initial_state, return_best_sequence=True):
        """
        Plan action sequences using CEM to maximize predicted rewards.
        
        Args:
            initial_state: Dictionary containing initial state 
                          - For DreamerV3: {'h', 'z', 'z_probs'}
                          - For SimpleWorldModel: {'pose'}
            return_best_sequence: If True, returns best action sequence; else returns action mean
            
        Returns:
            best_actions: Tensor of shape (horizon, action_dim) with the best action sequence
            best_reward: Float value of the sum of predicted rewards for the best sequence
        """
        # TODO: Part 1.3 - Implement CEM planning algorithm
        ## Sample action sequences, evaluate with world model, select elites, update distribution

        # Random actions samples and scale to normalize to -1, 1
        # variable name with s with more than 1 samples


        # TODO if state/pose dim = (B, T, pose_dim)
        if isinstance(self.world_model, SimpleWorldModel):

            pose = initial_state["pose"]
            device = pose.device
            
            mean_actions = torch.zeros(self.horizon, self.action_dim, device=device)
            std_actions = torch.ones(self.horizon, self.action_dim, device=device)

            for iteration in range(self.num_iterations):
                # distribution sampling
                actions = [torch.normal(mean_actions, std_actions, device=device) for _ in range(self.num_samples)]
                actions = torch.stack(actions)
                
                # Vectorized Sampling
                # N = num_samples, H = horizon, A = action_dim
                # mean = mean_actions.unsqueeze(0).expand(self.num_samples, -1, -1)  # (N, H, A)
                # std  = std_actions.unsqueeze(0).expand(self.num_samples, -1, -1)   # (N, H, A)
                # actions = torch.normal(mean, std)  # (N, H, A)
                
                # Normalized actions
                actions = actions.clamp(-1.0, 1.0)
                
                # eval seqs
                rewards = self._evaluate_sequences(initial_state, actions)

                # update mean/std from the elites
                # choose n actions sequences that have the highest rewards (arg max?) then calculate mean/std from that
                _, idx = torch.topk(rewards, k=self.num_elites)
                best_id = torch.argmax(rewards)
                elites = actions[idx]
                mean_actions = elites.mean(dim=0)
                std_actions = elites.std(dim=0) + 1e-6

            # return top action/action mean and top reward/reward mean
            if return_best_sequence: 
                return actions[best_id], rewards[best_id].item()
            else: 
                return mean_actions, rewards.mean()

 
    def _evaluate_sequences(self, initial_state, action_sequences):
        """
        Evaluate a batch of action sequences by rolling them out in the world model.
        
        Args:
            initial_state: Dictionary with initial state (RSSM state for DreamerV3 or pose for SimpleWorldModel)
            action_sequences: Tensor of shape (num_samples, horizon, action_dim)
            
        Returns:
            rewards: Tensor of shape (num_samples,) with sum of predicted rewards
        """
        # TODO: Part 1.3 - Route to appropriate evaluation method
        ## Determine if using DreamerV3 or SimpleWorldModel and call appropriate method
        if type(self.world_model).__name__ == "DreamerV3": 
            return self._evaluate_sequences_dreamer(initial_state, action_sequences)
        elif type(self.world_model).__name__ == "SimpleWorldModel": 
            return self._evaluate_sequences_simple(initial_state, action_sequences)
    
    def _evaluate_sequences_dreamer(self, initial_state, action_sequences):
        """
        Evaluate sequences using DreamerV3 RSSM-based rollout.
        """
        # TODO: Part 3.3 - Implement CEM planning with DreamerV3
        ## Roll out action sequences in the DreamerV3 world model and compute total rewards
        pass
    
    def _evaluate_sequences_simple(self, initial_state, action_sequences):
        """
        Evaluate sequences using SimpleWorldModel pose-based rollout.
        """
        # TODO: Part 1.3 - Implement CEM planning with SimpleWorldModel
        ## Roll out action sequences using SimpleWorldModel and compute total rewards
        
        # initial pose can only be (1, pose_dim) or (pose_dim, ) because 1 kind of pose only at first
        # action sequence will be (B, T, action_dim) where B is batch, T is time/horizon
        rollout_fn = self.world_model.forward
        pose = initial_state['pose']
        device = pose.device
        
        # matching (pose_dim, ) to (1, pose_dim)
        if pose.dim() == 1: 
            pose = pose.unsqueeze(0)
        
        # batching the initial pose (1, pose_dim) to (B, pose_dim) and reward (B, 1)
        current_pose = pose.expand(self.num_samples, -1)
        cumul_rewards = torch.zeros(self.num_samples, 1, device=device)
        
        for t in range(self.horizon):
            # is this (B, 1, action_dim) or (B, action_dim)? Im leaning on the former. 
            # If former, the shape will mismatch, and I will need to update the first function again.
            # Update: its (B, action_dim)
            action_t = action_sequences[:, t, :]
            
            # input pose: (B, pose_dim); action: (B, action_dim)
            # output next_pose: (B, pose_dim), reward: (B, 1)
            # is it sufficient to distinguish in my og function?
            next_pose, reward_t = rollout_fn(current_pose, action_t)

            cumul_rewards += reward_t
            current_pose = next_pose
            
        return cumul_rewards.squeeze(-1)
            
    
    def forward(self, observations=None, prev_actions=None, prev_state=None,
                mask_=True, pose=None, last_action=None,
                text_goal=None, goal_image=None, return_full_sequence=False):
        """
        Unified interface for planning that works with both DreamerV3 and SimpleWorldModel.
        This wrapper obtains the current state and plans actions.
        
        Args:
            observations: Tensor of shape (B, T, C, H, W) - input observations (for DreamerV3)
            prev_actions: Previous actions (optional, for state initialization)
            prev_state: Previous state (optional)
            mask_: Mask parameter (kept for API compatibility)
            pose: Pose information (B, pose_dim) - for SimpleWorldModel
            last_action: Last action taken (kept for API compatibility)
            text_goal: Text goal (kept for API compatibility)
            goal_image: Goal image (kept for API compatibility)
            return_full_sequence: If True, returns full planned sequence; else just first action
            
        Returns:
            Dictionary containing:
                - 'actions': Planned action(s) (B, action_dim) or (B, horizon, action_dim)
                - 'predicted_reward': Expected cumulative reward
                - 'final_state': Final state after processing inputs
        """
        # TODO: Part 1.3 - Route forward pass to appropriate model
        ## Determine if using DreamerV3 or SimpleWorldModel and call appropriate method
        
        if isinstance(self.world_model, SimpleWorldModel):
            return self._forward_simple_cem(pose, return_full_sequence)
        else:
            # DreamerV3 path (Part 3)
            return self._forward_dreamer(
                observations, prev_actions, prev_state, return_full_sequence
            )
    
    def _forward_simple_cem(self, pose, return_full_sequence):
        """Forward for CEM + SimpleWorldModel: plan from current pose and return actions."""
        if pose is None:
            raise ValueError("CEMPlanner with SimpleWorldModel requires pose.")
        # pose: (B, pose_dim), often B=1
        initial_state = {"pose": pose}
        best_actions, best_reward = self.plan(initial_state, return_best_sequence=True)
        # best_actions: (horizon, action_dim), best_reward: float
        if return_full_sequence:
            actions = best_actions.unsqueeze(0)  # (1, horizon, action_dim)
        else:
            actions = best_actions[0:1]         # (1, action_dim) — first step only
        reward_val = best_reward if isinstance(best_reward, float) else best_reward.item()
        return {
            "actions": actions,
            "predicted_reward": reward_val,
            "final_state": initial_state,
        }
    
    def _forward_dreamer(self, observations, prev_actions, prev_state, return_full_sequence):
        """Forward pass for DreamerV3 model."""
        # TODO: Part 4.2 - Implement DreamerV3 forward pass for policy
        ## Encode observations, roll through RSSM, and plan with policy from current state
        pass

        # [Imagine method remains mostly the same, ensuring valid input shapes for heads]
    def preprocess_state(self, image):
        """Preprocess observation image"""
        # TODO: Preprocess image for input
        ## Resize, normalize, and convert to channel-first format
        pass


class PolicyPlanner(GRPBase):
    """
    Policy-based planner that uses a trained policy model to generate action sequences.
    Rolls out the policy over a horizon by predicting actions and states at each timestep.
    """
    def __init__(self, 
                 world_model,
                 policy_model,
                 action_dim,
                 cfg=None,
                 horizon=None):
        """
        Initialize Policy planner.
        
        Args:
            world_model: World model (DreamerV3 or SimpleWorldModel) used for predicting future states
            policy_model: Trained policy model that predicts actions given states
            action_dim: Dimensionality of the action space
            cfg: Configuration object
            horizon: Planning horizon (number of timesteps to plan ahead)
        """
        # TODO: Part 2.2 - Initialize Policy planner
        ## Set up world model, policy model, optimizer, and scheduler
        pass

    def update(self, states, actions):
        """
        Docstring for update
        Update the policy model using collected states and actions.
        
        :param self: Description
        :param states: Description
        :param actions: Description
        """
        # TODO: Part 2.2 - Implement policy training
        ## Train the policy using behavior cloning on collected state-action pairs
        pass

    
    def plan(self, initial_state, return_best_sequence=True):
        """
        Plan action sequences by rolling out the policy model over the horizon.
        
        Args:
            initial_state: Dictionary containing initial state 
                          - For DreamerV3: {'h', 'z', 'z_probs'}
                          - For SimpleWorldModel: {'pose'}
            return_best_sequence: If True, returns the planned sequence (unused here for consistency)
            
        Returns:
            actions: Tensor of shape (horizon, action_dim) with the planned action sequence
            total_reward: Float value of the sum of predicted rewards
        """
        # TODO: Part 2.2 - Implement policy rollout planning
        ## Roll out the policy over the horizon, predicting actions and accumulating rewards
        pass
    
    def forward(self, observations=None, prev_actions=None, prev_state=None,
                mask_=True, pose=None, last_action=None,
                text_goal=None, goal_image=None, return_full_sequence=False):
        """
        Unified interface for planning that works with both DreamerV3 and SimpleWorldModel.
        This wrapper obtains the current state and plans actions using the policy.
        
        Args:
            observations: Tensor of shape (B, T, C, H, W) - input observations (for DreamerV3)
            prev_actions: Previous actions (optional, for state initialization)
            prev_state: Previous state (optional)
            mask_: Mask parameter (kept for API compatibility)
            pose: Pose information (B, pose_dim) - for SimpleWorldModel
            last_action: Last action taken (kept for API compatibility)
            text_goal: Text goal (kept for API compatibility)
            goal_image: Goal image (kept for API compatibility)
            return_full_sequence: If True, returns full planned sequence; else just first action
            
        Returns:
            Dictionary containing:
                - 'actions': Planned action(s) (B, action_dim) or (B, horizon, action_dim)
                - 'predicted_reward': Expected cumulative reward
                - 'final_state': Final state after processing inputs
        """
        # TODO: Part 2.2 - Route forward pass to appropriate model
        ## Determine if using DreamerV3 or SimpleWorldModel and call appropriate method
        pass
    
    def _forward_dreamer(self, observations, prev_actions, prev_state, return_full_sequence):
        """Forward pass for DreamerV3 model."""
        # TODO: Part 4.2 - Implement DreamerV3 forward pass for policy
        ## Encode observations, roll through RSSM, and plan with policy from current state
        pass
    
    def _forward_simple(self, pose, return_full_sequence):
        """Forward pass for SimpleWorldModel."""
        # TODO: Part 2.2 - Implement SimpleWorldModel forward pass for policy
        ## Plan from current pose using policy with SimpleWorldModel
        pass


class RandomPlanner(GRPBase):
    """
    Random action planner that generates random actions uniformly distributed between -1 and 1.
    Useful as a baseline for comparing planning algorithms.
    """
    def __init__(self, 
                 action_dim,
                 cfg):
        """
        Initialize Random planner.
        
        Args:
            world_model: World model (optional, not used but kept for API compatibility)
            action_dim: Dimensionality of the action space (default: 7)
            cfg: Configuration object (optional)
            horizon: Planning horizon (number of timesteps to plan ahead)
        """
        super(RandomPlanner, self).__init__(cfg)
        
        self.action_dim = action_dim
            
    def forward(self, observations=None, prev_actions=None, prev_state=None,
                mask_=True, pose=None, last_action=None,
                text_goal=None, goal_image=None, return_full_sequence=False):
        """
        Unified interface for planning that generates random actions.
        
        Args:
            observations: Tensor of shape (B, T, C, H, W) - input observations (optional)
            prev_actions: Previous actions (optional)
            prev_state: Previous state (optional)
            mask_: Mask parameter (kept for API compatibility)
            pose: Pose information (B, pose_dim) - for SimpleWorldModel
            last_action: Last action taken (kept for API compatibility)
            text_goal: Text goal (kept for API compatibility)
            goal_image: Goal image (kept for API compatibility)
            return_full_sequence: If True, returns full planned sequence; else just first action
            
        Returns:
            Dictionary containing:
                - 'actions': Random action(s) (B, action_dim) or (B, horizon, action_dim)
                - 'predicted_reward': 0.0 (no prediction for random actions)
                - 'final_state': None or dummy state
        """
        ## compute random actions
        actions = torch.rand((1, self.action_dim), device=pose.device) * 2 - 1  # (1, action_dim) in range [-1, 1]
        
        return {
            'actions': actions,
            'predicted_reward': 0.0,
            'final_state': prev_state if prev_state is not None else None
        }