import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dreamerV3 import GRPBase


class SimpleWorldModel(GRPBase):
    """
    Simple world model that predicts the next pose and reward given current pose and action.
    
    Architecture:
    - Takes current pose (7-d) + normalized action
    - Simple MLP to predict next pose (7-d) and reward (scalar)
    """
    
    def __init__(self, 
                 action_dim=7,
                 pose_dim=7,
                 hidden_dim=256,
                 cfg=None):
        # TODO: Part 1.1 - Initialize SimpleWorldModel architecture
        ## Define the feature network and output heads (pose and reward)
        super().__init__(cfg)

        in_dim = action_dim + pose_dim

        self.features = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), 
            nn.ReLU(), 
        )

        self.pose_head = nn.Linear(hidden_dim, pose_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, pose, action):
        """
        Forward pass to predict next pose and reward.
        
        Args:
            pose: Pose tensor of shape (B, pose_dim) or (B, T, pose_dim), normalized
            action: Action tensor of shape (B, action_dim) or (B, T, action_dim), normalized
        
        Returns:
            next_pose_pred: Predicted normalized pose (B, pose_dim) or (B, T, pose_dim)
            reward_pred: Predicted reward (B, 1) or (B, T, 1)
        """
        # TODO: Part 1.1 - Implement forward pass
        ## Concatenate pose and action, pass through feature network and output heads
        # X shape (B, T, in_dim) or (B, in_dim)
        x = torch.cat((pose.float(), action.float()), axis=-1)

        pose_dim = pose.shape[-1]
        is_seq = x.dim() == 3 
        
        # reshape to fit network
        if is_seq: 
            B, T, F = x.shape
            x = torch.reshape(x, (B * T, F))
            
        # fwd pass
        features = self.features(x)
        pose_pred = self.pose_head(features)
        reward_pred = self.reward_head(features)
        
        # reshape back
        if is_seq: 
            pose_pred = torch.reshape(pose_pred, (B, T, pose_dim))
            reward_pred = torch.reshape(reward_pred, (B, T, 1))

        return pose_pred, reward_pred
    
    def predict_next_pose(self, pose, action):
        """
        Convenience method to predict next pose and reward, decoding pose to original space.
        
        Args:
            pose: Normalized pose (7-d vector or batch)
            action: Action in original space (will be encoded)
        
        Returns:
            next_pose: Pose in original space
            reward: Predicted reward
        """
        # TODO: Part 1.1 - Implement prediction method
        ## Encode action, call forward, and decode pose to original space
        
        # Possible dim action/dim pose == 3 like in forward
        action = self.encode_action(action)
        next_pose, reward = self.forward(pose, action)
        next_pose = self.decode_pose(next_pose)
        return next_pose, reward
    
    def compute_loss(self, pose, action, target_pose, target_reward=None):
        """
        Compute MSE loss between predicted and target pose and reward.
        
        Args:
            pose: Current pose tensor (B, pose_dim) or (B, T, pose_dim), normalized
            action: Action tensor (B, action_dim) or (B, T, action_dim), normalized
            target_pose: Target pose tensor (B, pose_dim) or (B, T, pose_dim), normalized
            target_reward: Target reward tensor (B, 1) or (B, T, 1), optional
        
        Returns:
            loss: Total MSE loss (pose + reward if target_reward is provided)
        """
        # TODO: Part 1.2 - Implement SimpleWorldModel loss computation
        ## Compute MSE loss for pose and reward predictions
        pred_pose, pred_reward = self.forward(pose, action)
        pose_loss = F.mse_loss(pred_pose, target_pose.float())
        
        if target_reward is not None:
            reward_loss = F.mse_loss(pred_reward, target_reward.float())
            return reward_loss + pose_loss
        else: 
            return pose_loss
