import sys, os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import Framework
import numpy as np
import torch


class ourFramework(Framework):
    def __init__(self, args, task, fed_args):
        super().__init__(args, task, fed_args)
        self.local_es_mangnitude_grads = 0
        
    # @staticmethod   
    # def sample_unit_sphere(d, num_samples):
    #     """
    #     Sample points uniformly at random from the (d-1)-dimensional unit sphere.

    #     Args:
    #         d (int): Dimension of the Euclidean space.
    #         num_samples (int): Number of samples to generate.

    #     Returns:
    #         np.ndarray: Array of shape (num_samples, d) containing the sampled points.
    #     """

    #     direction = np.random.normal(size=(num_samples, d))
    #     direction *= np.sqrt(d) / np.linalg.norm(direction, axis=1, keepdims=True)
    #     return direction


    # def sample_model_directions(self):
    #     """
    #     Sample directions for the parameters of a deep learning model.
    #     Returns:
    #         list of np.ndarray: A list of sampled directions, one for each layer in the model.
    #     """
    #     self.explore_direction = {}
    #     for name, param in self.model.named_parameters():
    #         if param.requires_grad:
    #             dim = param.numel()
    #             direction = self.sample_unit_sphere(dim, 1).squeeze(0)
    #             self.explore_dire
    def before_broadcast(self, current_round):
        # sample_model_directions
        self.global_seed = current_round

    def agg_model_parameters(self, scalar):
        """
        Update/Aggregate the parameters of a deep learning model by adding scalar * direction.

        Args:
            model (nn.Module): The deep learning model.
            scalar (float): The scalar value to multiply with direction.
        """
        state_dict = self.model.state_dict()
        torch.manual_seed(self.global_seed) 
        
        for name in self.names_to_optm:
            param = state_dict[name]
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            # normalize the direction
            z  /= z.norm()
            # sgd update
            state_dict[name].grad = z * scalar
            self.trainer.optimizer.step()
            self.trainer.optimizer.zero_grad()
                
        
        


    # def update_model_parameters(self, scalars, iterations, seed):
    #     np.random.seed(seed)

    #     for r in range(iterations):
    #         self._update_model_parameters(scalars[r])
                    
    # def _reset_model_parameters(self, scalar):
    #     """
    #     Reset the parameters of a deep learning model to their original values.

    #     Args:
    #         model (nn.Module): The deep learning model.
    #         scalar (float): The scalar value to multiply with direction.
    #     """
    #     directions = self.sample_model_directions()
    #     with torch.no_grad():
    #         for name, param in self.model.named_parameters():
    #             if param.requires_grad:
    #                 param -= scalar * directions[name]

    # def rest_model_parameters(self, scalars, iterations, seed):

    #     np.random.seed(seed)
        
    #     for r in range(iterations):
    #         self._reset_model_parameters(scalars[r])

        
    def after_local_train(self, weight):
        # gather the mangnitude of the gradients
        assert self.client_id == -1
        
        self.local_es_mangnitude_grads += weight * self.local_es_mangnitude_grad
        
        
        
        