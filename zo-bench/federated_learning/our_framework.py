import sys, os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import Framework
import numpy as np
import torch


class ourFramework(Framework):
    def __init__(self, args, task, fed_args):
        super().__init__(args, task, fed_args)
        self.explore_direction = None
        self.local_es_mangnitude_grads = 0
        
    @staticmethod   
    def sample_unit_sphere(d, num_samples):
        """
        Sample points uniformly at random from the (d-1)-dimensional unit sphere.

        Args:
            d (int): Dimension of the Euclidean space.
            num_samples (int): Number of samples to generate.

        Returns:
            np.ndarray: Array of shape (num_samples, d) containing the sampled points.
        """

        direction = np.random.normal(size=(num_samples, d))
        direction *= np.sqrt(d) / np.linalg.norm(direction, axis=1, keepdims=True)
        return direction


    def sample_model_directions(self):
        """
        Sample directions for the parameters of a deep learning model.
        Returns:
            list of np.ndarray: A list of sampled directions, one for each layer in the model.
        """
        self.explore_direction = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                dim = param.numel()
                direction = self.sample_unit_sphere(dim, 1).squeeze(0)
                self.explore_direction[name] = np.reshape(direction, param.shape)

    def agg_model_parameters(self, scalar):
        """
        Update/Aggregate the parameters of a deep learning model by adding scalar * direction.

        Args:
            model (nn.Module): The deep learning model.
            scalar (float): The scalar value to multiply with direction.
        """
        state_dict = self.model.state_dict()
    
        for name in self.names_to_optm:
            state_dict[name].data.add_(scalar * self.explore_direction[name])
            
        self.model.load_state_dict(state_dict)
        


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

    def before_broadcast(self):
        # sample_model_directions
        self.sample_model_directions() # self.directions is set
        self.local_es_mangnitude_grads = 0
        
    def after_local_train(self, weight):
        # gather the mangnitude of the gradients
        assert self.client_id == -1
        
        self.local_es_mangnitude_grads += weight * self.local_es_mangnitude_grad
        
        
        
        