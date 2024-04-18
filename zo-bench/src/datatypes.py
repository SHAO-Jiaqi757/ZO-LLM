from collections import OrderedDict

import torch

class NamedParametersToOptimize(OrderedDict):
    """
    A custom data type that extends OrderedDict to represent named parameters to optimize.
    Supports the + operator to combine multiple NamedParametersToOptimize objects.
    """
    def __add__(self, other):
        """
        Overload the + operator to combine two NamedParametersToOptimize objects.
        
        Args:
            other (NamedParametersToOptimize): The other NamedParametersToOptimize object to add.
            
        Returns:
            NamedParametersToOptimize: A new NamedParametersToOptimize object that is the combination of the two input objects.
        """
        result = NamedParametersToOptimize(self)
        for name, param in other.items():
            result[name] = self[name] + param if name in self else param
        return result
    def __mul__(self, scalar):
        """
        Overload the * operator for scalar multiplication.
        
        Args:
            scalar (float): The scalar to multiply the NamedParametersToOptimize object by.
            
        Returns:
            NamedParametersToOptimize: A new NamedParametersToOptimize object with all parameters multiplied by the scalar.
        """
        result = NamedParametersToOptimize()
        for name, param in self.items():
            result[name] = scalar * param
        return result

# class MyModel(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.layer1 = torch.nn.Linear(10, 20)
#         self.layer2 = torch.nn.Linear(20, 10)

#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         return x

#     def get_named_parameters_to_optimize(self):
#         """
#         Get a NamedParametersToOptimize object containing the model's parameters that require gradients.
        
#         Returns:
#             NamedParametersToOptimize: A NamedParametersToOptimize object containing the model's parameters that require gradients.
#         """
#         named_params_to_optim = NamedParametersToOptimize()
#         for name, param in self.named_parameters():
#             if param.requires_grad:
#                 named_params_to_optim[name] = param
#                 param.grad = None
#         return named_params_to_optim