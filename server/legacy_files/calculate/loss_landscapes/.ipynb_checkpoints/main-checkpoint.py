"""
Functions for approximating loss/return landscapes in one and two dimensions.
"""
import copy
import typing
import abc

import torch.nn
import numpy as np

from loss_landscapes.model_interface.model_wrapper import ModelWrapper, wrap_model
from loss_landscapes.model_interface.model_parameters import ModelParameters, rand_u_like, rand_n_like, orthogonal_to
from loss_landscapes.metrics.metric import Metric

# noinspection DuplicatedCode
def point(model: typing.Union[torch.nn.Module, ModelWrapper], metric: Metric) -> tuple:
    """
    Returns the computed value of the evaluation function applied to the model
    or agent at a specific point in parameter space.

    The Metric supplied has to be a subclass of the loss_landscapes.metrics.Metric
    class, and must specify a procedure whereby the model passed to it is evaluated on the
    task of interest, returning the resulting quantity (such as loss, loss gradient, etc).

    The model supplied can be either a torch.nn.Module model, or a ModelWrapper from the
    loss_landscapes library for more complex cases.

    :param model: the model or model wrapper defining the point in parameter space
    :param metric: Metric object used to evaluate model
    :return: quantity specified by Metric at point in parameter space
    """
    return metric(wrap_model(model))

class LossLandscape(abc.ABC):
    """
    Computed value of the evaluation function applied to the model in a subspace.
    The models supplied can be either torch.nn.Module models, or ModelWrapper objects
    from the loss_landscapes library for more complex cases.

    The Metric supplied has to be a subclass of the loss_landscapes.metrics.Metric class,
    and must specify a procedure whereby the model passed to it is evaluated on the
    task of interest, returning the resulting quantity (such as loss, loss gradient, etc).
    """
    def __init__(self, model_start: typing.Union[torch.nn.Module, ModelWrapper], steps=100, deepcopy_model=False):
        """
        :param model_start: model to be evaluated, whose current parameters represent the start point
        :param steps: at how many steps from start to end the model is evaluated
        :param deepcopy_model: indicates whether the method will deepcopy the model(s) to avoid aliasing
        """
        self.deepcopy_model = deepcopy_model
        self.steps = steps
        self.model_start_wrapper = wrap_model(copy.deepcopy(model_start) if self.deepcopy_model else model_start)
        self.start_point = self.model_start_wrapper.get_module_parameters()

    @abc.abstractmethod
    def stats_initializer(self):
        pass

    @abc.abstractmethod
    def warm_up(self):
        pass

    @abc.abstractmethod
    def outer_warm_up(self):
        pass

    @abc.abstractmethod
    def compute(self):
        pass

    @abc.abstractmethod
    def outer_compute(self):
        pass

class LinearLossLandscape(LossLandscape):
    """
    Returns the computed value of the evaluation function applied to the model or agent along a linear subspace.
    """
    def __init__(self, model_start: typing.Union[torch.nn.Module, ModelWrapper], steps=100, deepcopy_model=False):
        """
        :param model_start: model to be evaluated, whose current parameters represent the start point
        :param steps: at how many steps from start to end the model is evaluated
        :param deepcopy_model: indicates whether the method will deepcopy the model(s) to avoid aliasing
        """
        super().__init__(model_start, steps, deepcopy_model)

    def interpolation(self, model_end: typing.Union[torch.nn.Module, ModelWrapper]):
        """
        The linear subspace of the parameter space defined by two end points.

        That is, given two models, for both of which the model's parameters define a
        vertex in parameter space, the evaluation is computed at the given number of steps
        along the straight line connecting the two vertices. A common choice is to
        use the weights before training and the weights after convergence as the start
        and end points of the line, thus obtaining a view of the "straight line" in
        parameter space from the initialization to some minima. There is no guarantee
        that the model followed this path during optimization. In fact, it is highly
        unlikely to have done so, unless the optimization problem is convex.

        Note that a simple linear interpolation can produce misleading approximations
        of the loss landscape due to the scale invariance of neural networks. The sharpness/
        flatness of minima or maxima is affected by the scale of the neural network weights.
        For more details, see `https://arxiv.org/abs/1712.09913v3`. It is recommended to
        use random_line() with filter normalization instead.

        :param model_end: the model defining the end point of the line in parameter space
        """
        end_model_wrapper = wrap_model(copy.deepcopy(model_end) if self.deepcopy_model else model_end)
        self.direction = (end_model_wrapper.get_module_parameters() - self.start_point) / self.steps

    def precomputed(self, direction: ModelParameters, distance: float, normalization='filter', centered=True):
        """
        The linear subspace of the parameter space defined by precomputed direction and distance.

        Given a user defined direction and distance, draw loss landscape a long the line towards the direction.

        :param direction: the precomputed direction given by the user
        :param distance: maximum distance in parameter space from the start point
        :param normalization: normalization of direction other, must be one of 'filter', 'layer', 'model'
        :param centered: boolean value defines if the start point should be located at the center or corner
        """
        self.direction = copy.deepcopy(direction)

        if normalization == 'model':
            self.direction.model_normalize_(self.start_point)
        elif normalization == 'layer':
            self.direction.layer_normalize_(self.start_point)
        elif normalization == 'filter':
            self.direction.filter_normalize_(self.start_point)
        elif normalization is None:
            pass
        else:
            raise AttributeError('Unsupported normalization argument. Supported values are model, layer, and filter')

        self.direction.mul_(((self.start_point.model_norm() * distance) / self.steps) / self.direction.model_norm())
        # Move start point so that original start params will be in the center of the plot
        if centered==True:
            self.direction.mul_(self.steps / 2)
            self.start_point.sub_(self.direction)
            self.direction.truediv_(self.steps / 2)

    def random_line(self, distance: float, normalization='filter', random='uniform', centered=True):
        """
        The linear subspace of the parameter space defined by a start point and a randomly sampled direction.

        That is, given a neural network model, whose parameters define a point in parameter
        space, and a distance, the evaluation is computed at 'steps' points along a random
        direction, from the start point up to the maximum distance from the start point.

        Note that the dimensionality of the model parameters has an impact on the expected
        length of a uniformly sampled other in parameter space. That is, the more parameters
        a model has, the longer the distance in the random other's direction should be,
        in order to see meaningful change in individual parameters. Normalizing the
        direction other according to the model's current parameter values, which is supported
        through the 'normalization' parameter, helps reduce the impact of the distance
        parameter. In future releases, the distance parameter will refer to the maximum change
        in an individual parameter, rather than the length of the random direction other.

        Note also that a simple line approximation can produce misleading views
        of the loss landscape due to the scale invariance of neural networks. The sharpness or
        flatness of minima or maxima is affected by the scale of the neural network weights.
        For more details, see `https://arxiv.org/abs/1712.09913v3`. It is recommended to
        normalize the direction, preferably with the 'filter' option.

        :param distance: maximum distance in parameter space from the start point
        :param normalization: normalization of direction other, must be one of 'filter', 'layer', 'model'
        :param random: normalization of direction other, must be one of 'uniform', 'normal'
        """

        # obtain start point in parameter space and random direction
        # random direction is randomly sampled, then normalized, and finally scaled by distance/steps
        if random == 'uniform':
            self.direction = rand_u_like(self.start_point)
        elif random == 'normal':
            self.direction = rand_n_like(self.start_point)
        else:
            raise AttributeError('Unsupported random argument. Supported values are uniform and normal')

        if normalization == 'model':
            self.direction.model_normalize_(self.start_point)
        elif normalization == 'layer':
            self.direction.layer_normalize_(self.start_point)
        elif normalization == 'filter':
            self.direction.filter_normalize_(self.start_point)
        elif normalization is None:
            pass
        else:
            raise AttributeError('Unsupported normalization argument. Supported values are model, layer, and filter')

        self.direction.mul_(((self.start_point.model_norm() * distance) / self.steps) / self.direction.model_norm())
        # Move start point so that original start params will be in the center of the plot
        if centered==True:
            self.direction.mul_(self.steps / 2)
            self.start_point.sub_(self.direction)
            self.direction.truediv_(self.steps / 2)

    def stats_initializer(self, running_stats: typing.Union[None, typing.List[dict]] = None):
        if running_stats is not None:
            self.running_stats = running_stats
            return
        self.model_start_running_stats = self.model_start_wrapper.get_module_running_stats()
        self.running_stats = [copy.deepcopy(self.model_start_running_stats) for _ in range(self.steps)]

    def warm_up(self, metric: Metric):
        """
        :param metric: function of form evaluation_f(model), used to evaluate model loss
        """
        # initilize the running statistics for all steps
        if hasattr(LinearLossLandscape, 'running_stats'):
            raise AttributeError('Running stats is not either initialized or precomputed. Call stats_initializer() first')
        self.model_start_wrapper.train()
        # Work on a copy of model wrapper to avoid changing the parameters of default model_start_wrapper
        model_start_wrapper = copy.deepcopy(self.model_start_wrapper)
        start_point = model_start_wrapper.get_module_parameters()
        # set the model wrapper into train mode to enable running statistics computation
        for i in range(self.steps):
            # add a step along the line to the model parameters, then evaluate
            start_point.add_(self.direction)
            model_start_wrapper.load_module_running_stats(self.running_stats[i])
            metric(model_start_wrapper)
            self.running_stats[i]=copy.deepcopy(model_start_wrapper.get_module_running_stats())

    def outer_warm_up(self, pos: int, outer_func: typing.Callable):
        """
        The function that calls outer_func to collect running statistics at position pos
        :param pos: the 0-indexed indicator of which running statistics to update
        :param outer_func: function used to call forward path of model
        """
        # initilize the running statistics for all steps
        if hasattr(LinearLossLandscape, 'running_stats'):
            raise AttributeError('Running stats is not either initialized or precomputed. Call stats_initializer() first')
        assert pos<self.steps, f'The specified running statistics at {pos} is greater than the setup {self.steps}'

        self.model_start_wrapper.train()
        # Work on a copy of model wrapper to avoid changing the parameters of default model_start_wrapper
        model_start_wrapper = copy.deepcopy(self.model_start_wrapper)
        start_point = model_start_wrapper.get_module_parameters()
        for i in range(pos+1):
            # add a step along the line to the model parameters, then evaluate
            start_point.add_(self.direction)
        model_start_wrapper.load_module_running_stats(self.running_stats[i])
        outer_func(model_start_wrapper)
        self.running_stats[i]=copy.deepcopy(model_start_wrapper.get_module_running_stats())

    def compute(self, metric: Metric) -> np.ndarray:
        """
        :param metric: function of form evaluation_f(model), used to evaluate model loss
        :return: 1-d array of loss values along the line
        """
        if hasattr(LinearLossLandscape, 'running_stats'):
            raise AttributeError('Running stats is not either initialized or precomputed. Call stats_initializer() first')
        self.model_start_wrapper.eval()
        model_start_wrapper = copy.deepcopy(self.model_start_wrapper)
        start_point = model_start_wrapper.get_module_parameters()
        # set the model wrapper into eval mode to load pretrained models and running statistics
        data_values = []
        for i in range(self.steps):
            # add a step along the line to the model parameters, then evaluate
            start_point.add_(self.direction)
            model_start_wrapper.load_module_running_stats(self.running_stats[i])
            data_values.append(metric(model_start_wrapper))

        return np.array(data_values)

    def outer_compute(self, pos: int, outer_func: typing.Callable) -> float:
        """
        The function that calls outer_func to evaluate loss at position pos
        :param pos: the 0-indexed indicator of which running statistics to load and evalutate loss
        :param outer_func: function used to evaluate the loss of model
        :return: loss value at pos
        """
        if hasattr(LinearLossLandscape, 'running_stats'):
            raise AttributeError('Running stats is not either initialized or precomputed. Call stats_initializer() first')
        assert pos<self.steps, f'The specified running statistics at {pos} is over the limit of {self.steps}'

        self.model_start_wrapper.eval()
        model_start_wrapper = copy.deepcopy(self.model_start_wrapper)
        start_point = model_start_wrapper.get_module_parameters()
        # set the model wrapper into eval mode to load pretrained models and running statistics
        data_values = []
        for i in range(pos+1):
            # add a step along the line to the model parameters, then evaluate
            start_point.add_(self.direction)
        model_start_wrapper.load_module_running_stats(self.running_stats[pos])
        return outer_func(model_start_wrapper)

class PlanarLossLandscape(LossLandscape):
    """
    Returns the computed value of the evaluation function applied to the model or agent along a planar subspace.
    """
    def __init__(self, model_start: typing.Union[torch.nn.Module, ModelWrapper], steps=20, deepcopy_model=False):
        """
        :param model_start: the model defining the origin point of the plane in parameter space
        :param steps: at how many steps from start to end the model is evaluated
        :param deepcopy_model: indicates whether the method will deepcopy the model(s) to avoid aliasing
        """
        super().__init__(model_start, steps, deepcopy_model)

    def interpolation(self, 
                      model_end_one: typing.Union[torch.nn.Module, ModelWrapper],
                      model_end_two: typing.Union[torch.nn.Module, ModelWrapper]):
        """
        The planar subspace of the parameter space defined by a start point and two end points.

        That is, given two models, for both of which the model's parameters define a
        vertex in parameter space, the loss is computed at the given number of steps
        along the straight line connecting the two vertices. A common choice is to
        use the weights before training and the weights after convergence as the start
        and end points of the line, thus obtaining a view of the "straight line" in
        paramater space from the initialization to some minima. There is no guarantee
        that the model followed this path during optimization. In fact, it is highly
        unlikely to have done so, unless the optimization problem is convex.

        That is, given three neural network models, 'model_start', 'model_end_one', and
        'model_end_two', each of which defines a point in parameter space, the loss is
        computed at 'steps' * 'steps' points along the plane defined by the start vertex
        and the two vectors (end_one - start) and (end_two - start), up to the maximum
        distance in both directions. A common choice would be for two of the points to be
        the model after initialization, and the model after convergence. The third point
        could be another randomly initialized model, since in a high-dimensional space
        randomly sampled directions are most likely to be orthogonal.

        :param model_end_one: the model representing the end point of the first direction defining the plane
        :param model_end_two: the model representing the end point of the second direction defining the plane
        """
        model_end_one_wrapper = wrap_model(copy.deepcopy(model_end_one) if self.deepcopy_model else model_end_one)
        model_end_two_wrapper = wrap_model(copy.deepcopy(model_end_two) if self.deepcopy_model else model_end_two)

        # compute direction vectors
        self.dir_one = (model_end_one_wrapper.get_module_parameters() - self.start_point) / self.steps
        self.dir_two = (model_end_two_wrapper.get_module_parameters() - self.start_point) / self.steps

    def precomputed(self, dir_one: ModelParameters, dir_two: ModelParameters, distance: float, normalization='filter', centered=True):
        """
        The planar subspace of the parameter space defined by two precomputed direction and distance.

        Given a user defined direction and distance, draw loss landscape the plain defined by two directions.

        :param dir_one: the precomputed direction 1 given by the user
        :param dir_two: the precomputed direction 2 given by the user
        :param distance: maximum distance in parameter space from the start point
        :param normalization: normalization of direction other, must be one of 'filter', 'layer', 'model'
        :param centered: boolean value defines if the start point should be located at the center or corner
        """
        self.dir_one = copy.deepcopy(dir_one)
        self.dir_two = copy.deepcopy(dir_two)

        if normalization == 'model':
            self.dir_one.model_normalize_(self.start_point)
            self.dir_two.model_normalize_(self.start_point)
        elif normalization == 'layer':
            self.dir_one.layer_normalize_(self.start_point)
            self.dir_two.layer_normalize_(self.start_point)
        elif normalization == 'filter':
            self.dir_one.filter_normalize_(self.start_point)
            self.dir_two.filter_normalize_(self.start_point)
        elif normalization is None:
            pass
        else:
            raise AttributeError('Unsupported normalization argument. Supported values are model, layer, and filter')

        # scale to match steps and total distance
        self.dir_one.mul_(((self.start_point.model_norm() * distance) / self.steps) / self.dir_one.model_norm())
        self.dir_two.mul_(((self.start_point.model_norm() * distance) / self.steps) / self.dir_two.model_norm())
        # Move start point so that original start params will be in the center of the plot
        if centered==True:
            self.dir_one.mul_(self.steps / 2)
            self.dir_two.mul_(self.steps / 2)
            self.start_point.sub_(self.dir_one)
            self.start_point.sub_(self.dir_two)
            self.dir_one.truediv_(self.steps / 2)
            self.dir_two.truediv_(self.steps / 2)

    def random_plain(self, distance: float, normalization='filter', random='uniform', centered=True):
        """
        :param distance: maximum distance in parameter space from the start point
        :param normalization: normalization of direction other, must be one of 'filter', 'layer', 'model'
        :param random: normalization of direction other, must be one of 'uniform', 'normal'

        That is, given a neural network model, whose parameters define a point in parameter
        space, and a distance, the loss is computed at 'steps' * 'steps' points along the
        plane defined by the two random directions, from the start point up to the maximum
        distance in both directions.

        Note that the dimensionality of the model parameters has an impact on the expected
        length of a uniformly sampled other in parameter space. That is, the more parameters
        a model has, the longer the distance in the random other's direction should be,
        in order to see meaningful change in individual parameters. Normalizing the
        direction other according to the model's current parameter values, which is supported
        through the 'normalization' parameter, helps reduce the impact of the distance
        parameter. In future releases, the distance parameter will refer to the maximum change
        in an individual parameter, rather than the length of the random direction other.

        Note also that a simple planar approximation with randomly sampled directions can produce
        misleading approximations of the loss landscape due to the scale invariance of neural
        networks. The sharpness/flatness of minima or maxima is affected by the scale of the neural
        network weights. For more details, see `https://arxiv.org/abs/1712.09913v3`. It is
        recommended to normalize the directions, preferably with the 'filter' option.

        :param distance: maximum distance in parameter space from the start point
        :param normalization: normalization of direction other, must be one of 'filter', 'layer', 'model'
        :param random: normalization of direction other, must be one of 'uniform', 'normal'
        """

        # obtain start point in parameter space and random direction
        # random direction is randomly sampled, then normalized, and finally scaled by distance/steps
        if random == 'uniform':
            self.dir_one = rand_u_like(self.start_point)
        elif random == 'normal':
            self.dir_one = rand_n_like(self.start_point)
        else:
            raise AttributeError('Unsupported random argument. Supported values are uniform and normal')
        self.dir_two = orthogonal_to(self.dir_one, random)

        if normalization == 'model':
            self.dir_one.model_normalize_(self.start_point)
            self.dir_two.model_normalize_(self.start_point)
        elif normalization == 'layer':
            self.dir_one.layer_normalize_(self.start_point)
            self.dir_two.layer_normalize_(self.start_point)
        elif normalization == 'filter':
            self.dir_one.filter_normalize_(self.start_point)
            self.dir_two.filter_normalize_(self.start_point)
        elif normalization is None:
            pass
        else:
            raise AttributeError('Unsupported normalization argument. Supported values are model, layer, and filter')

        # scale to match steps and total distance
        self.dir_one.mul_(((self.start_point.model_norm() * distance) / self.steps) / self.dir_one.model_norm())
        self.dir_two.mul_(((self.start_point.model_norm() * distance) / self.steps) / self.dir_two.model_norm())
        # Move start point so that original start params will be in the center of the plot
        if centered==True:
            self.dir_one.mul_(self.steps / 2)
            self.dir_two.mul_(self.steps / 2)
            self.start_point.sub_(self.dir_one)
            self.start_point.sub_(self.dir_two)
            self.dir_one.truediv_(self.steps / 2)
            self.dir_two.truediv_(self.steps / 2)

    def stats_initializer(self, running_stats: typing.Union[None, typing.List[typing.List[dict]]] = None):
        if running_stats is not None:
            self.running_stats = running_stats
            return
        self.model_start_running_stats = self.model_start_wrapper.get_module_running_stats()
        self.running_stats = []
        for _ in range(self.steps):
            self.running_stats.append([copy.deepcopy(self.model_start_running_stats) for _ in range(self.steps)])

    def warm_up(self, metric: Metric):
        """
        :param metric: function of form evaluation_f(model), used to evaluate model loss
        """
        # initilize the running statistics for all steps
        if hasattr(PlanarLossLandscape, 'running_stats'):
            raise AttributeError('Running stats is not either initialized or precomputed. Call stats_initializer() first')
        self.model_start_wrapper.train()
        # Work on a copy of model wrapper to avoid changing the parameters of default model_start_wrapper
        model_start_wrapper = copy.deepcopy(self.model_start_wrapper)
        start_point = model_start_wrapper.get_module_parameters()
        # set the model wrapper into train mode to enable running statistics computation
        for i in range(self.steps):
            for j in range(self.steps):
                # for every other column, reverse the order in which the column is generated
                # so you can easily use in-place operations to move along dir_two
                if i % 2 == 0:
                    start_point.add_(self.dir_two)
                    model_start_wrapper.load_module_running_stats(self.running_stats[i][j])
                    metric(model_start_wrapper)
                    self.running_stats[i][j]=copy.deepcopy(model_start_wrapper.get_module_running_stats())
                else:
                    start_point.sub_(self.dir_two)
                    model_start_wrapper.load_module_running_stats(self.running_stats[i][self.steps-j-1])
                    metric(model_start_wrapper)
                    self.running_stats[i][self.steps-j-1]=copy.deepcopy(model_start_wrapper.get_module_running_stats())
            start_point.add_(self.dir_one)

    def outer_warm_up(self, pos_i: int, pos_j: int, outer_func: typing.Callable):
        """
        The function that calls outer_func to collect running statistics at position pos
        :param pos_i: the 0-indexed row indicator of which running statistics to update
        :param pos_j: the 0-indexed column indicator of which running statistics to update
        :param outer_func: function used to call forward path of model
        """
        # initilize the running statistics for all steps
        if hasattr(PlanarLossLandscape, 'running_stats'):
            raise AttributeError('Running stats is not either initialized or precomputed. Call stats_initializer() first')
        assert pos_i<self.steps, f'The specified running statistics at {pos_i} is over the limit of {self.steps}'
        assert pos_j<self.steps, f'The specified running statistics at {pos_j} is over the limit of {self.steps}'
        
        self.model_start_wrapper.train()
        # Work on a copy of model wrapper to avoid changing the parameters of default model_start_wrapper
        model_start_wrapper = copy.deepcopy(self.model_start_wrapper)
        start_point = model_start_wrapper.get_module_parameters()
        # Z tranversal of the first pos_i-1 rows which are all complete tranversal of columns
        for i in range(pos_i):
            for j in range(self.steps):
            # for every other column, reverse the order in which the column is generated
                # so you can easily use in-place operations to move along dir_two
                if i % 2 == 0:
                    start_point.add_(self.dir_two)
                else:
                    start_point.sub_(self.dir_two)
            start_point.add_(self.dir_one)
        # Z tranversal to the last row which is an incomplete tranversal of columns
        if pos_i % 2 == 0:
            for j in range(pos_j):
                start_point.add_(self.dir_two)
        else:
            for j in range(self.steps-pos_j-1):
                start_point.sub_(self.dir_two)

        # update and save the running statistics
        if pos_i % 2 == 0:
            model_start_wrapper.load_module_running_stats(self.running_stats[pos_i][pos_j])
            outer_func(model_start_wrapper)
            self.running_stats[pos_i][pos_j]=copy.deepcopy(model_start_wrapper.get_module_running_stats())
        else:
            model_start_wrapper.load_module_running_stats(self.running_stats[pos_i][self.steps-pos_j-1])
            outer_func(model_start_wrapper)
            self.running_stats[pos_i][self.steps-j-1]=copy.deepcopy(model_start_wrapper.get_module_running_stats())

    def compute(self, metric: Metric) -> np.ndarray:
        """
        :param metric: function of form evaluation_f(model), used to evaluate model loss
        :return: 2-d array of loss values along the plane
        """
        if hasattr(PlanarLossLandscape, 'running_stats'):
            raise AttributeError('Running stats is not either initialized or precomputed. Call stats_initializer() first')
        self.model_start_wrapper.eval()
        model_start_wrapper = copy.deepcopy(self.model_start_wrapper)
        start_point = model_start_wrapper.get_module_parameters()
        data_matrix = []
        # evaluate loss in grid of (steps * steps) points, where each column signifies one step
        # along dir_one and each row signifies one step along dir_two. The implementation is again
        # a little convoluted to avoid constructive operations. Fundamentally we generate the matrix
        # [[start_point + (dir_one * i) + (dir_two * j) for j in range(steps)] for i in range(steps].
        for i in range(self.steps):
            data_column = []

            for j in range(self.steps):
                # for every other column, reverse the order in which the column is generated
                # so you can easily use in-place operations to move along dir_two
                if i % 2 == 0:
                    start_point.add_(self.dir_two)
                    model_start_wrapper.load_module_running_stats(self.running_stats[i][j])
                    data_column.append(metric(model_start_wrapper))
                else:
                    start_point.sub_(self.dir_two)
                    model_start_wrapper.load_module_running_stats(self.running_stats[i][self.steps-j-1])
                    data_column.insert(0, metric(model_start_wrapper))

            data_matrix.append(data_column)
            start_point.add_(self.dir_one)

        return np.array(data_matrix)

    def outer_compute(self, pos_i: int, pos_j: int, outer_func: typing.Callable) -> float:
        """
        The function that calls outer_func to evluate losses at position (pos_i, pos_j)
        :param pos_i: the 0-indexed row indicator of which loss to update
        :param pos_j: the 0-indexed column indicator of which loss to update
        :param outer_func: function used to evaluate the loss of model
        :return: loss value at (pos_i, pos_j)
        """
        # initilize the running statistics for all steps
        if hasattr(PlanarLossLandscape, 'running_stats'):
            raise AttributeError('Running stats is not either initialized or precomputed. Call stats_initializer() first')
        assert pos_i<self.steps, f'The specified running statistics at {pos_i} is over the limit of {self.steps}'
        assert pos_j<self.steps, f'The specified running statistics at {pos_j} is over the limit of {self.steps}'

        self.model_start_wrapper.eval()
        model_start_wrapper = copy.deepcopy(self.model_start_wrapper)
        start_point = model_start_wrapper.get_module_parameters()

        # Z tranversal of the first pos_i-1 rows which are all complete tranversal of columns
        for i in range(pos_i):
            for j in range(self.steps):
            # for every other column, reverse the order in which the column is generated
                # so you can easily use in-place operations to move along dir_two
                if i % 2 == 0:
                    start_point.add_(self.dir_two)
                else:
                    start_point.sub_(self.dir_two)
            start_point.add_(self.dir_one)
        # Z tranversal to the last row which is an incomplete tranversal of columns
        if pos_i % 2 == 0:
            for j in range(pos_j):
                start_point.add_(self.dir_two)
        else:
            for j in range(self.steps-pos_j-1):
                start_point.sub_(self.dir_two)

        # call function to evalute loss
        if pos_i % 2 == 0:
            model_start_wrapper.load_module_running_stats(self.running_stats[pos_i][pos_j])
            return outer_func(model_start_wrapper)
        else:
            model_start_wrapper.load_module_running_stats(self.running_stats[pos_i][self.steps-pos_j-1])
            return outer_func(model_start_wrapper)
        
