"""Defines Objective classes to be used in Problems

"""

from abc import ABC, abstractmethod
from os import path
from typing import Callable, Dict, List, NamedTuple, Tuple, Union

import numpy as np
import pandas as pd

from desdeo_problem.surrogatemodels.SurrogateModels import BaseRegressor, ModelError


class ObjectiveError(Exception):
    """Raised when an error related to the Objective class is encountered.

    """


class ObjectiveEvaluationResults(NamedTuple):
    """The return object of <problem>.evaluate methods.

    Attributes:
        objectives (Union[float, np.ndarray]): The objective function value/s for the
            input vector.
        uncertainity (Union[None, float, np.ndarray]): The uncertainity in the
            objective value/s.

    """

    objectives: Union[float, np.ndarray]
    uncertainity: Union[None, float, np.ndarray] = None

    def __str__(self):
        prnt_msg = (
            "Objective Evaluation Results Object \n"
            f"Objective values are: \n{self.objectives}\n"
            f"Uncertainity values are: \n{self.uncertainity}\n"
        )
        return prnt_msg


class ObjectiveBase(ABC):
    """The abstract base class for objectives.

    """

    def evaluate(self, decision_vector: np.ndarray, use_surrogate: bool = False) -> ObjectiveEvaluationResults:
        """Evaluates the objective according to a decision variable vector.

        Uses surrogate model if use_surrogates is true. If use_surrogates is False, uses
        func_evaluate which evaluates using the true objective function.

        Args:
            variables (np.ndarray): A vector of Variables to be used in
                the evaluation of the objective.
            use_surrogate (bool) : A boolean which determines whether to use surrogates
                or true function evaluator. False by default.

        """
        if use_surrogate:
            return self._surrogate_evaluate(decision_vector)
        else:
            return self._func_evaluate(decision_vector)

    @abstractmethod
    def _func_evaluate(self, decision_vector: np.ndarray) -> ObjectiveEvaluationResults:
        """Evaluates the true objective value according to a decision variable vector.

        Uses the true (potentially expensive) evaluator if available.

        Args:
            variables (np.ndarray): A vector of Variables to be used in
                the evaluation of the objective.

        """
        pass

    @abstractmethod
    def _surrogate_evaluate(self, decision_vector: np.ndarray) -> ObjectiveEvaluationResults:
        """Evaluates the objective value according to a decision variable vector.

        Uses the surrogartes if available.

        Args:
            variables (np.ndarray): A vector of Variables to be used in
                the evaluation of the objective.

        """
        pass


class VectorObjectiveBase(ABC):
    """The abstract base class for multiple objectives which are calculated at once.

    """

    def evaluate(self, decision_vector: np.ndarray, use_surrogate: bool = False) -> ObjectiveEvaluationResults:
        """Evaluates the objective according to a decision variable vector.

        Uses surrogate model if use_surrogates is true. If use_surrogates is False, uses
        func_evaluate which evaluates using the true objective function.

        Args:
            variables (np.ndarray): A vector of Variables to be used in
                the evaluation of the objective.
            use_surrogate (bool) : A boolean which determines whether to use surrogates
                or true function evaluator. False by default.

        """
        if use_surrogate:
            return self._surrogate_evaluate(decision_vector)
        else:
            return self._func_evaluate(decision_vector)

    @abstractmethod
    def _func_evaluate(self, decision_vector: np.ndarray) -> ObjectiveEvaluationResults:
        """Evaluates the true objective values according to a decision variable vector.

        Uses the true (potentially expensive) evaluator if available.

        Args:
            variables (np.ndarray): A vector of Variables to be used in
                the evaluation of the objective.

        """
        pass

    @abstractmethod
    def _surrogate_evaluate(self, decision_vector: np.ndarray) -> ObjectiveEvaluationResults:
        """Evaluates the objective values according to a decision variable vector.

        Uses the surrogartes if available.

        Args:
            variables (np.ndarray): A vector of Variables to be used in
                the evaluation of the objective.

        """
        pass


# TODO: Depreciate
class _ScalarObjective(ObjectiveBase):
    """A simple objective function that returns a scalar.

    Args:
        name (str): Name of the objective.
        evaluator (Callable): The function to evaluate the objective's value.
        lower_bound (float): The lower bound of the objective.
        upper_bound (float): The upper bound of the objective.
        maximize (bool): Boolean to determine whether the objective is to be maximized.

    Attributes:
        name (str): Name of the objective.
        value (float): The current value of the objective function.
        evaluator (Callable): The function to evaluate the objective's value.
        lower_bound (float): The lower bound of the objective.
        upper_bound (float): The upper bound of the objective.
        maximize (List[bool]): List of boolean to determine whether the objectives are
            to be maximized. All false by default

    Raises:
        ObjectiveError: When ill formed bounds are given.

    """

    def __init__(
        self,
        name: str,
        evaluator: Callable,
        lower_bound: float = -np.inf,
        upper_bound: float = np.inf,
        maximize: List[bool] = None,
    ) -> None:
        # Check that the bounds make sense
        if not (lower_bound < upper_bound):
            msg = ("Lower bound {} should be less than the upper bound " "{}.").format(lower_bound, upper_bound)
            raise ObjectiveError(msg)

        self.__name: str = name
        self.__evaluator: Callable = evaluator
        self.__value: float = 0.0
        self.__lower_bound: float = lower_bound
        self.__upper_bound: float = upper_bound
        if maximize is None:
            maximize = [False]
        self.maximize: bool = maximize  # TODO implement set/getters. Have validation.

    @property
    def name(self) -> str:
        return self.__name

    @property
    def value(self) -> float:
        return self.__value

    @value.setter
    def value(self, value: float):
        self.__value = value

    @property
    def evaluator(self) -> Callable:
        return self.__evaluator

    @property
    def lower_bound(self) -> float:
        return self.__lower_bound

    @property
    def upper_bound(self) -> float:
        return self.__upper_bound

    def _func_evaluate(self, decision_vector: np.ndarray) -> ObjectiveEvaluationResults:
        """Evaluate the objective functions value.

        Args:
            variables (np.ndarray): A vector of variables to evaluate the
                objective function with.
        Returns:
            ObjectiveEvaluationResults: A named tuple containing the evaluated value,
                and uncertainity of evaluation of the objective function.

        Raises:
            ObjectiveError: When a bad argument is supplied to the evaluator.

        """
        try:
            result = self.evaluator(decision_vector)
        except (TypeError, IndexError) as e:
            msg = "Bad argument {} supplied to the evaluator: {}".format(str(decision_vector), str(e))
            raise ObjectiveError(msg)

        # Store the value of the objective
        self.value = result
        uncertainity = np.full_like(result, np.nan, dtype=float)
        # Have to set dtype because if the tuple is of ints, then this array also
        # becomes dtype int. There's no nan value of int type
        return ObjectiveEvaluationResults(result, uncertainity)

    def _surrogate_evaluate(self, decusuib_vector: np.ndarray):
        raise ObjectiveError("Surrogates not trained")


# TODO: Rename to "Objective"
class VectorObjective(VectorObjectiveBase):
    """An objective object that calculated one or more objective functions.

    Args:
        name (List[str]): Names of the various objectives in a list
        evaluator (Callable): The function that evaluates the objective values
        lower_bounds (Union[List[float], np.ndarray), optional): Lower bounds of the
            objective values. Defaults to None.
        upper_bounds (Union[List[float], np.ndarray), optional): Upper bounds of the
            objective values. Defaults to None.
        maximize (List[bool]): *List* of boolean to determine whether the objectives are
            to be maximized. All false by default

    Raises:
        ObjectiveError: When lengths the input arrays are different.
        ObjectiveError: When any of the lower bounds is not smaller than the
            corresponding upper bound.

    """

    def __init__(
        self,
        name: List[str],
        evaluator: Callable,
        lower_bounds: Union[List[float], np.ndarray] = None,
        upper_bounds: Union[List[float], np.ndarray] = None,
        maximize: List[bool] = None,
    ):
        n_of_objectives = len(name)
        if lower_bounds is None:
            lower_bounds = np.full(n_of_objectives, -np.inf)
        if upper_bounds is None:
            upper_bounds = np.full(n_of_objectives, np.inf)
        lower_bounds = np.asarray(lower_bounds)
        upper_bounds = np.asarray(upper_bounds)
        # Check if list lengths are the same
        if not (n_of_objectives == len(lower_bounds)):
            msg = (
                "The length of the list of names and the number of elements in the "
                "lower_bounds array should be the same"
            )
            raise ObjectiveError(msg)
        if not (n_of_objectives == len(upper_bounds)):
            msg = (
                "The length of the list of names and the number of elements in the "
                "upper_bounds array should be the same"
            )
            raise ObjectiveError(msg)
        # Check if all lower bounds are smaller than the corresponding upper bounds
        if not (np.all(lower_bounds < upper_bounds)):
            msg = "Lower bounds should be less than the upper bound "
            raise ObjectiveError(msg)
        self.__name: List[str] = name
        self.__n_of_objectives: int = n_of_objectives
        self.__evaluator: Callable = evaluator
        self.__values: Tuple[float] = (0.0,) * n_of_objectives
        self.__lower_bounds: np.ndarray = lower_bounds
        self.__upper_bounds: np.ndarray = upper_bounds
        if maximize is None:
            self.maximize = [False] * n_of_objectives
        else:
            self.maximize: bool = maximize

    @property
    def name(self) -> str:
        return self.__name

    @property
    def n_of_objectives(self) -> int:
        return self.__n_of_objectives

    @property
    def values(self) -> Tuple[float]:
        return self.__values

    @values.setter
    def values(self, values: Tuple[float]):
        self.__values = values

    @property
    def evaluator(self) -> Callable:
        return self.__evaluator

    @property
    def lower_bounds(self) -> np.ndarray:
        return self.__lower_bounds

    @property
    def upper_bounds(self) -> np.ndarray:
        return self.__upper_bounds

    def _func_evaluate(self, decision_vector: np.ndarray) -> ObjectiveEvaluationResults:
        """Evaluate the multiple objective functions value.

        Args:
            decision_vector (np.ndarray): A vector of variables to evaluate the
                objective function with.
        Returns:
            ObjectiveEvaluationResults: A named tuple containing the evaluated value,
                and uncertainity of evaluation of the objective function.

        Raises:
            ObjectiveError: When a bad argument is supplies to the evaluator or when
                the evaluator returns an unexpected number of outputs.

        """
        try:
            result = self.evaluator(decision_vector)
        except (TypeError, IndexError) as e:
            msg = "Bad argument {} supplied to the evaluator: {}".format(str(decision_vector), str(e))
            raise ObjectiveError(msg)
        result = tuple(result)

        # Store the value of the objective
        self.values = result
        uncertainity = np.full_like(result, np.nan, dtype=float)
        # Have to set dtype because if the tuple is of ints, then this array also
        # becomes dtype int. There's no nan value of int type
        return ObjectiveEvaluationResults(result, uncertainity)

    def _surrogate_evaluate(self, decusuib_vector: np.ndarray):
        raise ObjectiveError("Surrogates not trained")


# TODO: Depreciate
class _ScalarDataObjective(_ScalarObjective):
    """A simple Objective class for single valued objectives. Use when the an evaluator/
    simulator returns a single objective value or when there is no evaluator/simulator.

    Parameters
    ----------
    name : List[str]
        The name of the objective. Should be the same as a column name in the data.
    data : pd.DataFrame
        The data in a pandas dataframe. The columns should be named after variables/
        objective.
    evaluator : Union[None, Callable], optional
        A python function that contains the analytical function or calls the simulator
        to get the true objective value. By default None, as this is not required.
    lower_bound : float, optional
        Lower bound of the objective, by default -np.inf
    upper_bound : float, optional
        Upper bound of the objective, by default np.inf
    maximize : List[bool], optional
        Boolean describing whether the objective is to be maximized or not, by default
        None, which defaults to [False], hence minimizes.

    Raises
    ------
    ObjectiveError
        When the name provided during initialization does not match any name in the
        columns of the data provided during initilizaiton.
    """

    def __init__(
        self,
        name: List[str],
        data: pd.DataFrame,
        evaluator: Union[None, Callable] = None,
        lower_bound: float = -np.inf,
        upper_bound: float = np.inf,
        maximize: List[bool] = None,
    ) -> None:
        if name in data.columns:
            super().__init__(name, evaluator, lower_bound, upper_bound, maximize)
        else:
            msg = f'Name "{name}" not found in the dataframe provided'
            raise ObjectiveError(msg)
        self.X = data.drop(name, axis=1)
        self.y = data[name]
        self.variable_names = self.X.columns
        self._model = None

    def train(
        self, model: BaseRegressor, model_parameters: Dict = None, index: List[int] = None, data: pd.DataFrame = None
    ):
        """Train surrogate model for the objective.

        Parameters
        ----------
        model : BaseRegressor
            A regressor. The regressor, when initialized, should have a fit method and a
            predict method. The predict method should return the predicted objective
            value, as well as the uncertainity value, in a tuple. If the regressor does
            not support calculating uncertainity, return a tuple of objective value and
            None.
        model_parameters : Dict
            **model_parameters is passed to the model when initialized.
        index : List[int], optional
            Indices of the samples (in self.X and self.y), to be used to train the
            surrogate model. By default None, which trains the model on the entire
            dataset. This behaviour may be changed in the future to support test-train
            split or cross validation.
        data : pd.DataFrame, optional
            Extra data to be used for training only. This data is not saved. By default
            None, which then uses self.X and self.y for training.

        Raises
        ------
        ObjectiveError
            For unexpected errors
        """
        if model_parameters is None:
            model_parameters = {}
        self._model = model(**model_parameters)
        if index is None and data is None:
            self._model.fit(self.X, self.y)
            return
        elif index is not None:
            self._model.fit(self.X[index], self.y[index])
            return
        elif data is not None:
            self._model.fit(data[self.variable_names], data[self.name])
            return
        msg = "I don't know how you got this error"
        raise ObjectiveError(msg)

    def _surrogate_evaluate(self, decision_vector: np.ndarray) -> ObjectiveEvaluationResults:
        if self._model is None:
            raise ObjectiveError("Model not trained yet")
        try:
            result, uncertainity = self._model.predict(decision_vector)
        except ModelError:
            msg = "Bad argument supplied to the model"
            raise ObjectiveError(msg)
        return ObjectiveEvaluationResults(result, uncertainity)

    def _func_evaluate(self, decision_vector: np.ndarray) -> ObjectiveEvaluationResults:
        if self.evaluator is None:
            msg = "No analytical function provided"
            raise ObjectiveError(msg)
        results = super()._func_evaluate(decision_vector)
        self.X = np.vstack((self.X, decision_vector))
        self.y = np.vstack((self.y, results.objectives))
        return results


class VectorDataObjective(VectorObjective):
    """A Objective class for multi/valued objectives. Use when the an evaluator/
    simulator returns a multiple objective values or when there is no evaluator/
    simulator.

    Parameters
    ----------
    name : List[str]
        The names of the objectives. Should be the same as a column names in the data.
    data : pd.DataFrame
        The data in a pandas dataframe. The columns should be named after variables/
        objectives.
    evaluator : Union[None, Callable], optional
        A python function that contains the analytical function or calls the simulator
        to get the true objective values. By default None, as this is not required.
    lower_bound : float, optional
        Lower bound of the objectives, by default -np.inf
    upper_bound : float, optional
        Upper bound of the objectives, by default np.inf
    maximize : List[bool], optional
        Boolean describing whether the objective is to be maximized or not, by default
        None, which defaults to [False], hence minimizes.

    Raises
    ------
    ObjectiveError
        When the names provided are not found in the columns of the dataframe provided.
    """

    def __init__(
        self,
        name: List[str],
        data: pd.DataFrame,
        evaluator: Union[None, Callable] = None,
        lower_bounds: Union[List[float], np.ndarray] = None,
        upper_bounds: Union[List[float], np.ndarray] = None,
        maximize: List[bool] = None,
    ) -> None:

        if all(obj in data.columns for obj in name):
            super().__init__(name, evaluator, lower_bounds, upper_bounds, maximize)
        else:
            msg = f'Name "{name}" not found in the dataframe provided'
            raise ObjectiveError(msg)
        self.X = data.drop(name, axis=1)
        self.y = data[name]
        self.variable_names = self.X.columns
        self._model = dict.fromkeys(name)  # TODO: Make the set of keys immutable?
        self._model_trained = dict.fromkeys(name, False)

    def train(
        self,
        models: Union[BaseRegressor, List[BaseRegressor]],
        model_parameters: Union[Dict, List[Dict]] = None,
        index: List[int] = None,
        data: pd.DataFrame = None,
    ):
        """Train surrogate models for the objective.

        Parameters
        ----------
        model : BaseRegressor or List[BaseRegressors]
            A regressor or a list of regressors. The regressor/s, when initialized,
            should have a fit method and a predict method.
            The predict method should return the predicted objective
            value, as well as the uncertainity value, in a tuple. If the regressor does
            not support calculating uncertainity, return a tuple of objective value and
            None.
            If a single regressor is provided, that regressor is used for all the
            objectives.
            If a list of regressors is provided, and if the list contains one regressor
            for each objective, then those individual regressors are used to model the
            objectives. If the number of regressors is not equal to the number of
            objectives, an error is raised.
        model_parameters: Dict or List[Dict]
            The parameters for the regressors. Should be a dict if a single regressor is
            provided. If a list of regressors is provided, the parameters should be in a
            list of dicts, same length as the list of regressors(= number of objs).
        index : List[int], optional
            Indices of the samples (in self.X and self.y), to be used to train the
            surrogate model. By default None, which trains the model on the entire
            dataset. This behaviour may be changed in the future to support test-train
            split or cross validation.
        data : pd.DataFrame, optional
            Extra data to be used for training only. This data is not saved. By default
            None, which then uses self.X and self.y for training.

        Raises
        ------
        ObjectiveError
            If the formats of the model and model parameters do not match
        ObjectiveError
            If the lengths of list of models and/or model parameter dictionaries are not
            equal to the number of objectives.
        """
        if model_parameters is None:
            model_parameters = {}
        if not isinstance(models, list):
            if not (isinstance(model_parameters, dict)):
                msg = "If only one model is provided, model parameters should be a dict"
                raise ObjectiveError(msg)
            models = [models] * len(self.name)
            model_parameters = [model_parameters] * len(self.name)
        elif not (len(models) == len(model_parameters) == self.n_of_objectives):
            msg = (
                "The length of lists of models and parameters should be the same as"
                "the number of objectives in this objective class"
            )
        for model, model_params, name in zip(models, model_parameters, self.name):
            self._train_one_objective(name, model, model_params, index, data)

    def _train_one_objective(
        self,
        name: str,
        model: BaseRegressor,
        model_parameters: Dict,
        index: List[int] = None,
        data: pd.DataFrame = None,
    ):
        """Train surrogate model for the objective.

        Parameters
        ----------
        name : str
            Name of the objective for which you want to train the surrogate model
        model : BaseRegressor
            A regressor. The regressor, when initialized, should have a fit method and a
            predict method. The predict method should return the predicted objective
            value, as well as the uncertainity value, in a tuple. If the regressor does
            not support calculating uncertainity, return a tuple of objective value and
            None.
        model_parameters : Dict
            **model_parameters is passed to the model when initialized.
        index : List[int], optional
            Indices of the samples (in self.X and self.y), to be used to train the
            surrogate model. By default None, which trains the model on the entire
            dataset. This behaviour may be changed in the future to support test-train
            split or cross validation.
        data : pd.DataFrame, optional
            Extra data to be used for training only. This data is not saved. By default
            None, which then uses self.X and self.y for training.

        Raises
        ------
        ObjectiveError
            For unexpected errors
        """
        if name not in self.name:
            raise ObjectiveError(f'"{name}" not found in the list of' f"original objective names: {self.name}")
        if model_parameters is None:
            model_parameters = {}
        self._model[name] = model(**model_parameters)
        if index is None and data is None:
            self._model[name].fit(self.X, self.y[name])
            self._model_trained[name] = True
            return
        elif index is not None:
            self._model[name].fit(self.X[index], self.y[name][index])
            self._model_trained[name] = True
            return
        elif data is not None:
            self._model[name].fit(data[self.variable_names], data[name])
            self._model_trained[name] = True
            return
        msg = "I don't know how you got this error"
        raise ObjectiveError(msg)

    def _surrogate_evaluate(self, decision_vector: np.ndarray) -> ObjectiveEvaluationResults:
        if not all(self._model_trained.values()):
            msg = (
                f"Some or all models have not been trained.\n"
                f"Models for the following objectives have been trained:\n"
                f"{self._model_trained}"
            )
            raise ObjectiveError(msg)
        result = pd.DataFrame(index=range(decision_vector.shape[0]), columns=self.name)
        uncertainity = pd.DataFrame(index=range(decision_vector.shape[0]), columns=self.name)
        for name, model in self._model.items():
            try:
                result[name], uncertainity[name] = model.predict(decision_vector)
            except ModelError:
                msg = "Bad argument supplied to the model"
                raise ObjectiveError(msg)
        return ObjectiveEvaluationResults(result, uncertainity)

    def _func_evaluate(self, decision_vector: np.ndarray) -> ObjectiveEvaluationResults:
        if self.evaluator is None:
            msg = "No analytical function provided"
            raise ObjectiveError(msg)
        results = super()._func_evaluate(decision_vector)
        self.X = np.vstack((self.X, decision_vector))
        self.y = np.vstack((self.y, results.objectives))
        return results
