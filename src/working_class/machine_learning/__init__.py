import threading

from src.working_class import Worker
from typing import NoReturn


class MachineLearningWorker(Worker):
    """
    Base class for a MachineLearningWorker.

    A MachineLearningWorker is a thread-safe object which performs some work on a given input, using a machine learning
    model. It may produce an output, but that is dependent on the implementation of the subclass.
    """

    _MODEL = None
    """Model to be used by the MachineLearningWorker."""
    _MODEL_LOCK = threading.Lock()
    """Lock for the model."""

    def __init__(self):
        """
        Constructs a new MachineLearningWorker.
        """
        super().__init__()

        self._model = None
        self._model_lock = threading.Lock()

    def _load_model(self, load_model: callable) -> NoReturn:
        """
        Attempts to load the model by calling the given function. If the model is already loaded, this method does
        nothing.

        :param load_model: A callable function which loads and returns the model.
        """
        if load_model is None:
            raise ValueError("Load model function cannot be None.")

        if not callable(load_model):
            raise ValueError("Load model function must be callable.")

        with MachineLearningWorker._MODEL_LOCK:
            self._logger.debug(f"Starting to load model for {self.__class__.__name__}.")

            if MachineLearningWorker._MODEL is None:
                self._logger.debug(f"Calling provided `load_model` function.")
                MachineLearningWorker._MODEL = load_model()

            self._logger.debug(f"Finished loading model for {self.__class__.__name__}.")

            self._model = MachineLearningWorker._MODEL

    def _unload_model(self, unload_model: callable) -> NoReturn:
        """
        Attempts to unload the model by calling the given function. If the model is already unloaded, this method does
        nothing.

        :param unload_model: A callable function which unloads the model.
        """
        if unload_model is None:
            raise ValueError("Unload model function cannot be None.")

        if not callable(unload_model):
            raise ValueError("Unload model function must be callable.")

        with MachineLearningWorker._MODEL_LOCK:
            self._logger.debug(f"Starting to unload model for {self.__class__.__name__}")

            if MachineLearningWorker._MODEL is not None:
                self._logger.debug(f"Calling provided `unload_model` function.")
                unload_model(MachineLearningWorker._MODEL)

                del MachineLearningWorker._MODEL
                MachineLearningWorker._MODEL = None

            self._logger.debug(f"Finished unloading model for {self.__class__.__name__}.")

    def get_model(self) -> object:
        """
        Retrieves a reference to the MachineLearningWorker's model.

        :return: MachineLearningWorker's model.
        """
        with MachineLearningWorker._MODEL_LOCK:
            self._logger.debug(f"Returning a reference to the model of {self.__class__.__name__}.")
            return MachineLearningWorker._MODEL
