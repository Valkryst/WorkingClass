import copy
import threading
import logging

from typing import NoReturn


class Worker(threading.Thread):
    """
    Base class for a Worker.

    A Worker is a thread-safe object which performs some work on a given input. It may produce an output, but that is
    dependent on the implementation of the subclass.
    """

    def __init__(self):
        """
        Constructs a new Worker.
        """
        super().__init__()

        self._logger = logging.getLogger(__name__)

        self._output = None
        self._output_lock = threading.Lock()

    def run(self) -> NoReturn:
        """
        Performs some work on a given input.

        This method is called when the Worker's start() method is called. It should be overridden by all subclasses, and
        should call this superclass method at the start of the subclass method.
        """
        self._logger.debug(f"{self.__class__.__name__} started.")
        self._set_output(None)

    def get_output(self) -> NoReturn:
        """
        Retrieves the Worker's output, if any.

        :return: Worker's output, if any.
        """
        with self._output_lock:
            self._logger.debug("Returning a deepcopy of `_output`.")
            return copy.deepcopy(self._output)

    def _set_output(self, output: any) -> any:
        """
        Sets the Worker's output.

        :param output: New output.
        """
        with self._output_lock:
            self._logger.debug(f"Setting `_output` to:\n{output}.")
            self._output = output
