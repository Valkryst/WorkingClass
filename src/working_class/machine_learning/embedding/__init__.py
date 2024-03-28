import threading

from datetime import timedelta
from src.working_class.machine_learning import MachineLearningWorker
from theine import Cache
from theine.theine import CORES
from typing import Hashable, NoReturn


class EmbeddingWorker(MachineLearningWorker):
    def __init__(self, enable_cache: bool = True):
        """
        Constructs a new EmbeddingWorker.

        :param enable_cache: Whether to use an in-memory cache to temporarily store embeddings for reuse.
        """
        super().__init__()

        self._cache = None
        self._cache_lock = threading.Lock()
        self._cache_eviction_policy = "lru"
        self._cache_size = 1024
        self._cache_ttl = timedelta(seconds=60)

        if enable_cache:
            self._create_cache()

    def _create_cache(self) -> NoReturn:
        """
        Creates a new cache.
        """
        with self._cache_lock:
            self._logger.debug(f"Creating a new cache for {self.__class__.__name__}.")
            self._cache = Cache(
                self._cache_eviction_policy,
                self._cache_size
            )

    def add_to_cache(self, key: Hashable, value: any) -> NoReturn:
        """
        Adds a new key-value pair to the cache.

        :param key: Key to add.
        :param value: Value to add.
        """
        with self._cache_lock:
            if self._cache is None:
                return

            self._logger.debug(f"Adding KV pair to cache for {self.__class__.__name__}:\nKey: {key}\nValue: {value}.")
            self._cache.set(key, value, self._cache_ttl)

    def clear_cache(self) -> NoReturn:
        """
        Clears the cache.

        :raises RuntimeError: If the cache is not enabled.
        """
        with self._cache_lock:
            if self._cache is None:
                raise RuntimeError("Cache is not enabled.")

            self._logger.debug(f"Clearing cache for {self.__class__.__name__}.")

            if self._cache is not None:
                self._cache.clear()

    def enable_cache(self, enabled: bool) -> NoReturn:
        """
        En/disables the cache.

        :param enabled: Whether to enable the cache.
        """
        if enabled is None:
            raise ValueError("`enabled` cannot be None.")

        if not isinstance(enabled, bool):
            raise ValueError("`enabled` must be a boolean.")

        with self._cache_lock:
            self._logger.debug(f"Setting cache enabled to {enabled} for {self.__class__.__name__}.")

            if self._cache is None:
                if enabled:
                    self._create_cache()
                else:
                    pass
            else:
                if enabled:
                    pass
                else:
                    self._cache = None

    def remove_from_cache(self, key: Hashable) -> NoReturn:
        """
        Removes a key-value pair from the cache.

        :param key: Key to remove.

        :raises RuntimeError: If the cache is not enabled.
        """
        with self._cache_lock:
            if self._cache is None:
                raise RuntimeError("Cache is not enabled.")

            self._logger.debug(f"Removing KV pair from cache for {self.__class__.__name__}:\nKey: {key}.")
            self._cache.delete(key)

    def retrieve_from_cache(self, key: Hashable, default: any = None) -> any:
        """
        Retrieves a value from the cache.

        :param key: Key to retrieve.
        :param default: Default value to return, if the key is not found.

        :return: Value from the cache, or the default value.

        :raises RuntimeError: If the cache is not enabled.
        """
        with self._cache_lock:
            if self._cache is None:
                raise RuntimeError("Cache is not enabled.")

            self._logger.debug(f"Retrieving value from cache for {self.__class__.__name__}:\nKey: {key}.")
            return self._cache.get(key, default)

    def set_cache_eviction_policy(self, policy: str) -> NoReturn:
        """
        Defines the eviction policy for the cache.

        This will cause the cache to be cleared.

        :param policy: New cache eviction policy.
        """
        if policy is None:
            raise ValueError("Cache eviction policy cannot be None.")

        if policy not in CORES.keys():
            raise ValueError(f"Invalid cache eviction policy: {policy}\nAllowed policies: {CORES.keys()}")

        with self._cache_lock:
            self._logger.debug(f"Setting `_cache_eviction_policy` to {policy} for {self.__class__.__name__}.")
            self._cache_eviction_policy = policy

        self._create_cache()

    def set_cache_size(self, size: int) -> NoReturn:
        """
        Defines the maximum number of elements that can be stored in the cache.

        This will cause the cache to be cleared.

        :param size: New cache size.
        """
        if size is None:
            raise ValueError("Cache size cannot be None.")

        if not isinstance(size, int):
            raise ValueError(f"Cache size must be an integer.")

        if size <= 0:
            raise ValueError(f"Cache size must be a positive, non-zero value. The given value was {size}.")

        with self._cache_lock:
            self._logger.debug(f"Setting `_cache_size` to {size} for {self.__class__.__name__}.")
            self._cache_size = size

        self._create_cache()

    def set_cache_ttl(self, ttl: timedelta) -> NoReturn:
        """
        Defines the TTL (time-to-live) for elements in the cache.

        This will not affect existing elements in the cache, nor will it clear the cache. However, all future elements
        will be subject to this TTL.

        :param ttl: New cache TTL.
        """
        if ttl is None:
            raise ValueError("Cache TTL cannot be None.")

        if not isinstance(ttl, timedelta):
            raise ValueError(f"Cache TTL must be a timedelta object.")

        if ttl.total_seconds() <= 0:
            raise ValueError(
                f"Cache TTL must be a positive, non-zero value. The given value was {ttl.total_seconds()}.")

        with self._cache_lock:
            self._logger.debug(f"Setting `_cache_ttl` to {ttl} for {self.__class__.__name__}.")
            self._cache_ttl = ttl
