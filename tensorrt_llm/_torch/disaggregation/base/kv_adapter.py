from abc import ABC

"""
Convenient interface for the following functions.

CacheManager:
- getCacheType()
- isEnableBlockReuse()
- getDataType()

BlockManager:
- getTokensPerBlock()
- getBlockSize()
- getNumLayers()
- getPoolWindowSize()
- getNumPools()
- getNumLayers()
- getStreamDevice()

"""


class KVAdapter(ABC): ...
