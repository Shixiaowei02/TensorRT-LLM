from enum import Enum

import torch

from ..dsa import DSAtrtllmAttentionMetadata


class MewtwoTrtllmAttentionMetadata(DSAtrtllmAttentionMetadata):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_layer_types = len(set(self.compress_ratio))
        self.layer_type_mapping = {
            compress_ratio: i for i, compress_ratio in enumerate(set(self.compress_ratio))
        }

    def __post_init__(self):
        super().__post_init__()
        capture_graph = self.is_cuda_graph

        # Create buffers for different compress ratios
        self.all_kv_lens_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (
                self.num_layer_types,
                self.max_num_sequences,
            ),
            cache_name="all_kv_lens_cuda",
            dtype=torch.int,
            capture_graph=capture_graph,
        )
        self.all_kv_lens = torch.empty_like(self.all_kv_lens_cuda, device="cpu", pin_memory=True)

        # Create buffers for the compressor

    def prepare(self):
        super().prepare()

        # Prepare buffers for the compressor


class MewtwoAttentionType(Enum):
    SWA = 0
    COMPRESS = 1
    COMPRESSOR_STATE = 2
    COMPRESSOR_SCORE = 3
    INDEXER_COMPRESS = 4
    INDEXER_COMPRESSOR_STATE = 5
    INDEXER_COMPRESSOR_SCORE = 6
