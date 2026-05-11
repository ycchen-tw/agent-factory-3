import numpy as np
import torch
from itertools import chain, repeat, accumulate

class DataCollatorWithPacking:
    """Data collator with sequence packing for efficient training."""
    
    def __init__(self, pad_token_id: int = 0, pad_to_multiple_of: int = 128):
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of
    
    def __call__(self, instances):
        seq_lengths = [len(inst['input_ids']) for inst in instances]
        total_length = sum(seq_lengths)
        num_padding = (self.pad_to_multiple_of - total_length % self.pad_to_multiple_of) % self.pad_to_multiple_of
        
        # Pack core fields
        batch = {
            'input_ids': self._pack_and_pad(
                chain.from_iterable(inst['input_ids'] for inst in instances),
                self.pad_token_id, num_padding, torch.long
            ),
            'attention_mask': self._pack_and_pad(
                repeat(True, total_length),
                False, num_padding, torch.int32
            ),
            'position_ids': self._pack_position_ids(
                chain.from_iterable(range(l) for l in seq_lengths),
                num_padding, torch.long
            ),
        }

        # Add cumulative sequence lengths for packed attention
        cu_seqlens = [0] + list(accumulate(seq_lengths))
        if num_padding > 0:
            cu_seqlens.append(cu_seqlens[-1] + num_padding)
        batch['cu_seqlens'] = torch.tensor(cu_seqlens, dtype=torch.long)
        
        # Pack optional fields
        if 'advantages' in instances[0]:
            packed = chain.from_iterable(
                (adv if isinstance(adv, list) else repeat(adv, len(inst['input_ids'])))
                for inst, adv in ((inst, inst['advantages']) for inst in instances)
            )
            batch['advantages'] = self._pack_and_pad(packed, 0.0, num_padding, torch.float32)

        for field, dtype in [('completion_mask', torch.int32),
                             ('gen_logprobs', torch.float32),
                             ('gen_entropy', torch.float32)]:
            if field in instances[0]:
                packed = chain.from_iterable(inst[field] for inst in instances)
                batch[field] = self._pack_and_pad(packed, 0, num_padding, dtype)
                
        if 'routing_indices' in instances[0]:
            # routing_indices: list of np.ndarray [Ti, L, K] uint8
            arrays = [inst['routing_indices'] for inst in instances]
            if num_padding > 0:
                L, K = arrays[0].shape[1], arrays[0].shape[2]
                arrays.append(np.zeros((num_padding, L, K), dtype=np.uint8))
            packed_np = np.concatenate(arrays, axis=0)
            batch['routing_indices'] = torch.from_numpy(packed_np).to(torch.long).unsqueeze(0)

        if 'idx' in instances[0]:
            batch['indices'] = torch.tensor([inst['idx'] for inst in instances], dtype=torch.long)

        if 'sample_id' in instances[0]:
            batch['sample_id'] = [inst['sample_id'] for inst in instances]

        # Metadata for loss aggregation (not packed, just collected as lists)
        for field in ['seq_token_count', 'prompt_token_count', 'prompt_sequence_count', 'prompt_id', 'divisor']:
            if field in instances[0]:
                batch[field] = [inst[field] for inst in instances]

        return batch
    
    def _pack_and_pad(self, values, pad_value, num_padding, dtype):
        """Pack values into tensor with padding."""
        packed = list(values)
        if num_padding > 0:
            packed.extend(repeat(pad_value, num_padding))
        return torch.tensor(packed, dtype=dtype).unsqueeze(0)

    def _pack_position_ids(self, values, num_padding, dtype):
        """Pack position_ids with sequential padding (0, 1, 2, ..., num_padding-1).

        This ensures padding tokens form a separate sequence when using packed sequence
        detection, as the jump from the last real sequence to position 0 triggers a new
        sequence boundary.
        """
        packed = list(values)
        if num_padding > 0:
            packed.extend(range(num_padding))
        return torch.tensor(packed, dtype=dtype).unsqueeze(0)