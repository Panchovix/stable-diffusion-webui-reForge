import logging
import math
from typing import Optional

import torch
import ldm_patched.modules.model_management
from .base import WeightAdapterBase, weight_decompose

class AbbaAdapter(WeightAdapterBase):
    """
    Weight adapter for ABBA (Hadamard Product Adaptation).
    ABBA models the weight update as a Hadamard product of two independent low-rank matrices:
    ΔW = s * ((B1 @ A1) * (B2 @ A2))
    """
    name = "abba"

    def __init__(self, loaded_keys, weights):
        self.loaded_keys = loaded_keys
        self.weights = weights

    @classmethod
    def load(
        cls,
        x: str,
        lora: dict[str, torch.Tensor],
        alpha: float,
        dora_scale: torch.Tensor, # Not used by ABBA, but part of the interface
        loaded_keys: set[str] = None,
    ) -> Optional["AbbaAdapter"]:
        if loaded_keys is None:
            loaded_keys = set()

        # Define the expected keys for the two adapter pairs in ABBA
        down1_name = f"{x}.lora_down1.weight"
        up1_name = f"{x}.lora_up1.weight"
        down2_name = f"{x}.lora_down2.weight"
        up2_name = f"{x}.lora_up2.weight"

        # Check for the presence of the primary keys to identify an ABBA module
        if up1_name in lora and up2_name in lora:
            logging.debug(f"Loading ABBA weights for module: {x}")

            # Load the four weight tensors
            down1 = lora[down1_name]
            up1 = lora[up1_name]
            down2 = lora[down2_name]
            up2 = lora[up2_name]

            # Store the weights and the alpha value
            # The structure is (up1, down1, up2, down2, alpha, dora_scale)
            packaged_weights = (up1, down1, up2, down2, alpha, dora_scale)

            # Mark all associated keys as loaded
            current_loaded_keys = {down1_name, up1_name, down2_name, up2_name}
            loaded_keys.update(current_loaded_keys)

            return cls(current_loaded_keys, packaged_weights)
        else:
            return None

    def calculate_weight(
        self,
        weight: torch.Tensor,
        key: str,
        strength: float,
        strength_model: float,
        offset,
        function,
        intermediate_dtype=torch.float32,
        original_weight=None,
    ):
        """
        Calculates the final weight by applying the ABBA update.
        """
        if strength == 0.0:
            return weight

        # Unpack the stored weights and alpha
        up1_w, down1_w, up2_w, down2_w, alpha, dora_scale = self.weights

        # Cast tensors to the appropriate device and dtype for calculation
        device = weight.device
        up1 = ldm_patched.modules.model_management.cast_to_device(up1_w, device, intermediate_dtype)
        down1 = ldm_patched.modules.model_management.cast_to_device(down1_w, device, intermediate_dtype)
        up2 = ldm_patched.modules.model_management.cast_to_device(up2_w, device, intermediate_dtype)
        down2 = ldm_patched.modules.model_management.cast_to_device(down2_w, device, intermediate_dtype)
        
        # Get ranks from the down matrices
        r1 = down1.shape[0]
        r2 = down2.shape[0]
        
        if r1 == 0 or r2 == 0:
            # If either rank is zero, the update is zero.
            return weight

        # Define the scaling factor based on ABBA's formulation (alpha_LORA**2 / sqrt(r1 * r2))
        if alpha is not None:
            scale = alpha**2 / math.sqrt(r1 * r2)
        else:
            scale = 1.0

        try:
            # Calculate the first low-rank matrix update
            delta1 = torch.mm(up1.flatten(start_dim=1), down1.flatten(start_dim=1))
            
            # Calculate the second low-rank matrix update
            delta2 = torch.mm(up2.flatten(start_dim=1), down2.flatten(start_dim=1))

            # Combine them using the Hadamard (element-wise) product
            lora_diff = delta1 * delta2
            
            # Reshape the final difference to match the original weight's shape
            lora_diff = lora_diff.reshape(weight.shape)

            if dora_scale is not None:
                weight = weight_decompose(
                    dora_scale,
                    weight,
                    lora_diff,
                    scale,
                    strength,
                    intermediate_dtype,
                    function,
                )
            else:
                # Apply strength and scale, then add to the original weight
                final_diff = function((strength * scale) * lora_diff).to(weight.dtype)
                weight += final_diff
        except Exception as e:
            logging.error("ERROR {} {} {}".format(self.name, key, e))
            
        return weight