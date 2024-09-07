from typing import List, Union

import torch

from diffusers.loaders import TextualInversionLoaderMixin
from diffusers.utils import logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class CustomTextualInversionMixin(TextualInversionLoaderMixin):
    r"""
    add a add_tokens function to the TextualInversionLoaderMixin
    https://github.com/huggingface/diffusers/blob/v0.27.0/src/diffusers/loaders/textual_inversion.py#L112C35-L113C5
    """

    def add_tokens(
        self,
        placeholder_tokens: Union[str, List[str]],
        initializer_tokens: Union[str, List[str]] = None,
        num_vectors_per_token: Union[int, List[int]] = 1,
        initialize_parameters: bool = False,
    ):
        r"""Add token for training."""
        if not isinstance(placeholder_tokens, list):
            placeholder_tokens = [placeholder_tokens]
        if not isinstance(initializer_tokens, list):
            initializer_tokens = [initializer_tokens] * len(placeholder_tokens)
        if not isinstance(num_vectors_per_token, list):
            num_vectors_per_token = [num_vectors_per_token] * len(placeholder_tokens)

        assert len(placeholder_tokens) == len(
            num_vectors_per_token
        ), "placeholder_token should be the same length as num_vectors_per_token"
        assert len(placeholder_tokens) == len(
            initializer_tokens
        ), "placeholder_token should be the same length as initialize_token"

        # add tokens into tokenizer
        new_embeds_ids = []
        for p, i, n in zip(placeholder_tokens, initializer_tokens, num_vectors_per_token):
            new_ids = self._add_token(p, i, n, initialize_parameters)
            new_embeds_ids += new_ids

        return sorted(new_embeds_ids)

    def _add_token(
        self, placeholder_token: str, initializer_token: str, num_vectors_per_token: int, initialize_parameters: bool
    ):
        r"""Add placeholder tokens to the tokenizer.
        borrowed from https://github.com/huggingface/diffusers/blob/main/
        examples/textual_inversion/textual_inversion.py#L669 # noqa
        """
        assert num_vectors_per_token >= 1, "num_vectors_per_token should be greater than 0"

        placeholder_tokens = [placeholder_token]

        # create dummy tokens for multi-vector
        additional_tokens = []
        for i in range(1, num_vectors_per_token):
            additional_tokens.append(f"{placeholder_token}_{i}")
        placeholder_tokens += additional_tokens

        # add dummy tokens into tokenizer
        num_added_tokens = self.tokenizer.add_tokens(placeholder_tokens)
        assert num_added_tokens == num_vectors_per_token, (
            f"The tokenizer already contains the token {placeholder_token}. Please pass a different"
            f" `placeholder_token` that is not already in the tokenizer."
        )

        # Convert the initializer_token, placeholder_token to ids
        # Check if initializer_token is a single token or a sequence of tokens
        token_ids = self.tokenizer.encode(initializer_token, add_special_tokens=False)
        assert len(token_ids) == 1, "The initializer token must be a single token."

        initializer_token_id = token_ids[0]
        placeholder_token_ids = self.tokenizer.convert_tokens_to_ids(placeholder_tokens)

        # skip initialization on text_encoder for trained models
        if initialize_parameters:
            # Resize the token embeddings as we are adding new special tokens to the tokenizer
            self.text_encoder.resize_token_embeddings(len(self.tokenizer))

            # Initialise the newly added placeholder token with the embeddings of the initializer token
            token_embeds = self.text_encoder.get_input_embeddings().weight.data
            with torch.no_grad():
                for token_id in placeholder_token_ids:
                    token_embeds[token_id] = token_embeds[initializer_token_id].clone()

        return placeholder_token_ids
