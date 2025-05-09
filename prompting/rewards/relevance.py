import time
from typing import ClassVar, Optional

import numpy as np
from angle_emb import AnglE
from pydantic import ConfigDict
from scipy import spatial

from prompting.rewards.reward import BaseRewardModel, BatchRewardOutput
from shared import settings
from shared.dendrite import DendriteResponseEvent

shared_settings = settings.shared_settings


class RelevanceRewardModel(BaseRewardModel):
    threshold: Optional[float] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)
    embedding_model: ClassVar[AnglE] = AnglE.from_pretrained(
        "WhereIsAI/UAE-Large-V1", pooling_strategy="cls", device=shared_settings.NEURON_DEVICE
    ).to(shared_settings.NEURON_DEVICE)

    async def reward(
        self, reference: str, response_event: DendriteResponseEvent, model_manager=None, **kwargs
    ) -> BatchRewardOutput:
        """Calculate the cosine similarity between sentence embeddings of the reference and completions.

        We subtract a baseline score which is what an empty string would get (a failed completion).
        This is usually around 0.35. We also clip the rewards between 0 and 1.
        The maximum effective score is around 0.65.
        """
        if not reference:
            raise Exception("Reference is empty - something went wrong during the reference generation")
        reference_embedding = self.embedding_model.encode(reference, to_numpy=True)
        reference_emb_flatten = reference_embedding.flatten()
        rewards: list[float] = []
        timings: list[float] = []
        completions: list[str] = response_event.completions
        # baseline is the cosine similarity between the reference and an empty string
        baseline = 1 - float(
            spatial.distance.cosine(reference_emb_flatten, self.embedding_model.encode("", to_numpy=True).flatten())
        )

        for comp in completions:
            if len(comp) == 0:
                rewards.append(0)
                timings.append(0)
                continue
            t0 = time.time()
            emb = self.embedding_model.encode(comp, to_numpy=True)
            # Calculate cosine similarity between reference and completion embeddings, and subtract baseline
            score = 1 - float(spatial.distance.cosine(reference_emb_flatten, emb.flatten() - baseline))

            rewards.append(score)
            timings.append(time.time() - t0)

        output = BatchRewardOutput(
            rewards=np.clip(np.array(rewards), 0, 1),
            timings=np.array(timings),
            threshold=self.threshold,
        )

        return output
