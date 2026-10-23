import random

from torch.utils.data import Sampler


class ClassBalancedBatchSampler(Sampler):
    """Deterministic batch sampler with joint red/green constraints."""

    def __init__(
        self,
        dataset,
        batch_size,
        min_red=1,
        min_green=1,
        shuffle=True,
        seed=0,
        green_requirement_probability=1.0,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.min_red = int(min_red)
        self.min_green = int(min_green)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.green_requirement_probability = float(green_requirement_probability)
        if not 0.0 <= self.green_requirement_probability <= 1.0:
            raise ValueError("green_requirement_probability must be in [0, 1]")
        self.epoch = 0
        self.num_samples = len(dataset)

        self.contains_red = getattr(dataset, "contains_red", None)
        self.contains_green = getattr(dataset, "contains_green", None)
        self.red_indices = getattr(dataset, "red_indices", [])
        self.green_indices = getattr(dataset, "green_indices", [])

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(self.num_samples))
        if self.shuffle:
            rng.shuffle(indices)

        batch = []
        for index in indices:
            batch.append(index)
            if len(batch) == self.batch_size:
                yield self._enforce_constraints(batch, rng)
                batch = []

        if batch:
            yield self._enforce_constraints(batch, rng)

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _enforce_constraints(self, batch, rng):
        batch = list(batch)
        if not self.contains_red or not self.contains_green:
            return batch

        requirements = [
            {
                "name": "red",
                "flags": self.contains_red,
                "candidates": self.red_indices,
                "minimum": min(self.min_red, len(batch)),
            }
        ]
        if rng.random() < self.green_requirement_probability:
            requirements.append(
                {
                    "name": "green",
                    "flags": self.contains_green,
                    "candidates": self.green_indices,
                    "minimum": min(self.min_green, len(batch)),
                }
            )
        requirements = [
            requirement
            for requirement in requirements
            if requirement["minimum"] > 0 and requirement["candidates"]
        ]
        if not requirements:
            return batch

        # Enforce requirements jointly. The previous sequential implementation
        # could insert red and then replace that same sample while adding green.
        max_replacements = len(batch) * max(1, len(requirements))
        for _ in range(max_replacements):
            counts = {
                requirement["name"]: sum(
                    requirement["flags"][index] for index in batch
                )
                for requirement in requirements
            }
            missing = [
                requirement
                for requirement in requirements
                if counts[requirement["name"]] < requirement["minimum"]
            ]
            if not missing:
                return batch

            target = missing[0]
            available = [
                index for index in target["candidates"] if index not in batch
            ]
            if not available:
                available = list(target["candidates"])

            best_score = max(
                sum(requirement["flags"][index] for requirement in missing)
                for index in available
            )
            candidates = [
                index
                for index in available
                if sum(
                    requirement["flags"][index] for requirement in missing
                )
                == best_score
            ]
            replacement = rng.choice(candidates)

            valid_slots = []
            for slot, old_index in enumerate(batch):
                preserves_satisfied = True
                for requirement in requirements:
                    name = requirement["name"]
                    new_count = (
                        counts[name]
                        - int(requirement["flags"][old_index])
                        + int(requirement["flags"][replacement])
                    )
                    if (
                        counts[name] >= requirement["minimum"]
                        and new_count < requirement["minimum"]
                    ):
                        preserves_satisfied = False
                        break
                if preserves_satisfied:
                    valid_slots.append(slot)
            if not valid_slots:
                break

            contribution = {
                slot: sum(
                    requirement["flags"][batch[slot]]
                    for requirement in requirements
                )
                for slot in valid_slots
            }
            minimum_contribution = min(contribution.values())
            best_slots = [
                slot
                for slot in valid_slots
                if contribution[slot] == minimum_contribution
            ]
            batch[rng.choice(best_slots)] = replacement

        return batch
