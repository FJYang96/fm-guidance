from .abstract_sampler import Sampler


class NominalSampler(Sampler):
    def __init__(self, model):
        self.model = model
        self.conditions = None

    def forward(self, t, x, ts=1):
        # TODO: I forgot why I added the ts here...
        return self.model(t / ts, x, self.conditions) / ts
