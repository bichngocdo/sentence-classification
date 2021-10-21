class AverageMeter(object):
    def __init__(self, default='avg'):
        self.sum = 0
        self.count = 0
        self.avg = 0

        assert default in {'sum', 'avg'}
        self.default = default

    def reset(self):
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def result(self):
        return getattr(self, self.default)


def batch_to_device(batch, device):
    for k, v in batch.items():
        if hasattr(v, 'to'):
            batch[k] = v.to(device)
    return batch
