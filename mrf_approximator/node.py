
class Node:
    def __init__(self, mean=0.0, count=0):
        self.mean = mean
        self.count = count

    def update(self, r):
        self.count += 1
        self.mean = self.mean * (self.count - 1) / self.count + r / self.count

    def __str__(self):
        return "Point Mean = {0}, Point Count = {1}".format(self.mean, self.count)

  