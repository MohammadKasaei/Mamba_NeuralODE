"""Counterfactual query control for synthetic associative recall."""

class ChangedRecallQuery:
    """Replace the query by another present key while retaining original labels.

    A model retrieving the original query should lose accuracy; a context-value
    frequency predictor is unchanged. No answer token is inserted in the input.
    """
    def __init__(self, data, symbols):
        self.data, self.symbols = data, symbols

    def batch(self, batch_size, length, seed, split="val"):
        x, y = self.data.batch(batch_size, length, seed, split)
        x = x.clone()
        for row in x:
            context = row[:-2]
            keys = context[(context >= 4) & (context < 4+self.symbols)]
            if len(keys) < 2:
                raise ValueError("Changing a recall query requires at least two keys")
            index = (keys == row[-2]).nonzero().flatten().item()
            row[-2] = keys[(index+1)%len(keys)]
        return x, y


class RemovedMemoryCue:
    """Remove task-specific memory cues, preserving the original answer labels.

    Copying loses its MARK tokens; induction loses the earlier query occurrence.
    This is an out-of-distribution sensitivity probe, not another test accuracy.
    """
    def __init__(self, data, task):
        self.data, self.task = data, task

    def batch(self, batch_size, length, seed, split="val"):
        x, y = self.data.batch(batch_size, length, seed, split)
        x = x.clone()
        if self.task == "selective_copying":
            x[x == 1] = 4
        elif self.task == "induction":
            for row in x:
                q = row[-1]
                pos = (row[:-1] == q).nonzero().flatten()
                if len(pos) != 1:
                    raise ValueError("Induction probe requires one previous query occurrence")
                replacement = row[:-1][row[:-1] != q][0].clone()
                row[pos] = replacement
        else:
            raise ValueError(f"Unsupported memory-cue task: {self.task}")
        return x, y
