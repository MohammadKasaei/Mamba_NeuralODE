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
