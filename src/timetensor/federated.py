class Client:

    def __init__(self, dataset, model=None):
        self.dataset = dataset
        self.model = model



def client_split(values, context, datetimes, splits, replace=False, seed=None, context_by_individuals=False):
    """splits individuals according to splits"""

    if seed is not None:
        np.random.seed(seed)

    individuals = values.shape[0]
    N = len(splits)

    remaining = list(range(individuals))
    for k in range(N):
        n = splits[k]*individuals #local number of individuals
        indices = np.random.choice(remaining, n, replace=False)
        if replace is False:
            remaining = [k for k in range remaining if k not in indices]

    if context_by_individuals:
        if context is None:
            return {f"node_{i}":(values[indices[i], :, :], None, datetimes) for i in range(N)}

        else:
            return {f"node_{i}":(values[indices[i], :, :], context[indices[i], :, :], datetimes) for i in range(N)}
    else:
        if context is None:
            return {f"node_{i}":(values[indices[i], :, :], None, datetimes) for i in range(N)}
        else:
            return {f"node_{i}":(values[indices[i], :, :], context, datetimes) for i in range(N)}




def partition_datasets(fetcher, path="datasets/", indiv_split=0.8, date_split=0.8, seed=None):
    values, context, datetimes = fetcher(path)
    data_dict = train_test_split(values, context, datetimes, indiv_split, date_split, seed)
    for key, (values, context, datetimes) in data_dict.items():
        torch.save(values, path + key + "_values.pt")
        if context is not None:
            torch.save(context, path + key + "_context.pt")
        torch.save(datetimes, path + key + "_datetimes.pt")