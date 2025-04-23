


def shapley_(x, background, model, players, j, n1, n2):

    coalitions = sample_coalitions(j, players, n1) # (n1, J-|j|)
    batch = replace(x, coalitions, background, n2) # (n1, n2, *dim(x))
    predictions = model(batch.view(-1, x.shape))  # (n1 * n2, *dim(x))
    shapley_value = compute_shapley(predictions)


