import numpy as np


def compute_avg_loss(counts, y, y_buckets):
    """Compute the loss of a sketch.
    Args:
        counts: estimated counts in each bucket, float - [num_buckets]
        y: true counts of each item, float - [num_items]
        y_bueckets: item -> bucket mapping - [num_items]

    Returns:
        Estimation error
    """
    assert np.sum(counts) == np.sum(y), "counts do not have all the flows!"
    assert len(y) == len(y_buckets)
    if len(y) == 0:
        return 0  # avoid division of 0
    loss = 0
    for i in range(len(y)):
        loss += np.abs(y[i] - counts[y_buckets[i]]) * y[i]
    return loss / np.sum(y)


def random_hash(y, n_buckets):
    """Sketch with a random hash
    Args:
        y: true counts of each item, float - [num_items]
        n_buckets: number of buckets

    Returns
        counts: estimated counts in each bucket, float - [num_buckets]
        loss: estimation error
        y_bueckets: item -> bucket mapping - [num_items]
    """
    counts = np.zeros(n_buckets)
    y_buckets = np.random.choice(np.arange(n_buckets), size=len(y))
    for i in range(len(y)):
        counts[y_buckets[i]] += y[i]
    loss = compute_avg_loss(counts, y, y_buckets)
    return counts, loss, y_buckets


def count_min(y, n_buckets, n_hash):
    """Count-Min
    Args:
        y: true counts of each item, float - [num_items]
        n_buckets: number of buckets
        n_hash: number of hash functions

    Returns:
        Estimation error
    """
    if len(y) == 0:
        return 0  # avoid division of 0

    counts_all = np.zeros((n_hash, n_buckets))
    y_buckets_all = np.zeros((n_hash, len(y)), dtype=int)
    for i in range(n_hash):
        counts, _, y_buckets = random_hash(y, n_buckets)
        counts_all[i] = counts
        y_buckets_all[i] = y_buckets

    loss = 0
    for i in range(len(y)):
        y_est = np.min([counts_all[k, y_buckets_all[k, i]] for k in range(n_hash)])
        loss += np.abs(y[i] - y_est) * y[i]
    return loss / np.sum(y)


def cutoff_countmin(y, n_buckets, b_cutoff, n_hashes):
    """Learned Count-Min
    Args:
        y: true counts of each item (sorted, largest first), float - [num_items]
        n_buckets: number of total buckets
        b_cutoff: number of unique buckets
        n_hash: number of hash functions

    Returns:
        loss_avg: estimation error
        space: space usage in bytes
    """
    assert b_cutoff <= n_buckets, "bucket cutoff cannot be greater than n_buckets"
    counts = np.zeros(n_buckets)
    if len(y) == 0:
        return 0  # avoid division of 0

    y_buckets = []
    for i in range(b_cutoff):
        if i >= len(y):
            break  # more unique buckets than # flows
        counts[i] += y[i]  # unique bucket for each flow
        y_buckets.append(i)

    loss_cf = compute_avg_loss(counts[:b_cutoff], y[:b_cutoff], y_buckets)  # loss = 0
    loss_cm = count_min(y[b_cutoff:], n_buckets - b_cutoff, n_hashes)

    loss_avg = (
        loss_cf * np.sum(y[:b_cutoff]) + loss_cm * np.sum(y[b_cutoff:])
    ) / np.sum(y)
    print("\tloss_cf %.2f\tloss_rd %.2f\tloss_avg %.2f" % (loss_cf, loss_cm, loss_avg))

    space = b_cutoff * 4 * 2 + (n_buckets - b_cutoff) * n_hashes * 4
    return loss_avg, space


def cutoff_countmin_wscore(y, scores, score_cutoff, n_cm_buckets, n_hashes):
    """Learned Count-Min (use predicted scores to identify heavy hitters)
    Args:
        y: true counts of each item (sorted, largest first), float - [num_items]
        scores: predicted scores of each item - [num_items]
        score_cutoff: threshold for heavy hitters
        n_cm_buckets: number of buckets of Count-Min
        n_hashes: number of hash functions

    Returns:
        loss_avg: estimation error
        space: space usage in bytes
    """
    if len(y) == 0:
        return 0  # avoid division of 0

    y_ccm = y[scores > score_cutoff]
    y_cm = y[scores <= score_cutoff]

    loss_cf = 0  # put y_ccm into cutoff buckets, no loss
    loss_cm = count_min(y_cm, n_cm_buckets, n_hashes)

    assert len(y_ccm) + len(y_cm) == len(y)
    loss_avg = (loss_cf * np.sum(y_ccm) + loss_cm * np.sum(y_cm)) / np.sum(y)
    print("\tloss_cf %.2f\tloss_rd %.2f\tloss_avg %.2f" % (loss_cf, loss_cm, loss_avg))

    space = len(y_ccm) * 4 * 2 + n_cm_buckets * n_hashes * 4
    return loss_avg, space


def cutoff_lookup(x, y, n_cm_buckets, n_hashes, d_lookup, y_cutoff, sketch="CountMin"):
    """Learned Count-Min (use predicted scores to identify heavy hitters)
    Args:
        x: feature of each item - [num_items]
        y: true counts of each item, float - [num_items]
        n_cm_buckets: number of buckets of Count-Min
        n_hashes: number of hash functions
        d_lookup: x[i] -> y[i] look up table
        y_cutoff: threshold for heavy hitters
        sketch: type of sketch (CountMin or CountSketch)

    Returns:
        loss_avg: estimation error
        space: space usage in bytes
    """
    if len(y) == 0:
        return 0  # avoid division of 0

    y_ccm = []
    y_cm = []
    for i in range(len(y)):
        if x[i] in d_lookup:
            if d_lookup[x[i]] > y_cutoff:
                y_ccm.append(y[i])
            else:
                y_cm.append(y[i])
        else:
            y_cm.append(y[i])

    loss_cf = 0  # put y_ccm into cutoff buckets, no loss
    if sketch == "CountMin":
        loss_cm = count_min(y_cm, n_cm_buckets, n_hashes)
    elif sketch == "CountSketch":
        loss_cm = count_sketch(y_cm, n_cm_buckets, n_hashes)
    else:
        assert False, "unknown sketch type"

    assert len(y_ccm) + len(y_cm) == len(y)
    loss_avg = (loss_cf * np.sum(y_ccm) + loss_cm * np.sum(y_cm)) / np.sum(y)
    print("\tloss_cf %.2f\tloss_rd %.2f\tloss_avg %.2f" % (loss_cf, loss_cm, loss_avg))
    print("\t# uniq", len(y_ccm), "# cm", len(y_cm))

    space = len(y_ccm) * 4 * 2 + n_cm_buckets * n_hashes * 4
    return loss_avg, space


def two_cutoff_lookup(
    x,
    y,
    n_cm_buckets_one,
    n_cm_buckets_two,
    n_hashes_one,
    n_hashes_two,
    d_lookup,
    y_cutoff_one,
    y_cutoff_two,
    sketch="CountMin",
):
    """Learned Count-Min (use predicted scores to identify heavy hitters)
    Args:
        x: feature of each item - [num_items]
        y: true counts of each item, float - [num_items]
        n_cm_buckets_one: for first threshold
        n_cm_buckets_two: number of buckets of Count-Min second threshold
        n_hashes_one: number of hash functions for first threshold
        n_hashes_two: number of hash functions for second threshold
        d_lookup: x[i] -> y[i] look up table
        y_cutoff_one: threshold for heavy hitters
        sketch: type of sketch (CountMin or CountSketch)

    Returns:
        loss_avg: estimation error
        space: space usage in bytes
    """
    if len(y) == 0:
        return 0  # avoid division of 0

    y_ccm = []
    y_cm_one = []
    y_cm_two = []
    for i in range(len(y)):
        if x[i] in d_lookup:
            if d_lookup[x[i]] > y_cutoff_one:
                y_ccm.append(y[i])
            if y_cutoff_one >= d_lookup[x[i]] >= y_cutoff_two:
                y_cm_one.append(y[i])
            else:
                y_cm_two.append(y[i])
        else:
            y_cm_two.append(y[i])

    loss_cf = 0  # put y_ccm into cutoff buckets, no loss
    if sketch == "CountMin":
        loss_cm_one = count_min(y_cm_one, n_cm_buckets_one, n_hashes_one)
        loss_cm_two = count_min(y_cm_two, n_cm_buckets_two, n_hashes_two)
    elif sketch == "CountSketch":
        loss_cm_one = count_sketch(y_cm_one, n_cm_buckets_one, n_hashes_one)
        loss_cm_two = count_sketch(y_cm_two, n_cm_buckets_two, n_hashes_two)
    else:
        assert False, "unknown sketch type"

    assert len(y_ccm) + len(y_cm_one) == len(y)
    loss_avg = (
        loss_cf * np.sum(y_ccm)
        + loss_cm_one * np.sum(y_cm_one)
        + loss_cm_two * np.sum(y_cm_two)
    ) / np.sum(y)
    print(
        "\tloss_cf %.2f\tloss_rd %.2f\tloss_avg %.2f" % (loss_cf, loss_cm_one, loss_avg)
    )
    print("\t# uniq", len(y_ccm), "# cm", len(y_cm_one))

    space = len(y_ccm) * 4 * 2 + n_cm_buckets_one * n_hashes_one * 4 + n_cm_buckets_two * n_hashes_two * 4
    return loss_avg, space


def random_hash_with_sign(y, n_buckets):
    """Assign items in y into n_buckets, randomly pick a sign for each item
    Args:
        y: true counts of each item, float - [num_items]
        n_buckets: number of buckets

    Returns
        counts: estimated counts in each bucket, float - [num_buckets]
        loss: estimation error
        y_bueckets: item -> bucket mapping - [num_items]
    """
    counts = np.zeros(n_buckets)
    y_buckets = np.random.choice(np.arange(n_buckets), size=len(y))
    y_signs = np.random.choice([-1, 1], size=len(y))
    for i in range(len(y)):
        counts[y_buckets[i]] += y[i] * y_signs[i]
    return counts, y_buckets, y_signs


def count_sketch(y, n_buckets, n_hash):
    """Count-Sketch
    Args:
        y: true counts of each item, float - [num_items]
        n_buckets: number of buckets
        n_hash: number of hash functions

    Returns:
        Estimation error
    """
    if len(y) == 0:
        return 0  # avoid division of 0

    counts_all = np.zeros((n_hash, n_buckets))
    y_buckets_all = np.zeros((n_hash, len(y)), dtype=int)
    y_signs_all = np.zeros((n_hash, len(y)), dtype=int)
    for i in range(n_hash):
        counts, y_buckets, y_signs = random_hash_with_sign(y, n_buckets)
        counts_all[i] = counts
        y_buckets_all[i] = y_buckets
        y_signs_all[i] = y_signs

    loss = 0
    for i in range(len(y)):
        y_est = np.median(
            [
                y_signs_all[k, i] * counts_all[k, y_buckets_all[k, i]]
                for k in range(n_hash)
            ]
        )
        loss += np.abs(y[i] - y_est) * y[i]
    return loss / np.sum(y)


def cutoff_countsketch(y, n_buckets, b_cutoff, n_hashes):
    """Learned Count-Sketch
    Args:
        y: true counts of each item (sorted, largest first), float - [num_items]
        n_buckets: number of total buckets
        b_cutoff: number of unique buckets
        n_hash: number of hash functions

    Returns:
        loss_avg: estimation error
        space: space usage in bytes
    """
    assert b_cutoff <= n_buckets, "bucket cutoff cannot be greater than n_buckets"
    counts = np.zeros(n_buckets)
    if len(y) == 0:
        return 0  # avoid division of 0

    y_buckets = []
    for i in range(b_cutoff):
        if i >= len(y):
            break  # more unique buckets than # flows
        counts[i] += y[i]  # unique bucket for each flow
        y_buckets.append(i)

    loss_cf = compute_avg_loss(counts[:b_cutoff], y[:b_cutoff], y_buckets)  # loss = 0
    loss_cs = count_sketch(y[b_cutoff:], n_buckets - b_cutoff, n_hashes)

    loss_avg = (
        loss_cf * np.sum(y[:b_cutoff]) + loss_cs * np.sum(y[b_cutoff:])
    ) / np.sum(y)
    print("\tloss_cf %.2f\tloss_rd %.2f\tloss_avg %.2f" % (loss_cf, loss_cs, loss_avg))

    space = b_cutoff * 4 * 2 + (n_buckets - b_cutoff) * n_hashes * 4
    return loss_avg, space


def cutoff_countsketch_wscore(y, scores, score_cutoff, n_cs_buckets, n_hashes):
    """Learned Count-Sketch (use predicted scores to identify heavy hitters)
    Args:
        y: true counts of each item (sorted, largest first), float - [num_items]
        scores: predicted scores of each item - [num_items]
        score_cutoff: threshold for heavy hitters
        n_cs_buckets: number of buckets of Count-Sketch
        n_hashes: number of hash functions

    Returns:
        loss_avg: estimation error
        space: space usage in bytes
    """
    if len(y) == 0:
        return 0  # avoid division of 0

    y_ccs = y[scores > score_cutoff]
    y_cs = y[scores <= score_cutoff]

    loss_cf = 0  # put y_ccs into cutoff buckets, no loss
    loss_cs = count_sketch(y_cs, n_cs_buckets, n_hashes)

    assert len(y_ccs) + len(y_cs) == len(y)
    loss_avg = (loss_cf * np.sum(y_ccs) + loss_cs * np.sum(y_cs)) / np.sum(y)
    print("\tloss_cf %.2f\tloss_rd %.2f\tloss_avg %.2f" % (loss_cf, loss_cs, loss_avg))

    space = len(y_ccs) * 4 * 2 + n_cs_buckets * n_hashes * 4
    return loss_avg, space


def cutoff_countsketch_wscore_two(y, scores, score_cutoff_mid, n_cs_buckets_mid, n_hashes_mid, score_cutoff_low, n_cs_buckets_low, n_hashes_low):
    """Learned Count-Sketch (use predicted scores to identify heavy hitters)
    Args:
        y: true counts of each item (sorted, largest first), float - [num_items]
        scores: predicted scores of each item - [num_items]
        score_cutoff_low: threshold for non-heavy hitters
        n_cs_buckets_low: number of buckets for non-heavy of Count-Sketch
        n_hashes_low: number of hash functions for non-heavy
        score_cutoff_mid: threshold for semi-heavy hitters
        n_cs_buckets_mid: number of buckets for semi-heavy of Count-Sketch
        n_hashes_mid: number of hash functions for semi-heavy

    Returns:
        loss_avg: estimation error
        space: space usage in bytes
    """
    if len(y) == 0:
        return 0  # avoid division of 0

    y_ccs_hh = y[scores > score_cutoff_mid]

    y_cs_semi_hh = y[scores <= score_cutoff_mid]
    y_cs_semi_hh = y_cs_semi_hh[y_cs_semi_hh > score_cutoff_low]

    y_cs = y[scores <= score_cutoff_low]

    loss_cf = 0  # put y_ccs into cutoff buckets, no loss

    loss_cs_semi = count_sketch(y_cs_semi_hh, n_cs_buckets_mid, n_hashes_mid)
    loss_cs = count_sketch(y_cs, n_cs_buckets_low, n_hashes_low)

    print(len(y_ccs_hh) + len(y_cs) + len(y_cs_semi_hh),len(y))
    assert len(y_ccs_hh) + len(y_cs)+len(y_cs_semi_hh) == len(y)
    loss_avg = (loss_cf * np.sum(y_ccs_hh) + loss_cs * np.sum(y_cs)+ + loss_cs_semi * np.sum(y_cs_semi_hh)) / np.sum(y)
    print("\tloss_cf %.2f\tloss_rd %.2f\tloss_avg %.2f" % (loss_cf, loss_cs, loss_avg))

    print(n_cs_buckets_mid, n_hashes_mid, score_cutoff_low, n_cs_buckets_low,
          n_hashes_low)

    space = len(y_ccs_hh) * 4 * 2 + n_cs_buckets_low * n_hashes_low * 4 + n_cs_buckets_mid * n_hashes_mid * 4
    return loss_avg, space


def order_y_wkey(y, results, key, n_examples=0):
    """ Order items based on the scores in results """
    print("loading results from %s" % results)
    results = np.load(results)
    pred_prob = results[key].squeeze()
    if n_examples:
        pred_prob = pred_prob[:n_examples]
    idx = np.argsort(pred_prob)[::-1]
    assert len(idx) == len(y)
    return y[idx], pred_prob[idx]


def order_y_wkey_list(y, results_list, key):
    """ Order items based on the scores in results """
    pred_prob = np.array([])
    for results in results_list:
        results = np.load(results)
        pred_prob = np.concatenate((pred_prob, results[key].squeeze()))
    idx = np.argsort(pred_prob)[::-1]
    assert len(idx) == len(y)
    return y[idx], pred_prob[idx]
