"""Iterated physical batches, with recollection on each new physical circuit."""


def iterate(initial, collect, plan, apply, eligible, rounds, on_round=None):
    """One apply per proposed batch; no per-LAC physical application.

    collect(current, index) MUST operate on the accepted physical circuit.
    A rejected batch never becomes the starting point of the next round.
    """
    current = initial; history = []
    for index in range(rounds):
        information = collect(current, index)
        proposal = plan(current, information, index)
        if not proposal:
            history.append(dict(round=index,accepted=False,reason='no_batch'))
            break
        candidate, details = apply(current, information, proposal, index)
        valid = eligible(candidate)
        improved = candidate.units < current.units
        event = dict(round=index, accepted=bool(valid and improved),
                     before_units=current.units, after_units=candidate.units,
                     reason='accepted' if valid and improved else ('constraint' if not valid else 'no_area_gain'),
                     details=details)
        history.append(event)
        if on_round: on_round(event)
        if not event['accepted']: break
        current = candidate
    return current, history
