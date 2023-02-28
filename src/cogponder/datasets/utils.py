import pandas as pd


def remove_non_decision_time(rts: pd.Series, response_step_interval=20):
    rts[rts <= 0] = pd.NA
    ndt = rts.min() - response_step_interval
    rts = rts.apply(lambda rt: rt - ndt if rt > 0 else rt)
    return rts
