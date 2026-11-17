import numpy as np
from scripts.calibrate import select_threshold


def test_threshold_cut_ties_and_singletons():
    scores=np.array([.9,.8,.8,.7,.6,.5],dtype=np.float64)
    threshold,groups=select_threshold(scores,8,4)
    assert groups==8-int((scores>=threshold).sum())==4
    threshold,groups=select_threshold(scores,8,5.5)
    assert abs(groups-5.5)==.5
    assert select_threshold(np.array([]),4,2)==(1.,4)
