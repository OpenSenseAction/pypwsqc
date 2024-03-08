from __future__ import annotations

import numpy as np

import pypwsqc


def test_fz_filter():
    # fmt: off
    pws_data = np.array(
      [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ,
       0.   , 0.101, 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ,
       0.   , 0.   , 0.101, 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ,
       0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.101, 0.   ,
       0.   ])

    reference = np.array(
      [0.101     , 0.25136087, 0.1010425 , 0.        , 0.05012029,
       0.101     , 0.101     , 0.101     , 0.303     , 0.20048115,
       0.202     , 0.202     , 0.202     , 0.303     , 0.202     ,
       0.202     , 0.202     , 0.202     , 0.101     , 0.05012029,
       0.101     , 0.10062029, 0.0505    , 0.        , 0.        ,
       0.101     , 0.101     , 0.101     , 0.0505    , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        ])

    expected = np.array(
        [-1., -1., -1., -1., -1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0])
    # fmt: on
    result = pypwsqc.flagging.fz_filter(pws_data, reference, nint=6)
    np.testing.assert_almost_equal(expected, result)

    # the same test as above but with different `nint`
    # fmt: off
    expected = np.array(
    [-1., -1., -1., -1., -1., -1.,  -1.,  0.,  0.,  0.,  0.,  0.,  0.,
      0.,  0.,  0.,  0.,  0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,
      0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0])
    # fmt: on
    result = pypwsqc.flagging.fz_filter(pws_data, reference, nint=7)
    np.testing.assert_almost_equal(expected, result)
