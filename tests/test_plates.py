"""FastPlateOcrReader: masked-character preservation + completeness-derived
confidence. SimulatedPlateReader is exercised via reasoning/cascade tests."""
from unittest.mock import MagicMock, patch

import numpy as np

from perception.plates import FastPlateOcrReader
from perception.types import SOURCE_MODEL


def _reader_with_mocked_engine(run_return):
    reader = FastPlateOcrReader()
    engine = MagicMock()
    engine.run.return_value = run_return
    reader._engine = engine
    return reader


def test_partial_read_keeps_mask_characters_not_stripped():
    reader = _reader_with_mocked_engine(["AB__1234"])
    result = reader.read(np.zeros((20, 60, 3), dtype=np.uint8))
    assert result is not None
    assert result.text == "AB__1234"
    assert result.source == SOURCE_MODEL


def test_edge_mask_characters_do_not_shift_positions():
    """The old .strip("_") behavior silently shifted the string when the
    unread characters were at an edge -- exactly the case that corrupts
    position-by-position comparison downstream."""
    reader = _reader_with_mocked_engine(["__AB1234"])
    result = reader.read(np.zeros((20, 60, 3), dtype=np.uint8))
    assert result.text == "__AB1234"
    assert len(result.text) == 8


def test_confidence_reflects_completeness():
    full = _reader_with_mocked_engine(["ABC1234"])
    half = _reader_with_mocked_engine(["AB__234"])
    mostly_masked = _reader_with_mocked_engine(["A______"])
    crop = np.zeros((20, 60, 3), dtype=np.uint8)
    conf_full = full.read(crop).confidence
    conf_half = half.read(crop).confidence
    conf_low = mostly_masked.read(crop).confidence
    assert conf_full > conf_half > conf_low


def test_fully_masked_read_returns_none():
    reader = _reader_with_mocked_engine(["_______"])
    assert reader.read(np.zeros((20, 60, 3), dtype=np.uint8)) is None


def test_no_read_returns_none():
    reader = _reader_with_mocked_engine([])
    assert reader.read(np.zeros((20, 60, 3), dtype=np.uint8)) is None
