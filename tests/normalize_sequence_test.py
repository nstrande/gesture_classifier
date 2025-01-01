import pytest
import numpy as np
from src.steps.preprocess_training_data import normalize_sequence

@pytest.fixture
def basic_sequence():
    return [
        # Frame 1
        [
            {"x": 100, "y": 200, "z": 0.5},
            {"x": 150, "y": 250, "z": 0.7}
        ],
        # Frame 2
        [
            {"x": 200, "y": 300, "z": 0.3},
            {"x": 250, "y": 350, "z": 0.9}
        ]
    ]

@pytest.fixture
def constant_sequence():
    return [
        [{"x": 100, "y": 100, "z": 0.5} for _ in range(2)]
        for _ in range(2)
    ]

@pytest.fixture
def negative_sequence():
    return [
        [
            {"x": -100, "y": -200, "z": -0.5},
            {"x": 100, "y": 200, "z": 0.5}
        ]
    ]

@pytest.fixture
def movement_sequence():
    return [
        [{"x": 0, "y": 0, "z": 0}, {"x": 10, "y": 10, "z": 0.1}],
        [{"x": 20, "y": 20, "z": 0.2}, {"x": 30, "y": 30, "z": 0.3}]
    ]

def test_output_range(basic_sequence):
    """Test if all normalized values are in [0,1] range"""
    normalized = normalize_sequence(basic_sequence)
    
    for frame in normalized:
        for landmark in frame:
            for coord in ['x', 'y', 'z']:
                assert 0 <= landmark[coord] <= 1

def test_constant_values(constant_sequence):
    """Test normalization of sequence with constant values"""
    normalized = normalize_sequence(constant_sequence)
    
    # All x values should be the same
    x_values = [lm["x"] for frame in normalized for lm in frame]
    y_values = [lm["y"] for frame in normalized for lm in frame]
    
    # Check if all values are equal
    assert all(x == x_values[0] for x in x_values)
    assert all(y == y_values[0] for y in y_values)

def test_negative_values(negative_sequence):
    """Test normalization of negative values"""
    normalized = normalize_sequence(negative_sequence)
    
    # Check extremes are mapped correctly
    x_values = [lm["x"] for frame in normalized for lm in frame]
    assert min(x_values) == pytest.approx(0)
    assert max(x_values) == pytest.approx(1)

def test_relative_distances(movement_sequence):
    """Test if relative distances between points are preserved"""
    normalized = normalize_sequence(movement_sequence)
    
    def get_distance_ratios(seq):
        ratios = []
        for frame in seq:
            if len(frame) > 1:
                dx = frame[1]["x"] - frame[0]["x"]
                dy = frame[1]["y"] - frame[0]["y"]
                ratios.append(np.sqrt(dx**2 + dy**2))
        return ratios
    
    original_ratios = get_distance_ratios(movement_sequence)
    normalized_ratios = get_distance_ratios(normalized)
    
    # Check if relative distances are preserved (up to scaling)
    ratio = normalized_ratios[0] / original_ratios[0]
    for orig, norm in zip(original_ratios[1:], normalized_ratios[1:]):
        assert norm / orig == pytest.approx(ratio, rel=1e-5)

def test_sequence_structure(basic_sequence):
    """Test if normalization preserves sequence structure"""
    normalized = normalize_sequence(basic_sequence)
    
    # Check if sequence structure is preserved
    assert len(normalized) == len(basic_sequence)
    for orig_frame, norm_frame in zip(basic_sequence, normalized):
        assert len(norm_frame) == len(orig_frame)
        for orig_lm, norm_lm in zip(orig_frame, norm_frame):
            assert set(norm_lm.keys()) == set(orig_lm.keys())