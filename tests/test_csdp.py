"""
Unit tests for CSDP (Contextual Scaffolding During Pretraining) module.

Tests cover:
- classify_domain() with edge cases
- Stage progression logic (get_stage)
- Loss weight creation (create_loss_mask)
- CSDP probability calculation (get_csdp_probability)
- Configuration validation
"""

import pytest
import warnings
from nanochat.csdp import (
    classify_domain,
    get_stage,
    get_csdp_probability,
    create_loss_mask,
    CSDPConfig,
    StageBoundaries,
    DOMAIN_MODES,
    DEFAULT_STAGE_BOUNDARIES,
)


class TestClassifyDomain:
    """Tests for classify_domain() function."""

    def test_metadata_override(self):
        """Metadata domain should take precedence over heuristics."""
        code_text = "def foo(): pass\nimport os\nclass Bar:"
        # Even with clear code patterns, metadata should win
        assert classify_domain(code_text, {"domain": "academic"}) == "academic"
        assert classify_domain(code_text, {"domain": "news"}) == "news"

    def test_invalid_metadata_falls_back_to_heuristics(self):
        """Invalid metadata domain should fall back to heuristics."""
        code_text = "def foo(): pass\nimport os\nclass Bar:"
        assert classify_domain(code_text, {"domain": "invalid_domain"}) == "code"

    def test_code_detection_python(self):
        """Python code should be detected as code."""
        python_code = """
def hello_world():
    print("Hello")

class MyClass:
    pass

import os
from pathlib import Path
        """
        assert classify_domain(python_code) == "code"

    def test_code_detection_javascript(self):
        """JavaScript code should be detected as code."""
        js_code = """
function hello() {
    console.log("Hello");
}

const foo = 5;
let bar = 10;
var baz = 15;

const arrow = () => {
    return true;
};
        """
        assert classify_domain(js_code) == "code"

    def test_code_detection_java(self):
        """Java code should be detected as code."""
        java_code = """
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello");
    }

    private int getValue() {
        return 42;
    }
}
        """
        assert classify_domain(java_code) == "code"

    def test_code_detection_cpp(self):
        """C/C++ code should be detected as code."""
        cpp_code = """
#include <iostream>
#include "myheader.h"

int main() {
    std::cout << "Hello" << std::endl;
    return 0;
}
        """
        assert classify_domain(cpp_code) == "code"

    def test_prose_about_code_not_code(self):
        """Prose discussing code should NOT be classified as code.

        This tests the min_code_patterns threshold to avoid false positives.
        """
        # This text mentions programming concepts but is prose, not code
        prose = """
        The import of goods from overseas has increased significantly.
        Many people define success differently. The class of workers
        who function in tech companies has grown. JavaScript developers
        often discuss the merits of different frameworks.
        """
        # With min_code_patterns=2 (default), this should NOT be classified as code
        # because it doesn't have 2+ actual code patterns
        result = classify_domain(prose)
        assert result != "code", f"Prose about code should not be classified as code, got: {result}"

    def test_min_code_patterns_threshold(self):
        """Test that min_code_patterns threshold works correctly."""
        # Text with only one code pattern (import statement)
        one_pattern = "import os\n\nThis is just regular text about things."

        # With min_code_patterns=2, this should NOT be code
        assert classify_domain(one_pattern, min_code_patterns=2) != "code"

        # With min_code_patterns=1, this SHOULD be code
        assert classify_domain(one_pattern, min_code_patterns=1) == "code"

    def test_academic_detection(self):
        """Academic text should be detected."""
        academic_text = """
        Abstract: This paper presents a novel approach to machine learning.

        According to Smith et al., the methodology shows significant improvement.
        Our hypothesis is that neural networks can learn this task.

        References:
        [1] Smith, J. (2023). Some paper. arXiv:2301.12345
        """
        assert classify_domain(academic_text) == "academic"

    def test_news_detection(self):
        """News text should be detected."""
        news_text = """
        Breaking: Major development in tech industry announced today.

        Officials said the new policy will take effect immediately.
        According to sources close to the matter, the deal is worth billions.
        The company reported quarterly earnings above expectations.
        """
        assert classify_domain(news_text) == "news"

    def test_creative_detection(self):
        """Creative/literary text should be detected."""
        creative_text = """
        "I never thought this would happen," she whispered, her voice trembling.
        The old man shouted across the room, his words echoing off the walls.
        He felt the cold wind on his face and dreamed of warmer days.
        "Perhaps," she thought, "there is still hope."
        """
        assert classify_domain(creative_text) == "creative"

    def test_conversational_detection(self):
        """Conversational/informal text should be detected."""
        conversational_text = """
        lol that's hilarious! gonna try that later
        btw did you see the game last night? omg it was crazy
        wanna grab lunch? idk where tho
        hey how's it going?
        """
        assert classify_domain(conversational_text) == "conversational"

    def test_general_fallback(self):
        """Generic text should fall back to 'general'."""
        general_text = """
        The weather today is quite pleasant. Many people enjoy spending
        time outdoors during this season. The local park has beautiful
        trees and flowers. Children play on the swings and slides.
        """
        assert classify_domain(general_text) == "general"

    def test_empty_text(self):
        """Empty text should return 'general'."""
        assert classify_domain("") == "general"
        assert classify_domain("   ") == "general"

    def test_short_text(self):
        """Very short text should still be classified."""
        assert classify_domain("def foo():") == "general"  # Only 1 pattern, needs 2
        assert classify_domain("def foo():\n    pass\nimport os") == "code"  # 2 patterns

    def test_all_domains_in_domain_modes(self):
        """All classified domains should be valid DOMAIN_MODES keys."""
        test_texts = [
            "def foo(): pass\nimport os",  # code
            "et al. Abstract methodology",  # academic
            "reported according to officials said",  # news
            '"whispered" she said thought',  # creative
            "lol btw gonna wanna hey",  # conversational
            "The weather is nice today.",  # general
        ]
        for text in test_texts:
            domain = classify_domain(text)
            assert domain in DOMAIN_MODES, f"Domain '{domain}' not in DOMAIN_MODES"


class TestGetStage:
    """Tests for get_stage() function."""

    def test_pre_comprehension_stage(self):
        """Early training should be pre_comprehension."""
        assert get_stage(0, 1000) == "pre_comprehension"
        assert get_stage(100, 1000) == "pre_comprehension"
        assert get_stage(149, 1000) == "pre_comprehension"

    def test_early_comprehension_stage(self):
        """15-40% should be early_comprehension."""
        assert get_stage(150, 1000) == "early_comprehension"
        assert get_stage(250, 1000) == "early_comprehension"
        assert get_stage(399, 1000) == "early_comprehension"

    def test_developing_comprehension_stage(self):
        """40-75% should be developing_comprehension."""
        assert get_stage(400, 1000) == "developing_comprehension"
        assert get_stage(500, 1000) == "developing_comprehension"
        assert get_stage(749, 1000) == "developing_comprehension"

    def test_full_comprehension_stage(self):
        """75-100% should be full_comprehension."""
        assert get_stage(750, 1000) == "full_comprehension"
        assert get_stage(900, 1000) == "full_comprehension"
        assert get_stage(1000, 1000) == "full_comprehension"

    def test_zero_total_steps_warning(self):
        """total_steps=0 should warn and return pre_comprehension."""
        with pytest.warns(UserWarning, match="total_steps=0"):
            result = get_stage(0, 0)
        assert result == "pre_comprehension"

    def test_negative_total_steps_warning(self):
        """Negative total_steps should warn and return pre_comprehension."""
        with pytest.warns(UserWarning, match="total_steps=-100"):
            result = get_stage(50, -100)
        assert result == "pre_comprehension"

    def test_custom_boundaries(self):
        """Custom stage boundaries should be respected."""
        custom = StageBoundaries(
            pre_to_early=0.10,
            early_to_developing=0.30,
            developing_to_full=0.60
        )
        # With custom boundaries: 10%, 30%, 60%
        assert get_stage(50, 1000, custom) == "pre_comprehension"   # 5%
        assert get_stage(100, 1000, custom) == "early_comprehension"  # 10%
        assert get_stage(300, 1000, custom) == "developing_comprehension"  # 30%
        assert get_stage(600, 1000, custom) == "full_comprehension"  # 60%

    def test_boundary_values(self):
        """Test exact boundary values."""
        # Default boundaries: 15%, 40%, 75%
        # At exactly 15% (150/1000)
        assert get_stage(150, 1000) == "early_comprehension"
        # Just before 15%
        assert get_stage(149, 1000) == "pre_comprehension"

    def test_step_beyond_total(self):
        """Step > total_steps should still return full_comprehension."""
        assert get_stage(1500, 1000) == "full_comprehension"


class TestGetCsdpProbability:
    """Tests for get_csdp_probability() function."""

    def test_early_training_always_includes_csdp(self):
        """Before 90%, CSDP should always be included."""
        assert get_csdp_probability(0, 1000) == 1.0
        assert get_csdp_probability(500, 1000) == 1.0
        assert get_csdp_probability(899, 1000) == 1.0

    def test_annealing_phase(self):
        """Between 90-98%, probability should linearly decrease."""
        # At 90%, should be 1.0
        prob_90 = get_csdp_probability(900, 1000)
        assert prob_90 == 1.0

        # At 94% (midpoint), should be ~0.525
        prob_94 = get_csdp_probability(940, 1000)
        assert 0.4 < prob_94 < 0.6

        # At 98%, should be 0.05
        prob_98 = get_csdp_probability(980, 1000)
        assert abs(prob_98 - 0.05) < 0.01

    def test_post_annealing_minimal(self):
        """After 98%, probability should be minimal (5%)."""
        assert get_csdp_probability(990, 1000) == 0.05
        assert get_csdp_probability(1000, 1000) == 0.05

    def test_zero_total_steps(self):
        """total_steps=0 should return 1.0 (always include)."""
        assert get_csdp_probability(0, 0) == 1.0
        assert get_csdp_probability(100, 0) == 1.0


class TestCreateLossMask:
    """Tests for create_loss_mask() function."""

    def test_basic_mask_creation(self):
        """Basic mask should have CSDP weight for prefix, 1.0 for rest."""
        mask = create_loss_mask(total_tokens=100, csdp_token_count=20, loss_weight=0.1)
        assert len(mask) == 100
        assert all(m == 0.1 for m in mask[:20])
        assert all(m == 1.0 for m in mask[20:])

    def test_zero_csdp_tokens(self):
        """No CSDP tokens should give all 1.0 weights."""
        mask = create_loss_mask(total_tokens=100, csdp_token_count=0, loss_weight=0.1)
        assert len(mask) == 100
        assert all(m == 1.0 for m in mask)

    def test_all_csdp_tokens(self):
        """All CSDP tokens should give all reduced weights."""
        mask = create_loss_mask(total_tokens=100, csdp_token_count=100, loss_weight=0.5)
        assert len(mask) == 100
        assert all(m == 0.5 for m in mask)

    def test_full_weight_csdp(self):
        """loss_weight=1.0 should give all 1.0 weights."""
        mask = create_loss_mask(total_tokens=100, csdp_token_count=50, loss_weight=1.0)
        assert len(mask) == 100
        assert all(m == 1.0 for m in mask)

    def test_zero_weight_csdp(self):
        """loss_weight=0.0 should mask CSDP tokens completely."""
        mask = create_loss_mask(total_tokens=100, csdp_token_count=30, loss_weight=0.0)
        assert len(mask) == 100
        assert all(m == 0.0 for m in mask[:30])
        assert all(m == 1.0 for m in mask[30:])


class TestCSDPConfig:
    """Tests for CSDPConfig dataclass validation."""

    def test_valid_curricula(self):
        """All valid curricula should be accepted."""
        for curriculum in ["none", "aria", "sage", "nova", "heart", "bare"]:
            config = CSDPConfig(curriculum=curriculum)
            assert config.curriculum == curriculum

    def test_invalid_curriculum_raises(self):
        """Invalid curriculum should raise ValueError."""
        with pytest.raises(ValueError, match="curriculum must be one of"):
            CSDPConfig(curriculum="invalid")

    def test_valid_loss_weights(self):
        """Loss weights in [0, 1] should be accepted."""
        for weight in [0.0, 0.1, 0.5, 1.0]:
            config = CSDPConfig(loss_weight=weight)
            assert config.loss_weight == weight

    def test_invalid_loss_weight_raises(self):
        """Loss weights outside [0, 1] should raise ValueError."""
        with pytest.raises(ValueError, match="loss_weight must be in"):
            CSDPConfig(loss_weight=-0.1)
        with pytest.raises(ValueError, match="loss_weight must be in"):
            CSDPConfig(loss_weight=1.5)

    def test_create_rng_with_seed(self):
        """create_rng with seed should return deterministic RNG."""
        config = CSDPConfig(seed=42)
        rng1 = config.create_rng()
        rng2 = config.create_rng()
        # Both should produce same sequence
        assert rng1.random() == rng2.random()

    def test_create_rng_without_seed(self):
        """create_rng without seed should return None."""
        config = CSDPConfig(seed=None)
        assert config.create_rng() is None

    def test_valid_max_csdp_ratio(self):
        """Valid max_csdp_ratio values should be accepted."""
        for ratio in [0.01, 0.15, 0.5, 1.0]:
            config = CSDPConfig(max_csdp_ratio=ratio)
            assert config.max_csdp_ratio == ratio

    def test_invalid_max_csdp_ratio_zero(self):
        """max_csdp_ratio=0 should raise ValueError."""
        with pytest.raises(ValueError, match="max_csdp_ratio must be in"):
            CSDPConfig(max_csdp_ratio=0.0)

    def test_invalid_max_csdp_ratio_negative(self):
        """Negative max_csdp_ratio should raise ValueError."""
        with pytest.raises(ValueError, match="max_csdp_ratio must be in"):
            CSDPConfig(max_csdp_ratio=-0.1)

    def test_invalid_max_csdp_ratio_above_one(self):
        """max_csdp_ratio > 1 should raise ValueError."""
        with pytest.raises(ValueError, match="max_csdp_ratio must be in"):
            CSDPConfig(max_csdp_ratio=1.5)

    def test_default_max_csdp_ratio(self):
        """Default max_csdp_ratio should be 0.15."""
        config = CSDPConfig()
        assert config.max_csdp_ratio == 0.15


class TestStageBoundaries:
    """Tests for StageBoundaries dataclass validation."""

    def test_valid_boundaries(self):
        """Valid increasing boundaries should be accepted."""
        sb = StageBoundaries(
            pre_to_early=0.1,
            early_to_developing=0.4,
            developing_to_full=0.8
        )
        assert sb.pre_to_early == 0.1
        assert sb.early_to_developing == 0.4
        assert sb.developing_to_full == 0.8

    def test_non_increasing_boundaries_raises(self):
        """Non-strictly-increasing boundaries should raise ValueError."""
        with pytest.raises(ValueError, match="strictly increasing"):
            StageBoundaries(
                pre_to_early=0.5,  # > early_to_developing
                early_to_developing=0.3,
                developing_to_full=0.8
            )

    def test_equal_boundaries_raises(self):
        """Equal boundaries should raise ValueError."""
        with pytest.raises(ValueError, match="strictly increasing"):
            StageBoundaries(
                pre_to_early=0.2,
                early_to_developing=0.2,  # == pre_to_early
                developing_to_full=0.8
            )

    def test_boundary_at_zero_raises(self):
        """Boundary at 0 should raise ValueError."""
        with pytest.raises(ValueError, match="strictly increasing"):
            StageBoundaries(
                pre_to_early=0.0,  # Not in (0, 1)
                early_to_developing=0.4,
                developing_to_full=0.8
            )

    def test_boundary_at_one_raises(self):
        """Boundary at 1 should raise ValueError."""
        with pytest.raises(ValueError, match="strictly increasing"):
            StageBoundaries(
                pre_to_early=0.2,
                early_to_developing=0.5,
                developing_to_full=1.0  # Not in (0, 1)
            )

    def test_default_boundaries(self):
        """Default boundaries should be valid."""
        sb = DEFAULT_STAGE_BOUNDARIES
        assert sb.pre_to_early == 0.15
        assert sb.early_to_developing == 0.40
        assert sb.developing_to_full == 0.75
