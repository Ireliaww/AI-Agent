import math

def get_transformer_learning_rate(step: int, d_model: int, warmup_steps: int) -> float:
    """
    Calculates the learning rate according to the Transformer's custom schedule.

    The learning rate schedule is defined as:
    lr = d_model^{-0.5} * min(step^{-0.5}, step * warmup_steps^{-1.5})

    It increases linearly for the first `warmup_steps` and then decreases
    proportionally to the inverse square root of the step number. This schedule
    helps the model stabilize early in training and then fine-tune later.

    Args:
        step: The current training step (1-indexed). Must be a positive integer.
        d_model: The dimension of the model (e.g., 512 for the base model).
                 Must be a positive integer.
        warmup_steps: The number of warmup steps (e.g., 4000).
                      Must be a positive integer.

    Returns:
        The calculated learning rate for the given step.

    Raises:
        ValueError: If step, d_model, or warmup_steps are not positive integers.
    """
    if not isinstance(step, int) or step <= 0:
        raise ValueError("Step must be a positive integer.")
    if not isinstance(d_model, int) or d_model <= 0:
        raise ValueError("d_model must be a positive integer.")
    if not isinstance(warmup_steps, int) or warmup_steps <= 0:
        raise ValueError("warmup_steps must be a positive integer.")

    # Calculate the two terms inside the min function
    # term1: step^{-0.5} - responsible for the inverse square root decay
    term1 = step ** -0.5
    # term2: step * warmup_steps^{-1.5} - responsible for the linear increase during warmup
    term2 = step * (warmup_steps ** -1.5)
    
    # The learning rate is scaled by d_model^{-0.5} and takes the minimum of the two terms
    lr = (d_model ** -0.5) * min(term1, term2)
    
    return lr

if __name__ == "__main__":
    # Define a small tolerance for floating point comparisons due to precision
    TOLERANCE = 1e-7

    # --- Test Cases for get_transformer_learning_rate ---

    # Standard hyperparameters from the original paper's base model
    D_MODEL_BASE = 512
    WARMUP_STEPS_BASE = 4000

    print("Running test cases for get_transformer_learning_rate function...")

    # Basic cases:

    # Test Case 1: Step 1 (very beginning of warmup phase)
    # Expected behavior: Learning rate should be very small, proportional to 1 * warmup_steps^-1.5
    lr_step1 = get_transformer_learning_rate(1, D_MODEL_BASE, WARMUP_STEPS_BASE)
    expected_lr_step1 = (D_MODEL_BASE ** -0.5) * (1 * (WARMUP_STEPS_BASE ** -1.5))
    assert math.isclose(lr_step1, expected_lr_step1, rel_tol=TOLERANCE), \
        f"Test Case 1 Failed: Step 1. Expected {expected_lr_step1:.8e}, Got {lr_step1:.8e}"

    # Test Case 2: Step halfway through warmup (linear increase phase)
    step_half_warmup = WARMUP_STEPS_BASE // 2 # 2000
    lr_half_warmup = get_transformer_learning_rate(step_half_warmup, D_MODEL_BASE, WARMUP_STEPS_BASE)
    expected_lr_half_warmup = (D_MODEL_BASE ** -0.5) * (step_half_warmup * (WARMUP_STEPS_BASE ** -1.5))
    assert math.isclose(lr_half_warmup, expected_lr_half_warmup, rel_tol=TOLERANCE), \
        f"Test Case 2 Failed: Half warmup. Expected {expected_lr_half_warmup:.8e}, Got {lr_half_warmup:.8e}"

    # Test Case 3: Step exactly at warmup_steps (peak learning rate)
    # At this point, step^-0.5 and step * warmup_steps^-1.5 should be equal.
    lr_warmup_end = get_transformer_learning_rate(WARMUP_STEPS_BASE, D_MODEL_BASE, WARMUP_STEPS_BASE)
    expected_lr_warmup_end = (D_MODEL_BASE ** -0.5) * (WARMUP_STEPS_BASE ** -0.5)
    assert math.isclose(lr_warmup_end, expected_lr_warmup_end, rel_tol=TOLERANCE), \
        f"Test Case 3 Failed: Warmup end. Expected {expected_lr_warmup_end:.8e}, Got {lr_warmup_end:.8e}"

    # Test Case 4: Step after warmup (decay phase)
    step_after_warmup = WARMUP_STEPS_BASE * 2 # 8000
    lr_after_warmup = get_transformer_learning_rate(step_after_warmup, D_MODEL_BASE, WARMUP_STEPS_BASE)
    expected_lr_after_warmup = (D_MODEL_BASE ** -0.5) * (step_after_warmup ** -0.5)
    assert math.isclose(lr_after_warmup, expected_lr_after_warmup, rel_tol=TOLERANCE), \
        f"Test Case 4 Failed: After warmup. Expected {expected_lr_after_warmup:.8e}, Got {lr_after_warmup:.8e}"

    # Test Case 5: Much later step (continued decay)
    step_very_late = WARMUP_STEPS_BASE * 10 # 40000
    lr_very_late = get_transformer_learning_rate(step_very_late, D_MODEL_BASE, WARMUP_STEPS_BASE)
    expected_lr_very_late = (D_MODEL_BASE ** -0.5) * (step_very_late ** -0.5)
    assert math.isclose(lr_very_late, expected_lr_very_late, rel_tol=TOLERANCE), \
        f"Test Case 5 Failed: Very late step. Expected {expected_lr_very_late:.8e}, Got {lr_very_late:.8e}"

    # Edge cases for d_model and warmup_steps:

    # Test Case 6: Different d_model (smaller)
    d_model_small = 128
    lr_small_d_model = get_transformer_learning_rate(WARMUP_STEPS_BASE, d_model_small, WARMUP_STEPS_BASE)
    expected_lr_small_d_model = (d_model_small ** -0.5) * (WARMUP_STEPS_BASE ** -0.5)
    assert math.isclose(lr_small_d_model, expected_lr_small_d_model, rel_tol=TOLERANCE), \
        f"Test Case 6 Failed: Small d_model. Expected {expected_lr_small_d_model:.8e}, Got {lr_small_d_model:.8e}"

    # Test Case 7: Different warmup_steps (shorter)
    warmup_steps_short = 1000
    # At step 500 (during warmup)
    lr_short_warmup_during = get_transformer_learning_rate(500, D_MODEL_BASE, warmup_steps_short)
    expected_lr_short_warmup_during = (D_MODEL_BASE ** -0.5) * (500 * (warmup_steps_short ** -1.5))
    assert math.isclose(lr_short_warmup_during, expected_lr_short_warmup_during, rel_tol=TOLERANCE), \
        f"Test Case 7 Failed: Short warmup (during). Expected {expected_lr_short_warmup_during:.8e}, Got {lr_short_warmup_during:.8e}"
    # At step 2000 (after warmup)
    lr_short_warmup_after = get_transformer_learning_rate(2000, D_MODEL_BASE, warmup_steps_short)
    expected_lr_short_warmup_after = (D_MODEL_BASE ** -0.5) * (2000 ** -0.5)
    assert math.isclose(lr_short_warmup_after, expected_lr_short_warmup_after, rel_tol=TOLERANCE), \
        f"Test Case 7 Failed: Short warmup (after). Expected {expected_lr_short_warmup_after:.8e}, Got {lr_short_warmup_after:.8e}"

    # Test Case 8: Minimum valid inputs
    lr_min_inputs = get_transformer_learning_rate(1, 1, 1)
    expected_lr_min_inputs = (1 ** -0.5) * min(1 ** -0.5, 1 * (1 ** -1.5)) # 1 * min(1, 1) = 1
    assert math.isclose(lr_min_inputs, expected_lr_min_inputs, rel_tol=TOLERANCE), \
        f"Test Case 8 Failed: Min inputs. Expected {expected_lr_min_inputs:.8e}, Got {lr_min_inputs:.8e}"

    # Test Case 9: Large step value (ensure no overflow/underflow issues for reasonable inputs)
    large_step = 1_000_000
    lr_large_step = get_transformer_learning_rate(large_step, D_MODEL_BASE, WARMUP_STEPS_BASE)
    expected_lr_large_step = (D_MODEL_BASE ** -0.5) * (large_step ** -0.5)
    assert math.isclose(lr_large_step, expected_lr_large_step, rel_tol=TOLERANCE), \
        f"Test Case 9 Failed: Large step. Expected {expected_lr_large_step:.8e}, Got {lr_large_step:.8e}"

    # Test Case 10: Check behavior at the transition point (warmup_steps)
    # The two terms should be equal at step = warmup_steps
    # term1 = warmup_steps ** -0.5
    # term2 = warmup_steps * (warmup_steps ** -1.5) = warmup_steps ** (1 - 1.5) = warmup_steps ** -0.5
    # So min(term1, term2) should be term1 (or term2)
    step_transition = WARMUP_STEPS_BASE
    lr_transition = get_transformer_learning_rate(step_transition, D_MODEL_BASE, WARMUP_STEPS_BASE)
    expected_lr_transition = (D_MODEL_BASE ** -0.5) * (step_transition ** -0.5)
    assert math.isclose(lr_transition, expected_lr_transition, rel_tol=TOLERANCE), \
        f"Test Case 10 Failed: Transition point. Expected {expected_lr_transition:.8e}, Got {lr_transition:.8e}"

    # Test Case 11: Verify ValueError for non-positive step
    try:
        get_transformer_learning_rate(0, D_MODEL_BASE, WARMUP_STEPS_BASE)
        assert False, "Test Case 11 Failed: Expected ValueError for step=0"
    except ValueError as e:
        assert str(e) == "Step must be a positive integer.", f"Test Case 11 Failed: Wrong error message for step=0: {e}"

    try:
        get_transformer_learning_rate(-1, D_MODEL_BASE, WARMUP_STEPS_BASE)
        assert False, "Test Case 11 Failed: Expected ValueError for step=-1"
    except ValueError as e:
        assert str(e) == "Step must be a positive integer.", f"Test Case 11 Failed: Wrong error message for step=-1: {e}"

    # Test Case 12: Verify ValueError for non-positive d_model
    try:
        get_transformer_learning_rate(1, 0, WARMUP_STEPS_BASE)
        assert False, "Test Case 12 Failed: Expected ValueError for d_model=0"
    except ValueError as e:
        assert str(e) == "d_model must be a positive integer.", f"Test Case 12 Failed: Wrong error message for d_model=0: {e}"

    # Test Case 13: Verify ValueError for non-positive warmup_steps
    try:
        get_transformer_learning_rate(1, D_MODEL_BASE, 0)
        assert False, "Test Case 13 Failed: Expected ValueError for warmup_steps=0"
    except ValueError as e:
        assert str(e) == "warmup_steps must be a positive integer.", f"Test Case 13 Failed: Wrong error message for warmup_steps=0: {e}"

    # Test Case 14: Verify ValueError for non-integer inputs
    try:
        get_transformer_learning_rate(1.5, D_MODEL_BASE, WARMUP_STEPS_BASE)
        assert False, "Test Case 14 Failed: Expected ValueError for float step"
    except ValueError as e:
        assert str(e) == "Step must be a positive integer.", f"Test Case 14 Failed: Wrong error message for float step: {e}"

    try:
        get_transformer_learning_rate(1, 512.0, WARMUP_STEPS_BASE)
        assert False, "Test Case 14 Failed: Expected ValueError for float d_model"
    except ValueError as e:
        assert str(e) == "d_model must be a positive integer.", f"Test Case 14 Failed: Wrong error message for float d_model: {e}"

    print("All test cases passed!")

    # --- Usage Examples ---
    print("\n--- Usage Examples ---")
    d_model_example = 512
    warmup_steps_example = 4000

    # Example 1: Early in training (warmup phase)
    step_early = 1000
    lr_early = get_transformer_learning_rate(step_early, d_model_example, warmup_steps_example)
    print(f"Learning rate at step {step_early} (d_model={d_model_example}, warmup_steps={warmup_steps_example}): {lr_early:.8f}")

    # Example 2: At the peak of the learning rate schedule
    step_peak = warmup_steps_example
    lr_peak = get_transformer_learning_rate(step_peak, d_model_example, warmup_steps_example)
    print(f"Learning rate at step {step_peak} (peak): {lr_peak:.8f}")

    # Example 3: After the warmup phase (decaying)
    step_late = 10000
    lr_late = get_transformer_learning_rate(step_late, d_model_example, warmup_steps_example)
    print(f"Learning rate at step {step_late} (decaying): {lr_late:.8f}")

    # Example 4: With different model dimensions
    d_model_large = 1024
    lr_large_d_model = get_transformer_learning_rate(step_peak, d_model_large, warmup_steps_example)
    print(f"Learning rate at step {step_peak} (d_model={d_model_large}, warmup_steps={warmup_steps_example}): {lr_large_d_model:.8f}")

    # Example 5: With different warmup steps
    warmup_steps_short_example = 2000
    lr_short_warmup = get_transformer_learning_rate(1000, d_model_example, warmup_steps_short_example)
    print(f"Learning rate at step {1000} (d_model={d_model_example}, warmup_steps={warmup_steps_short_example}): {lr_short_warmup:.8f}")