"""The planner accepts only what the three C trainers exercise, and refuses the rest by name."""
import dataclasses
import re
from pathlib import Path

import pytest

from odt.planner import Dtype, UnsupportedProblem, load_problem, require_supported

GOLDENS = Path(__file__).resolve().parent / "planner_goldens"
HAR = load_problem(GOLDENS / "har.problem.json")
TYPES = "CONV1D, RELU, MAXPOOL1D, AVGPOOL1D, FLATTEN, LINEAR, SOFTMAX"


def refused(problem, m, message):
    with pytest.raises(UnsupportedProblem, match=f"^{re.escape(message)}$"):
        require_supported(problem, m)


@pytest.mark.parametrize("model", ("har", "har_finetune", "mnist_cnn"))
def test_the_three_c_trainers_are_accepted_at_a_micro_batch_of_1(model):
    assert require_supported(load_problem(GOLDENS / f"{model}.problem.json"), 1) is None


@pytest.mark.parametrize("m", (0, 2, 8, 1.0, True))
def test_a_micro_batch_other_than_1_is_refused(m):
    refused(HAR, m, f"m: {m} has no C reference run; the planner accepts 1")


def test_mse_is_refused():
    refused(dataclasses.replace(HAR, loss="MSE"), 1,
            "loss: MSE has no C reference run; the planner accepts CROSS_ENTROPY")


def test_a_non_float32_input_is_refused():
    p = dataclasses.replace(HAR, input=dataclasses.replace(HAR.input, dtype=Dtype("SYM_INT32")))
    refused(p, 1, "input.dtype: SYM_INT32 has no C reference run; the planner accepts FLOAT32")


def test_an_input_of_rank_above_8_is_refused_and_rank_8_is_accepted():
    def of_rank(rank):
        return dataclasses.replace(HAR, input=dataclasses.replace(HAR.input, shape=(1,) * rank))

    assert require_supported(of_rank(8), 1) is None
    refused(of_rank(9), 1, "input.shape: rank 9 has no C reference run; the planner accepts "
                           "ranks 1 to 8 (the example harness's REMAT_DUMP_MAX_RANK)")


@pytest.mark.parametrize("field, attr", (("outputDtype", "output_dtype"),
                                         ("propLossDtype", "prop_loss_dtype")))
def test_a_non_float32_layer_dtype_is_refused_naming_the_layer(field, attr):
    layers = list(HAR.layers)
    layers[4] = dataclasses.replace(layers[4], **{attr: Dtype("BFP", (("exponentBits", 8),
                                                                      ("groupSize", 0),
                                                                      ("mantissaBits", 8)))})
    refused(dataclasses.replace(HAR, layers=tuple(layers)), 1,
            f"layers[4].{field}: BFP has no C reference run; the planner accepts FLOAT32")


@pytest.mark.parametrize("kind", ("BATCHNORM1D", "LAYERNORM", "GROUPNORM", "DROPOUT",
                                  "CONV1D_TRANSPOSED", "ADAPTIVE_AVGPOOL1D", "QUANTIZATION"))
def test_a_layer_type_without_a_c_reference_run_is_refused_naming_the_layer(kind):
    layers = list(HAR.layers)
    layers[1] = dataclasses.replace(layers[1], type=kind)
    refused(dataclasses.replace(HAR, layers=tuple(layers)), 1,
            f"layers[1].type: {kind} has no C reference run; the planner accepts {TYPES}")


def test_a_problem_of_more_than_64_layers_is_refused_before_its_layers_are_checked():
    relu = HAR.layers[1]
    assert require_supported(dataclasses.replace(HAR, layers=(relu,) * 64), 1) is None
    dropout_first = (dataclasses.replace(relu, type="DROPOUT"),) + (relu,) * 64
    refused(dataclasses.replace(HAR, layers=dropout_first), 1,
            "layers: 65 has no C reference run; the planner accepts at most 64 layers (the "
            "example harness's REMAT_SELECT_MAX_LAYERS)")


def test_a_problem_breaking_several_rules_names_the_first_in_check_order():
    p = dataclasses.replace(
        HAR, loss="MSE", layers=HAR.layers[:1] * 65,
        input=dataclasses.replace(HAR.input, dtype=Dtype("SYM_INT32")))
    refused(p, 2, "m: 2 has no C reference run; the planner accepts 1")
    refused(p, 1, "loss: MSE has no C reference run; the planner accepts CROSS_ENTROPY")
    p = dataclasses.replace(p, loss="CROSS_ENTROPY")
    refused(p, 1, "input.dtype: SYM_INT32 has no C reference run; the planner accepts FLOAT32")
