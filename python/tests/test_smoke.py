"""Smoke test: verify package structure and the creator IR reducer hook are accessible."""


def test_odt_package_importable():
    from odt import providers, ir2c, resource_estimator

    assert providers is not None
    assert ir2c is not None
    assert resource_estimator is not None


def test_creator_execution_order_graph_reducer_accessible():
    from elasticai.creator.ir import ExecutionOrderGraphReducer

    assert callable(ExecutionOrderGraphReducer)
