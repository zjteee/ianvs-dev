from sedna.common.class_factory import ClassFactory, ClassType

from result_parser import parse_joint_inference_result


@ClassFactory.register(ClassType.GENERAL, alias="Acceptance Rate")
def acceptance_rate(_, y_pred):
    infer_res = [parse_joint_inference_result(pred) for pred in y_pred]
    acceptance_values = [
        item.result.simulation.acceptance_rate
        for item in infer_res
        if item.result.simulation.acceptance_rate is not None
    ]
    if not acceptance_values:
        return ""
    average_acceptance = sum(acceptance_values) / len(acceptance_values)
    return round(average_acceptance, 3)
