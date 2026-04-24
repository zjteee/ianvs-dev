from sedna.common.class_factory import ClassFactory, ClassType

from result_parser import parse_joint_inference_result


@ClassFactory.register(ClassType.GENERAL, alias="End-to-End Latency")
def end_to_end_latency(_, y_pred):
    infer_res = [parse_joint_inference_result(pred) for pred in y_pred]
    average_latency = (
        sum(item.result.simulation.end_to_end_latency for item in infer_res) / len(infer_res)
    )
    return round(average_latency, 3)
