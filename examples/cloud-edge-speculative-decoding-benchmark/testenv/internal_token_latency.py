from sedna.common.class_factory import ClassFactory, ClassType

from result_parser import parse_joint_inference_result


@ClassFactory.register(ClassType.GENERAL, alias="Internal Token Latency")
def internal_token_latency(_, y_pred):
    infer_res = [parse_joint_inference_result(pred) for pred in y_pred]
    average_itl = sum(item.result.internal_token_latency for item in infer_res) / len(infer_res)
    return round(average_itl, 4)
