from sedna.common.class_factory import ClassFactory, ClassType

from result_parser import parse_joint_inference_result


@ClassFactory.register(ClassType.GENERAL, alias="Throughput")
def throughput(_, y_pred):
    infer_res = [parse_joint_inference_result(pred) for pred in y_pred]
    average_throughput = sum(item.result.throughput for item in infer_res) / len(infer_res)
    return round(average_throughput, 2)
