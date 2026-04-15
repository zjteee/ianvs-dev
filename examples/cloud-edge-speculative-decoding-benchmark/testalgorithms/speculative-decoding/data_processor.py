from sedna.common.class_factory import ClassFactory, ClassType

@ClassFactory.register(ClassType.GENERAL, alias="SpeculativeDecodingDatasetProcessor")
class SpeculativeDecodingDatasetProcessor:
    def __init__(self, **kwargs):
        sample_size = kwargs.get("sample_size")
        self.sample_size = int(sample_size) if sample_size is not None else None

    def __call__(self, dataset):
        dataset_name = getattr(dataset, "dataset_name", "default")
        processed = [
            {
                "request_id": f"request-{index:03d}",
                "query": x,
                "gold": y,
                "task_name": dataset_name,
            }
            for index, (x, y) in enumerate(zip(dataset.x, dataset.y))
        ]

        if self.sample_size is not None and self.sample_size > 0:
            processed = processed[: self.sample_size]
            dataset.y = dataset.y[: self.sample_size]

        dataset.x = processed
        return dataset
