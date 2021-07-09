class SamplerFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_by_name(dataset_name, dataset):
        if dataset_name == 'Mixed_EXPR':
            from .imbalanced_SLML import ImbalancedDatasetSamplerSLML
            sampler = ImbalancedDatasetSamplerSLML(dataset)
        elif dataset_name == 'Mixed_VA':
            from .imbalanced_VA import ImbalancedDatasetSamplerVA
            sampler = ImbalancedDatasetSamplerVA(dataset)
        elif dataset_name == 'EXPR_VA':
            from .imbalanced_EXPR_VA import ImbalancedDatasetSamplerEXPRVA
            sampler = ImbalancedDatasetSamplerEXPRVA(dataset)
        else:
            raise ValueError("Dataset [%s] not recognized." % dataset_name)
        return sampler
