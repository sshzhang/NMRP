class AttentionNetworkConfiguration(object):
    def __init__(self, vocabulary_size, m, document_size, sequence_length, unit_type, s, dim, K, dropout):

        self.vocabulary_size = vocabulary_size
        self.m = m
        self.document_size = document_size
        self.sequence_length = sequence_length
        self.unit_type = unit_type
        self.s = s
        self.dim = dim
        self.K = K
        self.dropout = dropout

    def print_configurations(self):
        configuration_dict = self.__dict__
        configuration_list = ["{}: {}".format(configuration, configuration_dict[configuration]) for configuration in configuration_dict]
        print("Attention Network Configuration: {}.".format(", ".join(configuration_list)))
