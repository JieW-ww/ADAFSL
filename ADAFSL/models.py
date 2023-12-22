import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class Mapping(nn.Module):
    def __init__(self, in_dimension, out_dimension):
        super(Mapping, self).__init__()
        self.preconv = nn.Conv2d(in_dimension, out_dimension, 1, 1, bias=False)
        self.preconv_bn = nn.BatchNorm2d(out_dimension)

    def forward(self, x):
        x = self.preconv(x)
        x = self.preconv_bn(x)
        return x

def conv3x3x3(in_channel, out_channel):
    layer = nn.Sequential(
        nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm3d(out_channel),
        # nn.ReLU(inplace=True)
    )
    return layer


class DomainClassifier(nn.Module):
    def __init__(self):# torch.Size([1, 64, 7, 3, 3])
        super(DomainClassifier, self).__init__() #
        self.layer = nn.Sequential(
            nn.Linear(1024, 1024), #nn.Linear(320, 512), nn.Linear(FEATURE_DIM*CLASS_NUM, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),

        )
        self.domain = nn.Linear(1024, 1) # 512

    def forward(self, x, iter_num):
        coeff = calc_coeff(iter_num, 1.0, 0.0, 10,10000.0)
        x.register_hook(grl_hook(coeff))
        x = self.layer(x)
        domain_y = self.domain(x)
        return domain_y

class RandomLayer(nn.Module):
    def __init__(self, input_dim_list=[], output_dim=1024):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0/len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        # print(self.random_matrix.shape)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]
############################
def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

def MD_distance(support_feature, support_labels, query_features):
    NUM_SAMPLES=1
    class_representations, class_precision_matrices = build_class_reps_and_covariance_estimates(support_feature, support_labels)

    class_means = torch.stack(list(class_representations.values())).squeeze(1)
    class_precision_matrices = torch.stack(list(class_precision_matrices.values()))

    # grabbing the number of classes and query examples for easier use later in the function
    number_of_classes = class_means.size(0)
    number_of_targets = query_features.size(0)

    repeated_target = query_features.repeat(1, number_of_classes).view(-1, class_means.size(1))
    repeated_class_means = class_means.repeat(number_of_targets, 1)
    repeated_difference = (repeated_class_means - repeated_target)
    repeated_difference = repeated_difference.view(number_of_targets, number_of_classes,
                                                   repeated_difference.size(1)).permute(1, 0, 2)
    first_half = torch.matmul(repeated_difference, class_precision_matrices)
    sample_logits = torch.mul(first_half, repeated_difference).sum(dim=2).transpose(1, 0) * -1

    # return split_first_dim_linear(sample_logits, [NUM_SAMPLES, query_features.shape[0]])
    return sample_logits

def MD_distance_test1(support_feature, support_labels, query_features):
    NUM_SAMPLES=1
    class_representations, class_precision_matrices = build_class_reps_and_covariance_estimates(support_feature, support_labels)

    class_means = torch.stack(list(class_representations.values())).squeeze(1)
    class_precision_matrices = torch.stack(list(class_precision_matrices.values()))

    # grabbing the number of classes and query examples for easier use later in the function
    number_of_classes = class_means.size(0)
    number_of_targets = query_features.size(0)

    repeated_target = query_features.repeat(1, number_of_classes).view(-1, class_means.size(1))
    repeated_class_means = class_means.repeat(number_of_targets, 1)
    repeated_difference = (repeated_class_means - repeated_target)
    repeated_difference = repeated_difference.view(number_of_targets, number_of_classes,
                                                   repeated_difference.size(1)).permute(1, 0, 2)
    first_half = torch.matmul(repeated_difference, class_precision_matrices)
    sample_logits = torch.mul(first_half, repeated_difference).sum(dim=2).transpose(1, 0) * -1

    # return split_first_dim_linear(sample_logits, [NUM_SAMPLES, query_features.shape[0]])
    return sample_logits,class_representations, class_precision_matrices

def MD_distance_test2(query_features,class_representations, class_precision_matrices):
    # class_representations, class_precision_matrices = build_class_reps_and_covariance_estimates(support_feature, support_labels)
    #
    class_means = torch.stack(list(class_representations.values())).squeeze(1)
    # class_precision_matrices = torch.stack(list(class_precision_matrices.values()))
    #
    # # grabbing the number of classes and query examples for easier use later in the function
    number_of_classes = class_means.size(0)
    number_of_targets = query_features.size(0)

    repeated_target = query_features.repeat(1, number_of_classes).view(-1, query_features.size(1))
    repeated_class_means = class_means.repeat(number_of_targets, 1)
    repeated_difference = (repeated_class_means - repeated_target)
    repeated_difference = repeated_difference.view(number_of_targets, number_of_classes,
                                                   repeated_difference.size(1)).permute(1, 0, 2)
    first_half = torch.matmul(repeated_difference, class_precision_matrices)
    sample_logits = torch.mul(first_half, repeated_difference).sum(dim=2).transpose(1, 0) * -1

    return sample_logits

def mean_pooling(x):
    return torch.mean(x, dim=0, keepdim=True)

def build_class_reps_and_covariance_estimates(context_features, context_labels):
    class_representations={}
    class_precision_matrices={}
    task_covariance_estimate = estimate_cov(context_features)
    for c in torch.unique(context_labels):
        # filter out feature vectors which have class c
        class_mask = torch.eq(context_labels, c)
        class_mask_indices = torch.nonzero(class_mask)
        class_features = torch.index_select(context_features, 0, torch.reshape(class_mask_indices, (-1,)).cuda())
        # mean pooling examples to form class means
        class_rep = mean_pooling(class_features)
        # updating the class representations dictionary with the mean pooled representation
        class_representations[c.item()] = class_rep
        """
        Calculating the mixing ratio lambda_k_tau for regularizing the class level estimate with the task level estimate."
        Then using this ratio, to mix the two estimate; further regularizing with the identity matrix to assure invertability, and then
        inverting the resulting matrix, to obtain the regularized precision matrix. This tensor is then saved in the corresponding
        dictionary for use later in infering of the query data points.
        """
        lambda_k_tau = (class_features.size(0) / (class_features.size(0) + 1))
        class_precision_matrices[c.item()] = torch.inverse(
            (lambda_k_tau * estimate_cov(class_features)) + ((1 - lambda_k_tau) * task_covariance_estimate) \
            + torch.eye(class_features.size(1), class_features.size(1)).cuda(0))
    return class_representations,class_precision_matrices

def estimate_cov(examples, rowvar=False, inplace=False):
    if examples.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if examples.dim() < 2:
        examples = examples.view(1, -1)
    if not rowvar and examples.size(0) != 1:
        examples = examples.t()
    factor = 1.0 / (examples.size(1) - 1)
    if inplace:
        examples -= torch.mean(examples, dim=1, keepdim=True)
    else:
        examples = examples - torch.mean(examples, dim=1, keepdim=True)
    examples_t = examples.t()
    return factor * examples.matmul(examples_t).squeeze()

@staticmethod
def extract_class_indices(labels, which_class):
    class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
    class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
    return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector


class WeightedBCEWithLogitsLoss(nn.Module):

    def __init__(self, size_average=True):
        super(WeightedBCEWithLogitsLoss, self).__init__()
        self.size_average = size_average

    def weighted(self, input, target, weight, alpha, beta):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

        if weight is not None:
            loss = alpha * loss + beta * loss * weight

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

    def forward(self, input, target, weight, alpha, beta):
        if weight is not None:
            return self.weighted(input, target, weight, alpha, beta)
        else:
            return self.weighted(input, target, None, alpha, beta)


def cosine_similarity(pred1, pred2):
    pred1 = pred1.cpu().detach().numpy()
    pred2 = pred2.cpu().detach().numpy()
    sim_matrix = np.dot(pred1.reshape(-1), pred2.reshape(-1)) / \
                 (np.linalg.norm(pred1) * np.linalg.norm(pred2))
    return sim_matrix

def weightmap(pred1, pred2):
    similarity_matrix = cosine_similarity(pred1, pred2)
    normalized_similarity_matrix = (similarity_matrix + 1) / 2
    weighted_sum = pred1 * normalized_similarity_matrix + pred2 * (1 - normalized_similarity_matrix)
    return weighted_sum