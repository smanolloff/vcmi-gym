import torch
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower


class Model(torch.nn.Module):
    def forward(self, x):
        # ERROR:
        # return x.mean(0)
        # return x.sum(0)
        # return x.logsumexp(0)
        return x.argmax(dim=-1)
        # OK:
        # return x.mean(dim=1, keepdim=True).squeeze(1)
        # return x.logsumexp(dim=0, keepdim=True).squeeze(0)


model = Model()
inputs = (torch.randn(2, 3, 4),)
ep = torch.export.export(model.eval(), inputs)
lowered = to_edge_transform_and_lower(ep, partitioner=[VulkanPartitioner()])
