from torch_geometric.datasets import Planetoid
import torchvision
import torchvision.transforms as transforms

CORA = Planetoid(root = 'Datasets', name = 'Cora')
PubMed = Planetoid(root = 'datasets', name = 'PubMed')
CiteSeer = Planetoid(root = 'datasets', name = 'CiteSeer')

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist = torchvision.datasets.MNIST(root='Datasets', train=True, transform=transform, download=True)

