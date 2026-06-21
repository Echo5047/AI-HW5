import torchvision
import torchvision.transforms as transforms

mnist = torchvision.datasets.MNIST(download=True, train=True, root="./data")
mean = mnist.data.float().mean() / 255
std = mnist.data.float().std() / 255

data_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((mean, mean, mean), (std, std, std))
])
