from torchvision.datasets import MNIST
import torchvision.transforms as transforms


def DatasetSelector(args, train_transform, test_transfrom):

    if args.dataset == 'mnist':
        train_dataset = MNIST(root=args.data_path, train=True, download=False, transform=train_transform)
        test_dataset = MNIST(root=args.data_path, train=False, download=False, transform=test_transfrom)

    return train_dataset, test_dataset


def TransformSelector(input_size: str):
    normalize = transforms.Normalize((0.1307,), (0.3081,))

    if input_size == '32x32':
        transforms_train = transforms.Compose([
            transforms.Resize((36, 36)),
            transforms.RandomCrop((28, 28)),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            normalize])

        transforms_test = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            normalize])


    return transforms_train, transforms_test
