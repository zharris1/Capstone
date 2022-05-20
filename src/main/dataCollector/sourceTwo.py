from utils.utils import Utils

class SourceTwo(nn.Module):
    
    """
    
    SourceTwo.py: Public clothing datasets to recognize individual articles of clothing.

    -------------------------------------------------------------------------------------------

    Functions:

    """
    
    def __init__(self, seed, root):
        self.seed = seed 
        np.ramdom.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        train_data = datasets.FashionMNIST(root = root, 
        train = True, 
        download = True)
        test_data = datasets.MNIST(root=root,
                           train=False,
                           download=True,
                           transform=test_transforms)
        mean = train_data.data.float().mean() / 255
        std = train_data.data.float().std() / 255
        train_transforms = transforms.Compose([
                            transforms.RandomRotation(5, fill=(0,)),
                            transforms.RandomCrop(28, padding=2),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[mean], std=[std])
                                      ])

        test_transforms = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(mean=[mean], std=[std])
                                     ])        
        return train_data, test_data, train_transforms, test_transforms, mean, std
    
    def plot_images(self, train_data):
        images = [image for image, label in [train_data[i] for i in range(n_images)]]
        n_images = len(images)
        rows = int(np.sqrt(n_images))
        cols = int(np.sqrt(n_images))
        fig = plt.figure()
        for i in range(rows*cols):
            ax = fig.add_subplot(rows, cols, i + 1)
            ax.imshow(images[i].view(28, 28).cpu().numpy(), cmap = 'bone')
            ax.axis('off')
