import os
import pickle
from typing import List
from lightning.pytorch import LightningDataModule
from torchvision import transforms, datasets
from torch.utils.data import Subset, DataLoader, ConcatDataset, default_collate
import numpy as np


def get_custom_collate_fn(dataset_name):
    def custom_collate_fn(batch):
        # Now, dataset_name is accessible here as a closure variable
        batch = default_collate(batch)  # This handles the basic collation
        
        # Assuming your batch is a list of tuples (data, label)
        batch_dict = {
            'data': batch[0],
            'labels': batch[1],
            'dataset_name': dataset_name
        }
        
        return batch_dict

    return custom_collate_fn


class ModularDataModule(LightningDataModule):
    """
    A Custom DataModule that can load in multiple datasets at once split into subsets by labels
    """
    def __init__(self, data_dir: str = "./datasets",
                 batch_size: int = 32,
                 agent_config: List = None,
                 cache_dir: str = "./cache",
                 validation_split: float = 0.1,
                 custom_transforms: dict = None,
                 num_workers: int = 10):
        
        super().__init__()
        self.data_dir = data_dir
        self.agent_config = agent_config

        self.dataset_names = []
        for i, agent in enumerate(agent_config):
            num_agent_datasets = len(agent['data'])
            for i in range(num_agent_datasets):
                if agent['data'][i]['dataset'] not in self.dataset_names:
                    self.dataset_names.append(agent['data'][i]['dataset'])


        self.cache_dir = cache_dir

        self.transforms = custom_transforms or {
            'MNIST': transforms.Compose([
                transforms.ToTensor(),  # Converts images to PyTorch tensors with values in [0, 1]
                transforms.Normalize((0.5,), (0.5,))  # Normalizes tensors to have mean 0.5 and std 0.5
            ]),
            'FashionMNIST': transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            # Add default transforms for other datasets
        }

        self.cache = self.load_cache()
        self.validation_split = validation_split
        self.save_hyperparameters()

    def load_cache(self):
        cache_path = os.path.join(self.cache_dir, 'data_module_cache.pkl')
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        else:
            return {}
        
    def load_dataset(self, dataset_name, train, transform):
        if dataset_name == 'MNIST':
            return datasets.MNIST(self.data_dir, train=train, download=True, transform=transform)
        elif dataset_name == 'FashionMNIST':
            return datasets.FashionMNIST(self.data_dir, train=train, download=True, transform=transform)
        # Add Datasets Here
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

    def save_cache(self):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        cache_path = os.path.join(self.cache_dir, 'data_module_cache.pkl')
        with open(cache_path, 'wb') as f:
            pickle.dump(self.cache, f)

    def prepare_data(self):
        # Download datasets if not already present
        for dataset_name in self.dataset_names:
            self.load_dataset(dataset_name, train=True, transform=self.transforms[dataset_name])

    def setup(self, stage=None):
        self.agent_datasets = {agent["id"]: dict() for agent in self.agent_config}
        train_datasets = {dataset_name: self.load_dataset(dataset_name,
                                                         train=True,
                                                         transform=self.transforms[dataset_name]) for dataset_name in self.dataset_names}
        test_datasets = {dataset_name: self.load_dataset(dataset_name,
                                                         train=False,
                                                         transform=self.transforms[dataset_name]) for dataset_name in self.dataset_names}

        for dataset_name in self.dataset_names:
            config_key = f"{dataset_name}"
            if config_key not in self.cache:
                print(f"Creating labels cache for {dataset_name}...")
                train_dataset = train_datasets[dataset_name]
                test_dataset = test_datasets[dataset_name]
                train_indices_dict, val_indices_dict, test_indices_dict = dict(), dict(), dict()

                # Each dataset must have a targets property that allows you to access all the targets in the dataset
                unique_labels = train_dataset.targets.unique()
                for label in unique_labels:
                    label = label.item()
                    train_indices_dict[label] = list(np.where(train_dataset.targets == label)[0])
                    test_indices_dict[label] = list(np.where(test_dataset.targets == label)[0])

                    # Shuffle indices
                    np.random.shuffle(train_indices_dict[label])
                    split = int(np.floor(self.validation_split * len(train_indices_dict[label])))
                    train_indices_dict[label] = train_indices_dict[label][split:]
                    val_indices_dict[label] = train_indices_dict[label][:split]

                self.cache[config_key] = {"train": train_indices_dict, "val": val_indices_dict, "test": test_indices_dict}

                # Update the cache file with new indices
                self.save_cache()

        # Loop through the different agents
        for agent in self.agent_config:
            num_agent_datasets = len(agent['data'])
            agent_id = agent['id']
            agent_train_dataset, agent_val_dataset, agent_test_dataset = [], [], []

            # Each agent can mix and match datasets
            for i in range(num_agent_datasets):
                dataset_name = agent['data'][i]['dataset']
                train_dataset = train_datasets[dataset_name]
                test_dataset = train_datasets[dataset_name]
                for label in agent['data'][i]['labels']:
                    agent_train_dataset.append(Subset(train_dataset, self.cache[dataset_name]["train"][label]))
                    agent_val_dataset.append(Subset(train_dataset, self.cache[dataset_name]["val"][label]))
                    agent_test_dataset.append(Subset(test_dataset, self.cache[dataset_name]["test"][label]))

            self.agent_datasets[agent_id]["train"] = ConcatDataset(agent_train_dataset)
            self.agent_datasets[agent_id]["val"] = ConcatDataset(agent_val_dataset)
            self.agent_datasets[agent_id]["test"] = ConcatDataset(agent_test_dataset)

    def train_dataloader(self):
        # Combine MNIST and Fashion MNIST training subsets
        dataloaders = [DataLoader(self.agent_datasets[agent["id"]]["train"],
                                  batch_size=self.hparams.batch_size,
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=self.hparams.num_workers) for agent in self.agent_config]
        return dataloaders

    def val_dataloader(self):
        # Return a list of validation DataLoaders for MNIST and Fashion MNIST
        full_dataset = ConcatDataset([self.agent_datasets[agent["id"]]["val"] for agent in self.agent_config])
        dataloader = DataLoader(full_dataset,
                                batch_size=self.hparams.batch_size,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=self.hparams.num_workers)
        return dataloader
    
    def test_dataloader(self):
        # Return a list of validation DataLoaders for MNIST and Fashion MNIST
        full_dataset = ConcatDataset([self.agent_datasets[agent["id"]]["test"] for agent in self.agent_config])
        dataloader = DataLoader(full_dataset,
                                batch_size=self.hparams.batch_size,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=self.hparams.num_workers)
        return dataloader



