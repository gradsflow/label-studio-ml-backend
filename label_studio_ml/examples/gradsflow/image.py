import hashlib
import io
import os
import urllib

import numpy as np
import requests
import torch
from PIL import Image
from gradsflow import AutoDataset, AutoImageClassifier
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_single_tag_keys, get_choice, is_skipped


image_size = 224
image_transforms = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image_cache_dir = os.path.join(os.path.dirname(__file__), 'image-cache')
os.makedirs(image_cache_dir, exist_ok=True)


def get_transformed_image(url):
    is_local_file = url.startswith('/data')
    if is_local_file:
        filename, dir_path = url.split('/data/')[1].split('?d=')
        dir_path = str(urllib.parse.unquote(dir_path))
        filepath = os.path.join(dir_path, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(filepath)
        with open(filepath, mode='rb') as f:
            image = Image.open(f).convert('RGB')
    else:
        cached_file = os.path.join(image_cache_dir, hashlib.md5(url.encode()).hexdigest())
        if os.path.exists(cached_file):
            with open(cached_file, mode='rb') as f:
                image = Image.open(f).convert('RGB')
        else:
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with io.BytesIO(r.content) as f:
                image = Image.open(f).convert('RGB')
            with io.open(cached_file, mode='wb') as fout:
                fout.write(r.content)
    return image_transforms(image)


class ImageClassifierDataset(Dataset):

    def __init__(self, image_urls, image_classes):
        self.classes = list(set(image_classes))
        self.class_to_label = {c: i for i, c in enumerate(self.classes)}

        self.images, self.labels = [], []
        for image_url, image_class in zip(image_urls, image_classes):
            try:
                image = get_transformed_image(image_url)
            except Exception as exc:
                print(exc)
                continue
            self.images.append(image)
            self.labels.append(self.class_to_label[image_class])

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.images)


class ImageClassifier(object):

    def __init__(self, num_classes):
        # self.model = models.resnet18(pretrained=True)
        self.automodel = None
        self.num_classes = num_classes
        self.model = None

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def predict(self, image_urls):
        images = torch.stack([get_transformed_image(url) for url in image_urls])
        with torch.no_grad():
            return self.model(images).data.numpy()

    def train(self, autodataset: AutoDataset, num_epochs=5):
        self.automodel = AutoImageClassifier(train_dataloader=autodataset.train_dataloader,
            num_classes=self.num_classes,
            max_epochs=num_epochs,
        )
        analysis = self.automodel.hp_tune(gpu=torch.cuda.device_count())
        self.model = self.automodel.model

        return self.model, analysis


class ImageClassifierAPI(LabelStudioMLBase):

    def __init__(self, **kwargs):
        super(ImageClassifierAPI, self).__init__(**kwargs)
        self.from_name, self.to_name, self.value, self.classes = get_single_tag_keys(
            self.parsed_label_config, 'Choices', 'Image')
        if self.train_output:
            self.classes = self.train_output['classes']
            self.model = ImageClassifier(len(self.classes), )
            self.model.load(self.train_output['model_path'])
        else:
            self.model = ImageClassifier(len(self.classes), )

    def reset_model(self):
        self.model = ImageClassifier(len(self.classes))

    def predict(self, tasks, **kwargs):
        image_urls = [task['data'][self.value] for task in tasks]
        logits = self.model.predict(image_urls)
        predicted_label_indices = np.argmax(logits, axis=1)
        predicted_scores = logits[np.arange(len(predicted_label_indices)), predicted_label_indices]
        predictions = []
        for idx, score in zip(predicted_label_indices, predicted_scores):
            predicted_label = self.classes[idx]
            # prediction result for the single task
            result = [{
                'from_name': self.from_name,
                'to_name': self.to_name,
                'type': 'choices',
                'value': {'choices': [predicted_label]}
            }]

            # expand predictions with their scores for all tasks
            predictions.append({'result': result, 'score': float(score)})

        return predictions

    def fit(self, completions, workdir=None, batch_size=32, num_epochs=10, **kwargs):
        image_urls, image_classes = [], []
        print('Collecting annotations...')
        for completion in completions:
            if is_skipped(completion):
                continue
            image_urls.append(completion['data'][self.value])
            image_classes.append(get_choice(completion))

        print('Creating dataset...')
        dataset = ImageClassifierDataset(image_urls, image_classes)
        dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
        auto_dataset = AutoDataset(train_dataloader=dataloader, train_dataset=dataset)

        print('Train model...')
        self.reset_model()
        model, analysis = self.model.train(auto_dataset, num_epochs=num_epochs)

        print("Best config...")
        print(analysis.best_config)

        print('Save model...')
        model_path = os.path.join(workdir, 'model.pt')
        self.model.save(model_path)

        return {'model_path': model_path, 'classes': dataset.classes, "best_config": analysis.best_config}
