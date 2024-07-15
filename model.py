import os
import json
import torch
from torchvision import models, transforms
import torch.nn.functional as F
from skimage.segmentation import mark_boundaries
from PIL import Image
from logging import warning, info

from lime import lime_image

class ModelInception():
    '''Inception model wrapper for inference.
    '''
    def __init__(self):
        info('Initializing Inception_v3 model...')
        self.labels_path = './data/imagenet_class_index.json'
        self.idx2label, self.cls2label, self.cls2idx, self.label2idx = self._load_labels(self.labels_path)
        self.model = self._load_model()

    def _load_model(self):
        model = models.inception_v3(pretrained=True)
        model.eval()
        return model 

    def _load_labels(self, path): 
        idx2label, cls2label, cls2idx, label2idx = [], {}, {}, {}
        with open(os.path.abspath(path), 'r') as read_file:
            class_idx = json.load(read_file)
            idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
            cls2label = {class_idx[str(k)][0]: class_idx[str(k)][1] for k in range(len(class_idx))}
            cls2idx = {class_idx[str(k)][0]: k for k in range(len(class_idx))} 
            label2idx = {class_idx[str(k)][1]:k for k in range(len(class_idx))}
        return idx2label, cls2label, cls2idx, label2idx
    
    def transform_input_data(self, data):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])       
        transf = transforms.Compose([
            transforms.Resize((256, 256)),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])    

        # unsqeeze converts single image to batch of 1
        return transf(data).unsqueeze(0)

    def _get_top_k(self, logits, k=5):
        probs = F.softmax(logits, dim=1)
        probs5 = probs.topk(k)
        return tuple((p,c, self.idx2label[c]) for p, c in zip(probs5[0][0].detach().numpy(), probs5[1][0].detach().numpy()))

    def __call__(self, x, k=5):
        return self._prediction(x, k)

    def _prediction(self, x, k):
        x = self.transform_input_data(x)
        logits = self.model(x)
        return self._get_top_k(logits, k)

    def preprocess_transform(self, data):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])     
        transf = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            normalize
        ])    
        return transf(data)

    def batch_predict(self, images):
        '''Predicting batch of images, called by LIME.
        '''
        batch = torch.stack(tuple(self.preprocess_transform(Image.fromarray(i)) for i in images), dim=0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        batch = batch.to(device)

        logits = self.model(batch)
        probs = F.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()

    def device(self, device):
        self.model.to(device)


class ModelInterpretation:
    '''Wrapper for LIME model 
    '''
    def __init__(self):
        self.explainer = lime_image.LimeImageExplainer()
        self.explanation = None

    def __call__(self, x, batch_predict, batch_size=32):
        info('Running LIME model.')
        
        self.explanation = self.explainer.explain_instance(
            x, 
            batch_predict, # classification function
            top_labels=5, 
            hide_color=0, 
            batch_size=batch_size,
            num_samples=1000) # number of images that will be sent to classification function

    def get_classes(self):
        return self.explanation.top_labels

    def mark_boundaries(self, label_num=None, positive_only=False, num_features=10):
        '''Draws boundaries of a labels.
        Args:
            label_num: label to explain
            positive_only: if True, only take superpixels that positively
                contribute to the prediction of the label
            num_features: number of superpixels to include in explanation
        '''
        if label_num == None:
            label_num = self.explanation.top_labels[0]

        temp, mask = self.explanation.get_image_and_mask(
            label_num,
            positive_only=positive_only, 
            num_features=num_features, 
            hide_rest=False)
        img_boundry = mark_boundaries(temp/255.0, mask)

        return img_boundry