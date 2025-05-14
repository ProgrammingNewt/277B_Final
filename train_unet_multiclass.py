import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from pycocotools.coco import COCO
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from image_utils import *
import datetime
import pandas as pd

#global variables

DATA_DIR = 'revamp_tile_output'
ANNOTATION_PATH = 'revamp_tile_output/annotations/revamp_tile_coco.json'
BATCH_SIZE = 24
NUM_CLASSES = 14  #7 tumor + 6 tissue + background
EPOCHS = 50
LR = 1e-4
IMG_SIZE = 256
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = 'unet_revamped_tiles_resnet34.pth'
TEST = False
TEST_SIZE = 500
RESTART = True

class CellSegDataset(Dataset):
    def __init__(self, image_dir, image_ids, coco_path, transform=None):

        self.image_dir = image_dir
        self.coco = COCO(coco_path)
        self.new_image_ids = []
        self.image_list = self.coco.loadImgs(image_ids)
        self.transform = transform

        #make it easy to access file names and category ids
        self.filename_to_id = {img['file_name']: img['id'] for img in self.image_list}
        self.cat_id_to_index = {cat['id']: idx for idx, cat in enumerate(self.coco.loadCats(self.coco.getCatIds()))}
        
        

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        #get filename of image, load the image, and get the image ID for annotation use
        filename = self.image_list[idx]['file_name']
        image_path = os.path.join(self.image_dir, filename.split('_')[0], filename)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        image_id = self.filename_to_id[filename]

        #find the annotations that correspond with the image id
        mask = np.zeros((height, width), dtype=np.uint8)
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        #check to make sure that the annotations have segmentation, then generate a mask
        for ann in anns:
            if 'segmentation' not in ann or not ann['segmentation']:
                continue
        
            cat_idx = self.cat_id_to_index[ann['category_id']]

            #since category indices are zero indexed, ensure that the background is 0 and all categories are shifted up one
            mask[self.coco.annToMask(ann) == 1] = cat_idx + 1

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].long()

        return image, mask


def get_datasets():
    #load annotations
    coco = COCO(ANNOTATION_PATH)
    all_images = coco.getImgIds()
    #test case for quick tests, only load a few images
    if TEST:
        all_images = np.random.choice(all_images, replace = False, size = TEST_SIZE)

    #need to shuffle the dataset since the dataset isn't shuffled to begin with; make sure that it's repeatable
    else:
        #make sure that validation images aren't leaking
        np.random.seed(42)
        np.random.shuffle(all_images)
    split_idx = int(len(all_images) * 0.80)
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]

    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(),
        ToTensorV2()
    ])

    train_ds = CellSegDataset(DATA_DIR, train_images, ANNOTATION_PATH, transform)
    val_ds = CellSegDataset(DATA_DIR, val_images, ANNOTATION_PATH, transform)
    return train_ds, val_ds


def train_one_epoch(model, loader, optimizer, loss_fn, scheduler):
    model.train()
    epoch_loss = 0
    for images, masks in tqdm(loader, desc='Train'):
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        preds = model(images)
        loss = loss_fn(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)

#visualization
def decode_segmentation_mask(mask, num_classes):
    #generates a colored mask from an input class mask
    
    #each class gets a different color
    colors = np.array([
        [0, 0, 0],        
        [255, 0, 0],      
        [0, 255, 0],      
        [0, 0, 255],      
        [255, 255, 0],    
        [255, 0, 255],
        [0, 255, 255],
        [128, 0, 0],
        [0, 128, 0],
        [0, 0, 128],
        [128, 128, 0],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128]
    ], dtype=np.uint8)

    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    tumor_class = []
    for cls in range(num_classes):
        if cls in mask and cls != 0:
            tumor_class.append(cls)
        color_mask[mask == cls] = colors[cls]
    
    return color_mask, tumor_class

def denormalize(image_tensor, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)):
    #reverses augmentation and normalization to form a regular image for visualization

    image = image_tensor.clone().cpu().numpy()
    for i in range(3):
        image[i] = image[i] * std[i] + mean[i]
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    return image.transpose(1, 2, 0)  # CHW -> HWC

def validate(model, loader, loss_fn, num_classes=NUM_CLASSES):
    #evaluates model with loss function
    model.eval()
    epoch_loss = 0
    ious = []

    #set timestamp for filename of visualizations folder
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    valdir = os.path.join(OUTPUT_DIR, 'unet_epochs', f'epoch_{timestamp}')
    os.makedirs(valdir, exist_ok=True)

    #this is the imagenet standard to denormalize
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)
    seed = np.random.choice(50)
    with torch.no_grad():
        for images, masks in tqdm(loader, desc='Val'):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            preds = model(images)
            loss = loss_fn(preds, masks)
            epoch_loss += loss.item()

            preds = torch.argmax(preds, dim=1)
            iou = compute_mean_iou(preds, masks, num_classes)
            ious.append(iou)

            #record every 50 as a bunch of images
            seed += 1
            if seed % 50 == 0:
                visual_preds = []
                for pred in preds:
                    visual_preds.append(pred.squeeze().cpu().numpy())
                images_np = []
                for image in images:
                    image_np = image.permute(1, 2, 0).cpu().numpy()
                    image_np = denormalize(image, mean=imagenet_mean, std=imagenet_std)
                    images_np.append(image_np)

                for image, mask, pred in zip(images_np, masks, visual_preds):
                    pred_mask, pred_class = decode_segmentation_mask(pred, NUM_CLASSES)
                    true_mask, true_class = decode_segmentation_mask(mask.cpu().numpy(), NUM_CLASSES)

                    gt_overlay = cv2.addWeighted(image, 0.6, true_mask, 0.4, 0)
                    pred_overlay = cv2.addWeighted(image, 0.6, pred_mask, 0.4, 0)

                    true_names = set(true_class)
                    pred_names = set(pred_class)
                
                    fig, axs = plt.subplots(1, 2, figsize=(12, 8))
                    axs[0].imshow(gt_overlay)
                    axs[0].set_title(f"GT Overlay: Classes {true_names}")
                    axs[0].axis('off')
                    axs[1].imshow(pred_overlay)
                    axs[1].set_title(f"Pred Overlay: Classes {pred_names}")
                    axs[1].axis('off')

                    plt.tight_layout()
                    save_path = os.path.join(valdir, f"vis_{datetime.datetime.now().strftime('%H-%M-%S')}.png")
                    plt.savefig(save_path)
                    plt.close()

    return epoch_loss / len(loader), np.nanmean(ious)


def compute_mean_iou(preds, targets, num_classes):
    #computes IOU for predictions (accuracy metric)
    ious = []
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (targets == cls)
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union == 0:
            ious.append(np.nan)
        else:
            ious.append(intersection / union)
    return np.nanmean(ious)


def main():

    print("Initializing model")
    #use pretrained model from SMP, either resnet34 or segformer mit_b2
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=NUM_CLASSES)
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters())
    #loss function is CE loss, used to use DICE + CE but not anymore
    loss_fn = nn.CrossEntropyLoss()
    
    #load datasets and make them into DataLoaders
    print("Loading datasets")
    train_ds, val_ds = get_datasets()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = 0.01, steps_per_epoch = len(train_loader), epochs = EPOCHS)

    #if restart flag, load weights from best prev run
    if RESTART:
        checkpoint = torch.load(CHECKPOINT_PATH, weights_only = False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_iou = checkpoint['best_iou']
        best_val_loss = checkpoint['best_val_loss']

    #otherwise, set val loss and iou to default values
    else:
        best_iou = 0
        best_val_loss = 1E8
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    global OUTPUT_DIR
    OUTPUT_DIR = os.path.join('./unet_models', f'model_{timestamp}')
    os.makedirs(OUTPUT_DIR)

    train_losses = []
    val_losses = []
    ious = []
    for epoch in range(EPOCHS):

        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, scheduler)
        val_loss, val_iou = validate(model, val_loader, loss_fn)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {val_iou:.4f}")
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        ious.append(val_iou)
        if val_loss < best_val_loss:
            best_iou = val_iou
            best_val_loss = val_loss
            torch.save(
                {'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'best_iou' : best_iou,
                 'best_val_loss' : best_val_loss}, CHECKPOINT_PATH)
            print(f"Saved new best model at IoU: {best_iou:.4f}, val loss: {best_val_loss:.4f}")

    df = pd.DataFrame({'train_loss' : train_losses, 'val_loss' : val_losses, 'mean iou' : ious})
    df.to_csv(os.path.join(OUTPUT_DIR, 'results.csv'))
if __name__ == '__main__':
    main()
