import torch.optim.optimizer
from construct_global_utils.del_freqs import *
from construct_global_utils.moment_estimate import *
from construct_global_utils.util import *
import copy


####################### Global Params&Class Def ############################

class SimpleDataset(datasets.VisionDataset):
    def __init__(self, root, transform=None):
        super(SimpleDataset, self).__init__(root, transform=transform)
        self.images = [os.path.join(root, file) for file in os.listdir(root) if os.path.isfile(os.path.join(root, file))]

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.images)

device = 'cpu'
if torch.cuda.is_available():
    device = "cuda:1"

traindir = '/data01/img_net_dataset/train/'

train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize_fun
        ])
    )

#################################################################


def mng_by_label(label):
    temp_dataset = SimpleDataset(train_dataset.root+train_dataset.classes[label]+"/", transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize_fun
        ]))
    data_loader = DataLoader(temp_dataset, batch_size=128, shuffle=False, pin_memory=True)
    return temp_dataset, data_loader


def find_support_freqs(model:nn.Module, l, del_rate):
    model.to(device)
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=0.1)
    label_datasets, label_loader = mng_by_label(l)
    freq_all = None
    for batch, X in enumerate(label_loader):
        model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
        X = X.to(device)
        with torch.no_grad():
            freq_origin = torch.fft.fft2(X)
        X.requires_grad = True 
        pred = model(X)
        pred_label = pred.argmax(-1)
        optimizer.zero_grad()
        loss = loss_fn(pred, pred_label)
        loss.backward()
        X_grad = X.grad.clone()
        
        with torch.no_grad():
            X_new = X * (X_grad)
            freq_new = torch.fft.fft2(X_new,dim=(-2,-1))
            freq_en_origin = torch.abs(freq_origin)
            freq_en_new = torch.abs(freq_new)
            freq_score = freq_en_origin - freq_en_new
            if freq_all is None:
                freq_all = torch.abs(freq_origin)
            else:
                freq_all = torch.concat([freq_all, torch.abs(freq_origin)], dim=0)
            """mask = construct_masks(freq_score, 0.6, False)
            X = torch.fft.fft2(X)*mask
            X = torch.fft.ifft2(X).real
            new_pred = model(X).argmax(-1)
            print((new_pred==pred_label).sum().item())"""

        torch.cuda.empty_cache()
    # We test the delete rate in the local explanation experiment
    with torch.no_grad():
        print(freq_all.sum())
        mask = construct_masks(freq_all, del_rate, False)
        mask = torch.sum(mask, dim=0)/mask.shape[0]
        thred = triangle_thred1D(mask)
        mask = torch.where(mask >= thred, 1, 0)
        print(mask.sum())
        torch.cuda.empty_cache()
        select_freqs = mask#moment_filt_score_mag(freq_all, mask, True)
        print("Selected ",select_freqs.sum(), " Total")
        return select_freqs


def val_select_freqs(model:nn.Module, select_freqs:torch.Tensor, l):
    model.to(device)
    model.eval()
    label_datasets, label_loader = mng_by_label(l)
    with torch.no_grad():
        corr = 0
        for batch, X in enumerate(label_loader):
            X = X.to(device)
            ori_pred = model(X).argmax(-1)
            X = torch.fft.fft2(X)
            X = X*select_freqs
            X = torch.fft.ifft2(X)
            X = X.real
            if batch == 0:
                save_img(X[:10],'Val.png')
            pred = model(X).argmax(-1)
            corr += (pred==ori_pred).sum().item()
    acc = corr/len(label_datasets.images)
    with open('del_rate_detect.log','a') as f:
        print(l, select_freqs.sum().item(), acc, corr, file=f)
    return acc


def run():
    res_dict = {}
    for l in range(1000):
        if l <= 359:
            continue
        res_dict[l] = {}
        for del_rate in [0.95,0.9,0.85,0.8,0.75,0.7,0.6]:
            model = resnet50(weights=ResNet50_Weights.DEFAULT)
            selected_freqs = find_support_freqs(model, l, del_rate)
            model = resnet50(weights=ResNet50_Weights.DEFAULT)
            acc = val_select_freqs(model, selected_freqs, l)
            res_dict[l][del_rate] = acc
            if acc > 0.89:
                break
    with open('del_rate_detect.txt','w') as f:
        print(res_dict, file=f)

    with open('del_rate_detect.json','w', encoding='UTF-8') as f:
        json.dump(res_dict, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    run()

