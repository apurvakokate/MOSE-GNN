import torch
from torch_geometric.data import InMemoryDataset, Data
# from utils.utils import to_tudataset
from DataLoader import build_graph
# from utils.utils import get_mol, sanitize_mol
from tqdm import tqdm
# from utils.bridge import bridge_list
from collections import defaultdict
import pdb
import torch.nn.functional as F

# HeterDataset("heter_data/"+data_name, train_list, data_name, motif_list, target_model)
class HeterDataset(InMemoryDataset):
    # def __init__(self, root, dataset, data_name, data_smiles, model, transform=None, pre_transform=None, pre_filter=None) -> None:
    def __init__(self, root, dataset, motif_list, model, transform=None, pre_transform=None, pre_filter=None) -> None:
        self.dataset = dataset
        # self.data_smiles = data_smiles
        self.motif_list = motif_list
        self.model = model
        # if len(self.dataset) != len(self.data_smiles):
        #     print("Error!")
        super().__init__(root, transform, pre_transform, pre_filter)
        # torch.load(self.processed_paths[0])
        # ✅ Correct way to load processed data
        self.data, self.slices = torch.load(self.processed_paths[0])
        
        
    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def process(self):
        # Read data into huge `Data` list.
        data_list = []

        x = []
        edge_index = []

        num_motif = len(self.motif_list)

        for motif in self.motif_list:

            data = build_graph(motif, None, None)

            device = next(self.model.parameters()).device

            batch = torch.zeros(data.x.size(0), dtype=torch.int64, device=device)
            data = data.to(device)  

            logit, _ = self.model(data.x, data.edge_index, batch)
            embedding = self.model.readout_output
            x.append(embedding)
            
        label_0 = []
        label_1 = []
            

        for i,data in enumerate(tqdm(self.dataset)):
            # motifs = motif_list[i].keys()
            motifs = data.nodes_to_motifs

            device = next(self.model.parameters()).device

            batch = torch.zeros(data.x.size(0), dtype=torch.int64, device=device)
            data = data.to(device) 
            logit, _ = self.model(data.x, data.edge_index, batch)
            embedding = self.model.readout_output
                              
            pred = F.sigmoid(logit)

            if pred[0, 0] > 0.5:
                label_0.append(i+num_motif)
            else:
                label_1.append(i+num_motif)
             
                
            x.append(embedding)
            # for motif in motifs:
            for motif_id in motifs:
                if motif_id != -1:
                    edge_index.append((motif_id, i+num_motif))

        x = torch.stack(x)
        x = x.squeeze(dim=1)
        edge_index = torch.tensor(edge_index).t()
        print(len(label_0), len(label_1))
        label_0 = torch.tensor(label_0)
        label_1 = torch.tensor(label_1)
        heter_data = Data(x, edge_index, label_0=label_0, label_1=label_1, motif_vocab=self.motif_list)
        data_list.append(heter_data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # torch.save(data_list, self.processed_paths[0])
        # ✅ Correct way to save
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
                                
 