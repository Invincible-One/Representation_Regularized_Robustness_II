import torch

from models import model_config



class Runner:
    def __init__(
            self,
            args,
            model,
            extended_loss,
            optimizer,
            scheduler,
            repr_module,
            device,
            ):
        self.model = model
        self.extended_loss = extended_loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.model_name = args.model

        if args.pretrained:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.fc.parameters():
                param.requires_grad = True

        self.representation = None
        exec(f"self.handle = self.model.{repr_module}.register_forward_hook(self._hook_fn)")

        self.reset_info()

    def _hook_fn(self, module, input, output):
        self.representation = output

    def remove_hook(self):
        self.handle.remove()

    def run_epoch(self, dataloader, train):
        if train:
            self.model.train()
        else:
            self.model.eval()
        
        self.representation = None
        total_loss = 0.0
        correct = 0
        total = len(dataloader.dataset)
        
        with torch.set_grad_enabled(train):
            for batch_idx, (X, y) in enumerate(dataloader):
                X, y = X.to(self.device), y.to(self.device)
                if model_config[self.model_name]["input_type"] == "flattened":
                    X = X.view(X.size(0), -1)
                
                if train:
                    self.optimizer.zero_grad()
                
                outputs = self.model(X)

                assert self.representation is not None
                loss_v = self.extended_loss(outputs, y, self.representation)
                total_loss += loss_v.item()
                
                if train:
                    loss_v.backward()
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
                
                _, predicted = torch.max(outputs, 1)
                correct_predictions =  (predicted == y).sum().item()
                correct += correct_predictions

                #if train:
                    #batch_acc = correct_predictions / X.shape[0]
                    #print(f"Batch [{batch_idx}], acc {batch_acc:.4f}, {loss_v.item():.4f}")
        self.extended_loss.display(epoch_counter=self.epoch_counter, split=self.split)
        return correct / total, total_loss / total

    def reset_info(self):
        self.epoch_counter = 0
        self.split = None

    def update_info(self, epoch_counter, split):
        self.epoch_counter = epoch_counter
        self.split = split
