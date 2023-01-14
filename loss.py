class LossComputer:
    def __init__(self, criterion, is_robust, n_groups, group_counts, robust_step_size, stable=True,
                 size_adjustments=None, auroc_version=False, class_map=None, use_cuda=False):
        self.criterion = criterion
        self.is_robust = is_robust
        self.auroc_version = auroc_version
        self.n_groups = n_groups
        
        device = torch.device("cuda")
        #group_idx = group_idx.to(device)
        #group_range  = group_range
        
        
        if auroc_version:
            assert (class_map is not None)
            self.n_gdro_groups = len(class_map[0]) * len(class_map[1])
            self.class_map = class_map
        else:
           
            self.n_gdro_groups = n_groups
        self.group_range = torch.arange(self.n_groups).unsqueeze(1).long()
        
        #self.group_range = self.group_range.to(device)
        print(self.group_range)
        self.group_range = self.group_range.to(device)
        
        
        if self.is_robust:
            self.robust_step_size = robust_step_size           
            logging.info(f'Using robust loss with inner step size {self.robust_step_size}')
            self.stable = stable
            self.group_counts = group_counts.to(self.group_range.device)

            if size_adjustments is not None:
              
                self.do_adj = True
                if auroc_version:
                    self.adj = torch.tensor(size_adjustments[0]).float().to(self.group_range.device)
                    self.loss_adjustment = self.adj / torch.sqrt(self.group_counts[:-1])
                else:
                    self.adj = torch.tensor(size_adjustments).float().to(self.group_range.device)
                    
                    self.loss_adjustment = self.adj / torch.sqrt(self.group_counts)
                  
            else:
                self.adj = torch.zeros(self.n_gdro_groups).float().to(self.group_range.device)
                self.do_adj = False
                self.loss_adjustment = self.adj

            logging.info(
                f'Per-group loss adjustments: {np.round(self.loss_adjustment.tolist(), 2)}')
            # The following quantities are maintained/updated throughout training
            if self.stable:
                logging.info('Using numerically stabilized DRO algorithm')
                self.adv_probs_logits = torch.zeros(self.n_gdro_groups).to(self.group_range.device)
                # print(self.adv_probs_logits)
            else:  # for debugging purposes
                logging.warn('Using original DRO algorithm')
                self.adv_probs = torch.ones(self.n_gdro_groups).to(
                    self.group_range.device) / self.n_gdro_groups
        else:
            logging.info('Using ERM')

    def loss(self, yhat, y, group_idx=None, mode =None,w=None,epoch=None, is_training=False):
        # compute per-sample and per-group losses
        per_sample_losses = self.criterion(yhat, y)
        batch_size = y.shape[0]

        group_losses, group_counts = self.compute_group_avg(per_sample_losses, group_idx)
        corrects = (torch.argmax(yhat, 1) == y).float()
        # print('corrects:------------')
        # print(corrects)
        #group_accs, group_counts = self.compute_group_avg(corrects, group_idx)

        # compute overall loss
        
        if self.is_robust:
            
            if self.auroc_version:
                neg_subclasses, pos_subclasses = self.class_map[0], self.class_map[1]
                pair_losses = []
                for neg_subclass in neg_subclasses:
                    neg_count = group_counts[neg_subclass]
                    neg_sbc_loss = group_losses[neg_subclass] * neg_count
                    for pos_subclass in pos_subclasses:
                        pos_count = group_counts[pos_subclass]
                        pos_sbc_loss = group_losses[pos_subclass] * pos_count
                        tot_count = neg_count + pos_count
                        tot_count = tot_count + (tot_count == 0).float()
                        pair_loss = (neg_sbc_loss + pos_sbc_loss) / tot_count
                        pair_losses.append(pair_loss)
                loss, _ = self.compute_robust_loss(torch.cat([l.view(1) for l in pair_losses]))
            else:
                
                loss, _ = self.compute_robust_loss(group_losses,mode,w = w,epoch=epoch)
        else:
            loss = per_sample_losses.mean()

        return loss, (per_sample_losses, corrects)

    def compute_robust_loss(self, group_loss,mode,w,epoch):

        if torch.is_grad_enabled():  # update adv_probs if in training mode
            # print ('group_loss:------')
            # print(group_loss)
            adjusted_loss = group_loss
            #print(adjusted_loss)
            if self.do_adj:
                # print("do_adj, loss_adjustment:-------")
                # print(self.loss_adjustment)
                adjusted_loss += self.loss_adjustment
            logit_step = self.robust_step_size * adjusted_loss.data
            # print('adjusted_loss.data:-------')
            # print(adjusted_loss.data)
            # print('self.robust_step_size:-------')
            # print(self.robust_step_size)           
            if self.stable:
                # print('stable, adv_probs_logits BEFORE Logist_step:------')
                # print(self.adv_probs_logits)
                # print('stable, logit_step:------')
                # print(logit_step)
                self.adv_probs_logits = self.adv_probs_logits + logit_step
                # print('stable, adv_probs_logits after Logist_step:------')
                # print(self.adv_probs_logits)
            else:
                # print('self.adv_probs:-------')
                # print(self.adv_probs)
                self.adv_probs = self.adv_probs * torch.exp(logit_step)
                self.adv_probs = self.adv_probs / self.adv_probs.sum()

        if self.stable:
            adv_probs = torch.softmax(self.adv_probs_logits, dim=-1)
            #print('adv_probs:-----------')
            #print(adv_probs)
        else:
            adv_probs = self.adv_probs
        
        robust_loss = group_loss @ adv_probs
        
            
        return robust_loss, adv_probs

    def compute_group_avg(self, losses, group_idx, num_groups=None, reweight=None):
        # compute observed counts and mean loss for each group
        
        
        device = torch.device("cuda")
        #num_groups = num_groups.to(device)
        if num_groups is None:
            group_range = self.group_range
        else:
            group_range = torch.arange(num_groups).unsqueeze(1).long()

        # print('group_range is:--------')
        # print(group_range)
        # Reweight is None?!
        if reweight is not None:           
            group_loss, group_count = [], []
            reweighted = losses * reweight
            for i in range(num_groups):
                inds = group_idx == i
                group_losses = reweighted[inds]
                group_denom = torch.sum(reweight[inds])
                group_denom = group_denom
                group_loss.append(
                    torch.sum(group_losses) / (group_denom + (group_denom == 0).float()))
                group_count.append(group_denom)
            group_loss, group_count = torch.tensor(group_loss), torch.tensor(group_count)
        else:
            #print('reweight is none')
            
            
            group_map = (group_idx == group_range).float()
            #print(group_map)
            group_count = group_map.sum(1)
            #print('group_count is:--------')
            #print(group_count)
            group_denom = group_count + (group_count == 0).float()  # avoid nans
            #print('group_denom is:--------')
            #print(group_denom)
            group_loss = (group_map @ losses.view(-1)) / group_denom
            #print('losses is:------------')
            #print(losses.view(-1))
            #print('group_loss is:--------')
            #print(group_loss)
            #print("group_loss",group_loss)
            #err_1.append(group_loss[0].detach())
            #err_2.append(group_loss[1].detach())
            #err_3.append(group_loss[2].detach())
            #err_4.append(group_loss[3].detach())
        return group_loss, group_count

    def __call__(self, yhat, y, group_idx,mode,w,epoch):
        return self.loss(yhat, y, group_idx,mode,w,epoch)