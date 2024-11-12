class CONCH(CustomCLIP):
    def __init__(self, device):
        from pt_models.conch.open_clip_custom import create_model_from_pretrained
        from pt_models.conch.open_clip_custom import get_tokenizer

        self.conch_image_normalization_stats = {
            "mean": [0.48145466, 0.4578275, 0.40821073],
            "std": [0.26862954, 0.26130258, 0.27577711]
        }
        self.image_normalization = transforms.Normalize(mean=self.conch_image_normalization_stats['mean'], std=self.conch_image_normalization_stats['std'])

        self.model, self.processor = create_model_from_pretrained(model_cfg='conch_ViT-B-16', 
                                                                  checkpoint_path='pt_models/conch/pytorch_model.bin',
                                                                  device=device,
                                                                  force_image_size=224)
        self.model.eval()
        self.tokenizer = get_tokenizer()
        self.device = device

    def encode_images_for_visual_tasks(self, image_batch):
        '''
        Produces image embeddings before the projection head and normalization, suitable for linear probe or working with WSIs under the multiple-instance learning framework. 
        '''
        image_batch = self.image_normalization(image_batch)
        conch_image_feats_for_visual_tasks = self.model.encode_image(image_batch, proj_contrast=False, normalize=False)
        return conch_image_feats_for_visual_tasks

    def convert_encoded_visual_features_for_retrieval_tasks(self, conch_image_feats_for_visual_tasks):
        '''
        Converts the image embeddings before the projection head and normalization to the normalized and projected image embeddings, suitable for image-text retrieval tasks. 
        '''
        conch_image_feats = self.model.visual.forward_project(conch_image_feats_for_visual_tasks)
        conch_image_feats = F.normalize(conch_image_feats, dim=-1)
        return conch_image_feats

    def encode_images(self, image_batch):
        '''
        Produces the projected and normalized image embeddings, suitable for image-text retrieval tasks. 
        '''
        image_batch = self.image_normalization(image_batch)
        conch_image_features = self.model.encode_image(image_batch, proj_contrast=True, normalize=True)
        return conch_image_features
    
    def encode_one_prompt(self, prompts, normalize=True):
        '''
        Encode a class prompt or a list of class prompts with CONCH, and return the average of the embeddings.
        '''
        from pt_models.conch.open_clip_custom import tokenize
        
        assert type(prompts) in [list, str]
        if type(prompts) == str:
            prompts = [prompts]

        class_prompt_tokens = tokenize(self.tokenizer, prompts).to(self.device)
        class_prompt_feats = self.model.encode_text(class_prompt_tokens)
        class_prompt_feats = class_prompt_feats.mean(dim=0)
        if normalize:
            class_prompt_feats = F.normalize(class_prompt_feats, dim=-1)
        return class_prompt_feats
    
    def get_similarity_logits(self, image_features, prompt_features):
        similarity = image_features @ prompt_features.T * self.model.logit_scale.exp()
        return similarity