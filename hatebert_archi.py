import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class HateBertConfig:
    """
    Configuration class for HateBERT model.
    HateBERT uses the same architecture as BERT but with specialized 
    pre-training and fine-tuning for hate speech detection.
    """
    def __init__(
        self,
        vocab_size=30522,  # Same as BERT
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        layer_norm_eps=1e-12,
        # HateBERT specific parameters
        num_labels=2,  # Binary classification (hateful or not)
        hate_threshold=0.5,  # Threshold for determining hate speech
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.layer_norm_eps = layer_norm_eps
        self.num_labels = num_labels
        self.hate_threshold = hate_threshold


class HateBertEmbeddings(nn.Module):
    """
    BERT Embedding layer - same as standard BERT
    """
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
            
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class HateBertSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    Same as standard BERT self-attention layer
    """
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Query, Key, Value linear projections
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # Calculate attention scores - dot product between query and key
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Apply softmax to get attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to value layer
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        return context_layer


class HateBertSelfOutput(nn.Module):
    """
    Output projection after self-attention with residual connection and layer normalization.
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class HateBertAttention(nn.Module):
    """
    Combines self-attention and its output projection.
    """
    def __init__(self, config):
        super().__init__()
        self.self = HateBertSelfAttention(config)
        self.output = HateBertSelfOutput(config)
        
    def forward(self, input_tensor, attention_mask=None):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class HateBertIntermediate(nn.Module):
    """
    Feed-forward network in the transformer encoder.
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        
        # GELU activation function
        if config.hidden_act == "gelu":
            self.intermediate_act_fn = self.gelu
        else:
            self.intermediate_act_fn = F.relu
            
    def gelu(self, x):
        """
        Implementation of the GELU activation function.
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class HateBertOutput(nn.Module):
    """
    Output of feed-forward network with residual connection and layer normalization.
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class HateBertLayer(nn.Module):
    """
    A single transformer encoder layer.
    """
    def __init__(self, config):
        super().__init__()
        self.attention = HateBertAttention(config)
        self.intermediate = HateBertIntermediate(config)
        self.output = HateBertOutput(config)
        
    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class HateBertEncoder(nn.Module):
    """
    Stack of transformer encoder layers.
    """
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([HateBertLayer(config) for _ in range(config.num_hidden_layers)])
        
    def forward(self, hidden_states, attention_mask=None):
        all_encoder_layers = []
        
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            all_encoder_layers.append(hidden_states)
            
        return all_encoder_layers


class HateBertPooler(nn.Module):
    """
    Pool the output of the encoder for sequence classification.
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        
    def forward(self, hidden_states):
        # Use the [CLS] token representation (first token)
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class HateBertModel(nn.Module):
    """
    The main HateBERT model that combines all components.
    Architecturally identical to BERT, but pre-trained on hate speech data.
    """
    def __init__(self, config):
        super().__init__()
        self.embeddings = HateBertEmbeddings(config)
        self.encoder = HateBertEncoder(config)
        self.pooler = HateBertPooler(config)
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
            
        # We create a 3D attention mask from a 2D tensor mask.
        # (batch_size, seq_length) -> (batch_size, 1, 1, seq_length)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoder_outputs = self.encoder(embedding_output, extended_attention_mask)
        
        sequence_output = encoder_outputs[-1]
        pooled_output = self.pooler(sequence_output)
        
        return sequence_output, pooled_output


class HateBertForSequenceClassification(nn.Module):
    """
    HateBERT model for hate speech classification.
    Adds specialized classification head on top of BERT.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = HateBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Classification head - standard binary classification
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        # Hate speech specific layers - optional additional features
        self.hate_intensity = nn.Linear(config.hidden_size, 1)  # Measures intensity of hate speech
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        pooled_output = self.dropout(pooled_output)
        
        # Main classification logits
        logits = self.classifier(pooled_output)
        
        # Optional: Hate intensity score (how severe the hate speech is)
        hate_intensity = torch.sigmoid(self.hate_intensity(pooled_output))
        
        outputs = {
            'logits': logits,
            'hate_intensity': hate_intensity
        }
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            outputs['loss'] = loss
            
        return outputs
    
    def predict(self, input_ids, token_type_ids=None, attention_mask=None):
        """
        Predicts if text contains hate speech and its intensity.
        """
        outputs = self.forward(input_ids, token_type_ids, attention_mask)
        logits = outputs['logits']
        hate_intensity = outputs['hate_intensity']
        
        # Get probabilities
        probs = F.softmax(logits, dim=-1)
        hate_prob = probs[:, 1]  # Probability of hate speech class
        
        # Binary prediction based on threshold
        predictions = (hate_prob > self.config.hate_threshold).int()
        
        return {
            'predictions': predictions,
            'hate_probability': hate_prob,
            'hate_intensity': hate_intensity
        }


class HateBertForTokenClassification(nn.Module):
    """
    HateBERT model for token-level hate speech classification.
    Useful for identifying which specific words/phrases are hateful.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = HateBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only calculate loss on actual tokens (not padding)
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.config.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            
            return loss, logits
        
        return logits


class HateBertForHateExplanation(nn.Module):
    """
    Extended HateBERT model that not only detects hate speech but also 
    provides explanations for why content was flagged as hateful.
    
    This is an advanced variant for explainable AI in content moderation.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = HateBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Main classification head
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        # Explanation head - predicts which category of hate speech
        self.explanation_categories = 6  # Different types of hate speech
        self.explanation_head = nn.Linear(config.hidden_size, self.explanation_categories)
        
        # Attention for token-level contribution to hate speech
        self.token_attention = nn.Linear(config.hidden_size, 1)
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        pooled_output = self.dropout(pooled_output)
        
        # Main hate speech classification
        logits = self.classifier(pooled_output)
        
        # Explanation category classification (e.g., racism, sexism, religious hatred, etc.)
        explanation_logits = self.explanation_head(pooled_output)
        
        # Token-level attention for explaining which tokens contributed to the hate speech
        token_attention_scores = self.token_attention(sequence_output)
        token_attention_probs = F.softmax(token_attention_scores, dim=1)
        
        outputs = {
            'logits': logits,
            'explanation_logits': explanation_logits,
            'token_attention': token_attention_probs
        }
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            outputs['loss'] = loss
            
        return outputs
    
    def explain_prediction(self, input_ids, tokenizer, token_type_ids=None, attention_mask=None):
        """
        Provides a human-readable explanation of why content was flagged.
        """
        outputs = self.forward(input_ids, token_type_ids, attention_mask)
        
        # Get main prediction
        logits = outputs['logits']
        probs = F.softmax(logits, dim=-1)
        is_hate = torch.argmax(probs, dim=-1) == 1
        
        # Get explanation category
        explanation_logits = outputs['explanation_logits']
        explanation_probs = F.softmax(explanation_logits, dim=-1)
        category_id = torch.argmax(explanation_probs, dim=-1)
        
        # Map category ID to category name (would be defined elsewhere)
        category_names = ["racist", "sexist", "religious hatred", "homophobic", "transphobic", "other"]
        category = category_names[category_id]
        
        # Get most important tokens
        token_attention = outputs['token_attention'].squeeze(-1)
        
        # Get top 5 most attended tokens
        _, top_indices = torch.topk(token_attention, min(5, token_attention.size(-1)))
        
        # Convert to tokens using provided tokenizer
        tokens = [tokenizer.convert_ids_to_tokens(input_ids[0][i].item()) for i in top_indices[0]]
        
        # Create human-readable explanation
        explanation = {
            'is_hate_speech': bool(is_hate.item()),
            'confidence': float(probs[0][1].item()),
            'category': category,
            'most_offensive_tokens': tokens
        }
        
        return explanation


def load_pretrained_hatebert():
    """
    Initialize HateBERT with pre-trained weights.
    In practice, these would be loaded from a checkpoint.
    """
    config = HateBertConfig()
    model = HateBertForSequenceClassification(config)
    # In practice, you would load pre-trained weights here
    # model.load_state_dict(torch.load('hatebert_weights.bin'))
    return model


def create_hatebert_explainable():
    """
    Creates an explainable HateBERT model for hate speech detection
    with rationales and explanations.
    """
    config = HateBertConfig(
        # Standard BERT base configuration  
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        # HateBERT specific - multiple label categories
        num_labels=2  # Binary classification: hateful or not
    )
    
    model = HateBertForHateExplanation(config)
    return model