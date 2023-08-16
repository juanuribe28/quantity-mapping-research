### Import Packages
from typing import Iterator, List, Dict

import torch
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

from allennlp.data import Instance
from allennlp.data.fields import TextField, ArrayField

from allennlp.data.dataset_readers import DatasetReader

from allennlp.common.file_utils import cached_path

from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

from allennlp.data.vocabulary import Vocabulary

from allennlp.models import Model

from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.nn.util import get_text_field_mask

from allennlp.training.metrics import BooleanAccuracy

from allennlp.data.iterators import BucketIterator

from allennlp.training.trainer import Trainer

### Set random Seed
torch.manual_seed(1)

### Read Data
class PosDatasetReader(DatasetReader):
    
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def text_to_instance(self, tokens_user: List[Token], tokens_database: List[Token], match: List[int] = None) -> Instance:
        user_field = TextField(tokens_user, self.token_indexers)
        database_field = TextField(tokens_database, self.token_indexers)
        fields = {"user": user_field, "database": database_field}

        if match:
            match_field = ArrayField(np.array([int(match)]))
            fields["match"] = match_field

        return Instance(fields)
    
    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as f:
            for line in f:
                user, database, match = line.split(',')
                yield self.text_to_instance([Token(word) for word in user], [Token(word) for word in database], match)
                
reader = PosDatasetReader()

train_path = 'data/s_train_dataset.csv' #train csv without headers
test_path = 'data/s_test_dataset.csv' #test csv without headers

train_dataset = reader.read(cached_path(train_path))
validation_dataset = reader.read(cached_path(test_path))

vocab = Vocabulary.from_instances(train_dataset+validation_dataset)

### Build Model
class Net(Model):
    
    def __init__(self, word_embeddings: TextFieldEmbedder, encoder: Seq2SeqEncoder, vocab: Vocabulary, padding: int) -> None:
        super().__init__(vocab)
        
        self.padding = padding
        
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        
        self.conv1_u = torch.nn.Conv1d(in_channels = encoder.get_output_dim(), out_channels = , kernel_size = 3)
        self.conv1_db = torch.nn.Conv1d(in_channels = encoder.get_output_dim(), out_channels = 1, kernel_size = 3)
        
        self.linear_u = torch.nn.Linear(in_features = self.conv1_u.get_output_dim(), out_features = 1)        
        self.linear_db = torch.nn.Linear(in_features = self.conv1_db.get_output_dim(), out_features = 1)
        
        self.bilinear = torch.nn.Bilinear(in1_features = self.padding, in2_features = self.padding, out_features = 1)
                        
        self.accuracy = BooleanAccuracy()
        
        self.loss = torch.nn.BCELoss()

    def forward(self, user: Dict[str, torch.Tensor], database: Dict[str, torch.Tensor], match: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        
#        print('')
#        print('User Tensor: {}'.format(user['tokens'].size()))
#        print('Database Tensor: {}'.format(database['tokens'].size()))
        
        batch_size = user['tokens'].size()[0]
        
        user_pad = torch.zeros(batch_size, self.padding).long()#.cuda()
        database_pad = torch.zeros(batch_size, self.padding).long()#.cuda()
        
        user_pad[:,:user['tokens'].size()[1]] = user['tokens']
        database_pad[:,:database['tokens'].size()[1]] = database['tokens']

        user['tokens'] = user_pad
        database['tokens'] = database_pad
        
#        print('User Pad: {}'.format(user['tokens'].size()))
#        print('Database Pad: {}'.format(database['tokens'].size()))
        
        user_mask = get_text_field_mask(user)
        database_mask = get_text_field_mask(database)

#        print('User Mask: {}'.format(user_mask.size()))
#        print('Database Mask: {}'.format(database_mask.size()))

        user_embeddings = F.relu(self.word_embeddings(user))
        database_embeddings = F.relu(self.word_embeddings(database))
        
#        print('User Embeddings: {}'.format(user_embeddings.size()))
#        print('Database Embeddings: {}'.format(database_embeddings.size()))
        
        user_encoder_out = F.relu(self.encoder(user_embeddings, user_mask))
        database_encoder_out = F.relu(self.encoder(database_embeddings, database_mask)) #batch norm after activation
        # add dropout        
#        print('User Encoded: {}'.format(user_encoder_out.size()))
#        print('Database Encoded: {}'.format(database_encoder_out.size()))

        user_conv1 = F.relu(self.conv1_u(user_encoder_out))
        database_conv1 = F.relu(self.conv1_db(database_encoder_out))
        
#        print('User Conv1: {}'.format(user_linear.size()))
#        print('Database Conv1: {}'.format(database_linear.size()))
        
        user_flatten = user_linear.reshape(batch_size,-1)
        database_flatten = database_linear.reshape(batch_size,-1)

#        print('User Flatten: {}'.format(user_flatten.size()))
#        print('Database Flatten: {}'.format(database_flatten.size()))

        bilinear = (self.bilinear(user_flatten, database_flatten)) #take dot product or meanpool
        
        match_prob = torch.sigmoid(bilinear)
        
        match_result = match_prob.clone()
        
        for result in match_result:
            if result[0] < 0.5:
                result[0] = 0
            else:
                result[0] = 1
        
#        print('\n****** Match result: {}'.format(match_result.size()))
#        print('Match size: {}'.format(match.size()))
#        print('')
#        print('**************Prob:{}'.format(match_prob))
#        print('**************Result:{}'.format(match_result))
                
        output = {'match_output':match_result}
        
        if match is not None:
            self.accuracy(match_result, match)
            output['loss'] = self.loss(match_prob, match)
            
        return output
    
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy':self.accuracy.get_metric(reset)}

EMBEDDING_DIM = 64 #between 50 and 300
HIDDEN_DIM = 128  #between 50 and 300

token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),embedding_dim=EMBEDDING_DIM)

word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first = True))

model = Net(word_embeddings, lstm, vocab, 150)

### Check Cuda
if torch.cuda.is_available():
    cuda_device = 0
    model = model.cuda(cuda_device)
else:
    cuda_device = -1

### Train Model

optimizer = optim.SGD(model.parameters(), lr=0.01)

iterator = BucketIterator(batch_size=16, sorting_keys=[("user", "num_tokens"), ("database", "num_tokens")]) #change batch to 16

iterator.index_with(vocab)

trainer = Trainer(model=model, 
                  optimizer=optimizer, 
                  iterator=iterator, 
                  train_dataset=train_dataset, 
                  validation_dataset=validation_dataset, 
                  patience=10, 
                  num_epochs=1000, 
                  cuda_device=cuda_device)

trainer.train()

































































